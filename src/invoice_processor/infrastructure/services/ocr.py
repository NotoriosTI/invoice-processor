import base64
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Union
from PIL import Image
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from ...config import get_settings
from ...prompts import INVOICE_OCR_PROMPT


class InvoiceOcrClient:
    def __init__(self):
        settings = get_settings()
        self.llm = ChatOpenAI(
            model=settings.llm_model,
            openai_api_key=settings.openai_api_key,
            max_tokens=2048,
            model_kwargs={"response_format": {"type": "json_object"}},
        )
        
    @staticmethod
    def _encode_image(image_path: str) -> str:
        data = Path(image_path).read_bytes()
        return base64.b64encode(data).decode("utf-8")

    @staticmethod
    def _crop_invoice(image_path: str) -> str:
        """Recorta la imagen a la porción superior (cabecera) e inferior (tabla) para reducir ruido."""
        img = Image.open(image_path)
        width, height = img.size
        top_box = (0, 0, width, int(height * 0.35))
        bottom_box = (0, int(height * 0.35), width, height)
        header = img.crop(top_box)
        table = img.crop(bottom_box)
        combined_height = header.size[1] + table.size[1]
        combined = Image.new("RGB", (width, combined_height))
        combined.paste(header, (0, 0))
        combined.paste(table, (0, header.size[1]))
        temp_path = Path(image_path).with_suffix(".cropped.jpg")
        combined.save(temp_path, format="JPEG")
        return str(temp_path)

    def _call_llm(self, image_path: str) -> Union[str, Dict[str, Any]]:
        image_b64 = self._encode_image(image_path)
        message = HumanMessage(
            content=[
                {"type": "text", "text": INVOICE_OCR_PROMPT},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                },
            ]
        )
        response = self.llm.invoke([message])
        return response.content

    @staticmethod
    def _to_float(value: Any) -> float | None:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            cleaned = value.replace(" ", "")
            if "," in cleaned and "." in cleaned:
                # Asume formato latino: punto miles, coma decimal.
                cleaned = cleaned.replace(".", "").replace(",", ".")
            elif "," in cleaned:
                # Solo coma: asume coma decimal, quita puntos de miles si hubiera.
                cleaned = cleaned.replace(".", "").replace(",", ".")
            elif cleaned.count(".") >= 1:
                parts = cleaned.split(".")
                if len(parts) > 2:
                    # Varios puntos: trátalos como miles.
                    cleaned = "".join(parts)
                elif len(parts) == 2 and len(parts[1]) in {2}:
                    # Un punto con 2 decimales: asume decimal.
                    cleaned = ".".join(parts)
                elif len(parts) == 2 and len(parts[1]) == 3 and parts[0].isdigit():
                    # Un punto con 3 dígitos: asume miles, quita el punto.
                    cleaned = "".join(parts)
                else:
                    cleaned = cleaned.replace(",", "")
            else:
                cleaned = cleaned.replace(",", "")
            try:
                return float(cleaned)
            except ValueError:
                return None
        return None

    def _postprocess_invoice_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if payload is None:
            return {"error": "no_data"}
        if payload.get("error"):
            return payload
        warnings: List[str] = []

        # Pass-through folio
        folio_raw = payload.get("folio")
        payload["folio"] = str(folio_raw).strip() if folio_raw else None

        # Normalizar fecha_emision a YYYY-MM-DD
        fecha_raw = payload.get("fecha_emision")
        if fecha_raw and isinstance(fecha_raw, str):
            fecha_raw = fecha_raw.strip()
            parsed_date = None
            for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y"):
                try:
                    parsed_date = datetime.strptime(fecha_raw, fmt)
                    break
                except ValueError:
                    continue
            if parsed_date is not None:
                payload["fecha_emision"] = parsed_date.strftime("%Y-%m-%d")
            else:
                warnings.append(f"fecha_emision no parseable: '{fecha_raw}'")
                payload["fecha_emision"] = None
        else:
            payload["fecha_emision"] = None

        def _normalize_unit(unit: Any) -> str | None:
            if not unit:
                return None
            text = str(unit).strip().lower()
            if not text:
                return None
            text = text.replace(".", "").replace(",", "").strip()
            # Mapear sinónimos
            mappings = {
                "kg": ["kg", "kilo", "kilos"],
                "g": ["g", "gr", "gramo", "gramos"],
                "ml": ["ml", "mililitro", "mililitros"],
                "l": ["l", "lt", "litro", "litros"],
                "unidad": ["un", "ud", "uds", "unidad", "unidades"],
                "saco": ["saco", "sacos"],
                "caja": ["caja", "cajas"],
            }
            for norm, variants in mappings.items():
                for v in variants:
                    if text == v:
                        return norm
            return text

        def _round2(val: Any) -> float | None:
            num = self._to_float(val)
            return round(num, 2) if num is not None else None

        payload["neto"] = _round2(payload.get("neto"))
        payload["iva_19"] = _round2(payload.get("iva_19"))
        payload["total"] = _round2(payload.get("total"))
        payload["descuento_global"] = _round2(payload.get("descuento_global") or 0.0) or 0.0

        cleaned_lines: List[Dict[str, Any]] = []
        for line in payload.get("lines") or []:
            cleaned_line = dict(line)
            cleaned_line.pop("warning", None)
            cleaned_line["cantidad"] = _round2(line.get("cantidad"))
            cleaned_line["precio_unitario"] = _round2(line.get("precio_unitario"))
            cleaned_line["unidad"] = _normalize_unit(line.get("unidad"))
            # Recalcula subtotal si difiere > $1 respecto de cantidad x precio.
            # calc_sub es siempre qty * price SIN descuentos; los descuentos se aplican
            # exclusivamente en _apply_line_discounts (processor.py).
            raw_sub = _round2(line.get("subtotal"))
            calc_sub = None
            if cleaned_line["cantidad"] is not None and cleaned_line["precio_unitario"] is not None:
                calc_sub = round(cleaned_line["cantidad"] * cleaned_line["precio_unitario"], 2)
            descuento_monto = _round2(line.get("descuento_monto"))
            descuento_pct = _round2(line.get("descuento_pct"))
            # Calcula el subtotal esperado con descuentos SOLO para validar coherencia del OCR.
            expected_discounted = calc_sub
            if expected_discounted is not None and descuento_pct is not None:
                expected_discounted = max(expected_discounted * (1 - descuento_pct / 100.0), 0.0)
            if expected_discounted is not None and descuento_monto:
                expected_discounted = max(expected_discounted - descuento_monto, 0.0)
            # Si el raw_sub coincide con el valor descontado, el OCR ya reportó el subtotal
            # con descuento incluido; almacenamos calc_sub (sin descuento) para que
            # _apply_line_discounts lo aplique una sola vez.
            has_discount = descuento_pct is not None or bool(descuento_monto)
            raw_matches_discounted = (
                raw_sub is not None
                and expected_discounted is not None
                and abs(expected_discounted - raw_sub) <= 1.0
            )
            if has_discount and raw_matches_discounted:
                # El OCR ya incluyó el descuento en raw_sub; guardamos el subtotal sin descuento.
                cleaned_line["subtotal"] = calc_sub
            elif raw_sub is not None and calc_sub is not None and abs(calc_sub - raw_sub) > 1.0 and not has_discount:
                detail_label = cleaned_line.get("detalle") or line.get("detalle")
                if detail_label:
                    warnings.append(
                        f"subtotal corregido en '{detail_label}' (calc {calc_sub} vs raw {raw_sub})"
                    )
                else:
                    warnings.append(f"subtotal corregido (calc {calc_sub} vs raw {raw_sub})")
                cleaned_line["subtotal"] = calc_sub
            else:
                cleaned_line["subtotal"] = raw_sub if raw_sub is not None else calc_sub
            cleaned_line["descuento_pct"] = descuento_pct
            cleaned_line["descuento_monto"] = descuento_monto
            cleaned_lines.append(cleaned_line)
        payload["lines"] = cleaned_lines

        sum_subtotales = sum((l.get("subtotal") or 0.0) for l in cleaned_lines)
        neto = payload.get("neto")
        if neto is not None:
            diff = abs(sum_subtotales - neto)
            if diff > 500 or diff > neto * 0.02:
                payload["ocr_dudoso"] = True
                warnings.append(f"suma lineas {sum_subtotales} vs neto cabecera {neto}")
            elif diff > 1.0:
                # Duda leve: avisar pero permitir continuar sin bloquear.
                warnings.append(f"suma lineas {sum_subtotales} vs neto cabecera {neto}")
        if warnings:
            payload["ocr_warning"] = " ; ".join(warnings)
        return payload

    def extract(self, image_path: str) -> Union[str, Dict[str, Any]]:
        """
        Intenta parsear JSON dos veces; si ambas fallan, retorna {"error": "..."}.
        """
        attempts = [image_path]
        temp_files: List[str] = []
        try:
            # Si la primera pasada arroja dudoso, recorta y reintenta una vez.
            raw = self._call_llm(image_path)
            try:
                parsed = json.loads(raw) if isinstance(raw, str) else raw
            except json.JSONDecodeError:
                parsed = {"error": "json_invalid"}
            post = self._postprocess_invoice_payload(parsed) if isinstance(parsed, dict) else parsed
            if isinstance(post, dict) and post.get("ocr_dudoso"):
                try:
                    cropped_path = self._crop_invoice(image_path)
                    attempts.append(cropped_path)
                    temp_files.append(cropped_path)
                except Exception:
                    pass

            for idx, attempt_path in enumerate(attempts):
                if idx == 0 and post and not isinstance(post, dict):
                    continue
                if idx > 0:
                    raw = self._call_llm(attempt_path)
                    try:
                        parsed = json.loads(raw) if isinstance(raw, str) else raw
                    except json.JSONDecodeError:
                        parsed = {"error": "json_invalid"}
                    post = self._postprocess_invoice_payload(parsed) if isinstance(parsed, dict) else parsed
                try:
                    if isinstance(post, dict) and post.get("error"):
                        if idx == len(attempts) - 1:
                            return post
                        continue
                    return post
                except Exception:
                    if idx == len(attempts) - 1:
                        return {"error": "no_data"}
                    continue
        finally:
            for tmp in temp_files:
                try:
                    Path(tmp).unlink(missing_ok=True)
                except Exception:
                    pass
