from typing import Dict, List, Optional
from pathlib import Path
import logging
import math
import os
import re
import unicodedata
from rich.logging import RichHandler
from rich.traceback import install
from difflib import SequenceMatcher

from ..infrastructure.services.odoo_connection_manager import MIN_PRODUCT_CONFIDENCE, odoo_manager
from .models import InvoiceData, InvoiceLine, ProcessedProduct, InvoiceResponseModel
from ..infrastructure.services.ocr import InvoiceOcrClient
from ..tools.odoo_tools import _resolve_product_by_term

install()
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[RichHandler(rich_tracebacks=True, markup=True)],
    )

logger = logging.getLogger(__name__)
ocr_client = InvoiceOcrClient()

def _truthy_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _apply_line_discounts(invoice: InvoiceData) -> InvoiceData:
    """Aplica descuentos por línea (% o monto) recalculando subtotal y precio unitario neto."""
    adjusted_lines: List[InvoiceLine] = []
    for line in invoice.lines:
        pct = line.descuento_pct or 0.0
        monto = line.descuento_monto or 0.0
        if pct or monto:
            factor = max(0.0, 1.0 - (pct / 100.0))
            subtotal_desc = line.precio_unitario * line.cantidad * factor
            subtotal_desc = max(subtotal_desc - monto, 0.0)
            precio_neto = subtotal_desc / line.cantidad if line.cantidad else line.precio_unitario
            adjusted_lines.append(
                line.model_copy(
                    update={
                        "subtotal": subtotal_desc,
                        "precio_unitario": precio_neto,
                    }
                )
            )
        else:
            adjusted_lines.append(line)
    return invoice.model_copy(update={"lines": adjusted_lines})


def _fmt_number(value: float | None) -> str:
    if value is None:
        return "N/D"
    try:
        num = float(value)
    except (TypeError, ValueError):
        return str(value)
    if abs(num - round(num)) < 0.01:
        return f"{int(round(num)):,}".replace(",", ".")
    return f"{num:.2f}"


def _format_invoice_overview(
    invoice: InvoiceData,
    image_path: str | None,
    supplier_id: Optional[int] = None,
    line_skus: Optional[List[Optional[str]]] = None,
) -> str:
    filename = Path(image_path).name if image_path else "factura"
    lines: List[str] = [
        f"Leí la **{filename}** (solo tabla) y comparé en **modo lectura** contra Odoo.",
        "",
        "**Proveedor (factura → Odoo)**",
        f"- {invoice.supplier_name or 'N/D'} — RUT **{invoice.supplier_rut or 'N/D'}**",
    ]
    if supplier_id is not None:
        lines.append(f"- Odoo: **supplier_id={supplier_id}** (resuelto OK)")
    lines.extend(
        [
            "",
            "**Totales factura**",
            f"- Descuento global: **{_fmt_number(getattr(invoice, 'descuento_global', 0.0))}**",
            f"- Neto: **{_fmt_number(invoice.neto)}**",
            f"- IVA 19%: **{_fmt_number(invoice.iva_19)}**",
            f"- Total: **{_fmt_number(invoice.total)}**",
            "",
            "**Líneas (Factura → Producto Odoo propuesto (SKU))**",
        ]
    )
    for idx, line in enumerate(invoice.lines, start=1):
        sku = None
        if line_skus and idx - 1 < len(line_skus):
            sku = line_skus[idx - 1]
        sku_label = sku if sku else "N/A"
        line_text = (
            f"{idx}) {line.detalle} → **{sku_label}** "
            f"(Cant {_fmt_number(line.cantidad)}, P.Unit {_fmt_number(line.precio_unitario)}, "
            f"Subtotal {_fmt_number(line.subtotal)})"
        )
        if not sku:
            line_text = f"{line_text} ← **sin mapeo**"
        lines.append(line_text)
    return "\n".join(lines)


def _build_pre_approval_summary(
    invoice: InvoiceData,
    image_path: str | None,
    order: Optional[dict],
    supplier_id: Optional[int] = None,
    issue_count: int = 0,
    header_mismatch: bool = False,
    ocr_warning: Optional[str] = None,
    line_skus: Optional[List[Optional[str]]] = None,
) -> str:
    parts = [
        _format_invoice_overview(
            invoice, image_path, supplier_id=supplier_id, line_skus=line_skus
        ),
        "",
        "**Riesgo/impacto**",
    ]
    if order:
        order_name = order.get("name") or "OC"
        order_state = order.get("state") or "desconocido"
        parts.append(f"- OC detectada: **{order_name}** (estado {order_state}).")
    else:
        parts.append(
            "- **No se encontró una OC coincidente**; si continuamos, el flujo "
            "**creará una nueva OC**, luego **recepcionará** y **creará la factura**."
        )
    if issue_count:
        parts.append(f"- Hay **{issue_count}** línea(s) con diferencias vs Odoo.")
    if header_mismatch:
        parts.append("- Cabecera no coincide con la factura.")
    if ocr_warning:
        parts.append(f"- Advertencia OCR: {ocr_warning}.")
    missing_details = [
        line.detalle
        for idx, line in enumerate(invoice.lines)
        if not (line_skus and idx < len(line_skus) and line_skus[idx])
    ]
    parts.extend(
        [
            "",
            "Antes de escribir en Odoo:",
        ]
    )
    if missing_details:
        missing_label = ", ".join(f"\"{detalle}\"" for detalle in missing_details)
        example_detail = missing_details[0]
        parts.append(
            f"1) Para {missing_label}, dime el **SKU** o el producto correcto en Odoo "
            f"(ej: \"Cambiar {example_detail} por MP0XX\")."
        )
        parts.append(
            "2) Con eso listo, **confirmo y continúo** (crear OC + recepcionar + crear factura)?"
        )
    else:
        parts.append(
            "1) Si todo está correcto, **confirmo y continúo** (crear OC + recepcionar + crear factura)?"
        )
    return "\n".join(parts)


def _resolve_invoice_type_field() -> str:
    try:
        fields = odoo_manager._execute_kw(
            "account.move", "fields_get", [], {"attributes": ["type"]}
        )
    except Exception:
        return "move_type"
    if "move_type" in fields:
        return "move_type"
    if "type" in fields:
        return "type"
    return "move_type"


def _finalize_and_post_order(
    order: dict,
    folio: str | None = None,
    fecha_emision: str | None = None,
) -> str:
    order_id = odoo_manager._normalize_id(order.get("id"))
    if order_id is None:
        raise RuntimeError("No se pudo resolver el id de la OC.")
    order_state = order.get("state")
    if order_state in {"draft", "sent"}:
        order = odoo_manager.confirm_purchase_order(order_id)
    elif order_state == "cancel":
        raise RuntimeError("La OC esta cancelada. No se puede continuar.")

    odoo_manager.confirm_order_receipt(order)
    order_name = order.get("name")
    picking_ids = set(odoo_manager._normalize_id_list(order.get("picking_ids", [])))
    if order_id is not None:
        try:
            ids_by_po = odoo_manager._execute_kw(
                "stock.picking",
                "search",
                [[["purchase_id", "=", order_id]]],
            )
            picking_ids.update(odoo_manager._normalize_id_list(ids_by_po or []))
        except Exception:
            pass
    if order_name:
        try:
            ids_by_origin = odoo_manager._execute_kw(
                "stock.picking",
                "search",
                [[["origin", "=", order_name]]],
            )
            picking_ids.update(odoo_manager._normalize_id_list(ids_by_origin or []))
        except Exception:
            pass

    picking_summaries = []
    if picking_ids:
        pickings = odoo_manager._execute_kw(
            "stock.picking",
            "read",
            [list(picking_ids)],
            {"fields": ["id", "name", "state"]},
        )
        pickings = [p for p in pickings if p.get("state") != "cancel"]
        pickings = sorted(
            pickings, key=lambda p: (p.get("name") or "", p.get("id") or 0)
        )
        for picking in pickings:
            name = picking.get("name") or str(picking.get("id"))
            state = picking.get("state")
            picking_summaries.append((name, state))
            if state != "done":
                raise RuntimeError(
                    f"Picking {name} quedo en estado '{state}'. Se requiere intervencion manual."
                )

    if picking_summaries:
        if len(picking_summaries) == 1:
            name, state = picking_summaries[0]
            pickings_info = f" Recepción realizada: {name} → {state}."
        else:
            lines = [
                f"Recepción {idx} realizada: {name} → {state}."
                for idx, (name, state) in enumerate(picking_summaries, start=1)
            ]
            pickings_info = " " + " ".join(lines)
    else:
        pickings_info = " Sin pickings."

    inv_result = odoo_manager.create_invoice_for_order(
        order_id, ref=folio, invoice_date=fecha_emision
    )
    inv_ids = inv_result.get("invoice_ids", []) if isinstance(inv_result, dict) else []
    posted = inv_result.get("posted", False) if isinstance(inv_result, dict) else False
    inv_info = ""
    if inv_ids:
        inv_info = f" Factura(s): {inv_ids}."
        if posted:
            inv_info += " Publicada."
    return f"OC {order.get('name')}: recepcionada y factura creada.{pickings_info}{inv_info}"


def _apply_split_plan(
    invoice: InvoiceData,
    split_plan: list[dict],
    supplier_id: Optional[int],
) -> tuple[InvoiceData, List[Optional[dict]]]:
    if not split_plan:
        return invoice, []
    def _get_field(obj, name: str):
        if isinstance(obj, dict):
            return obj.get(name)
        return getattr(obj, name, None)
    lines = list(invoice.lines)
    explicit_map: List[Optional[dict]] = [None for _ in lines]

    for plan in split_plan:
        keyword = _get_field(plan, "original_line_keyword")
        new_items = _get_field(plan, "new_items")
        keyword_norm = (keyword or "").strip().lower()
        if not keyword_norm:
            raise ValueError("original_line_keyword es requerido para el desglose.")
        if not new_items:
            raise ValueError(f"new_items vacío para '{keyword}'.")

        matches = [
            idx
            for idx, line in enumerate(lines)
            if keyword_norm in (line.detalle or "").lower()
        ]
        if not matches:
            raise ValueError(f"No se encontro linea que contenga '{keyword}'.")
        if len(matches) > 1:
            detalles = "; ".join(
                f"[{i}] {lines[i].detalle}" for i in matches if i < len(lines)
            )
            raise ValueError(
                f"Se encontraron multiples lineas con '{keyword}': {detalles}."
            )

        target_idx = matches[0]
        original_line = lines[target_idx]
        original_qty = float(original_line.cantidad or 0.0)
        total_new_qty = 0.0
        resolved_items: List[dict] = []

        for item in new_items:
            term = _get_field(item, "product_search_term")
            qty = _get_field(item, "qty")
            if qty is None:
                raise ValueError("Cada item debe incluir qty.")
            qty_value = float(qty)
            if qty_value <= 0:
                raise ValueError("Cada qty en el desglose debe ser > 0.")
            total_new_qty += qty_value
            product = _resolve_product_by_term(term, supplier_id=supplier_id)
            resolved_items.append(
                {
                    "qty": qty_value,
                    "product_id": odoo_manager._normalize_id(product.get("id")),
                    "name": product.get("name") or term,
                    "default_code": (
                        product.get("default_code")
                        if product.get("default_code") not in (False, None, "")
                        else None
                    ),
                }
            )

        if not math.isclose(total_new_qty, original_qty, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError(
                f"La suma de cantidades ({total_new_qty}) no coincide con la linea original ({original_qty})."
            )

        new_lines: List[InvoiceLine] = []
        new_explicit: List[Optional[dict]] = []
        for resolved in resolved_items:
            new_line = original_line.model_copy(
                update={
                    "detalle": resolved["name"],
                    "cantidad": resolved["qty"],
                    "precio_unitario": float(original_line.precio_unitario or 0.0),
                    "subtotal": float(original_line.precio_unitario or 0.0) * resolved["qty"],
                    "default_code": resolved["default_code"],
                }
            )
            new_lines.append(new_line)
            new_explicit.append(
                {
                    "product_id": resolved.get("product_id"),
                    "default_code": resolved.get("default_code"),
                }
            )

        lines = lines[:target_idx] + new_lines + lines[target_idx + 1 :]
        explicit_map = explicit_map[:target_idx] + new_explicit + explicit_map[target_idx + 1 :]

    return invoice.model_copy(update={"lines": lines}), explicit_map


def extract_invoice_data(image_path: str) -> InvoiceData:
    """Ejecuta OCR sobre la imagen y valida que tenga los campos esperados."""
    logger.info(f"Extrayendo datos de la factura {image_path}")
    raw = ocr_client.extract(image_path)
    if isinstance(raw, dict) and raw.get("error"):
        raise ValueError(f"OCR inválido: {raw['error']}")
    return (
        InvoiceData.model_validate_json(raw)
        if isinstance(raw, str)
        else InvoiceData.model_validate(raw)
    )


def _find_purchase_order(invoice: InvoiceData) -> dict | None:
    """Busca una orden/cotización similar en Odoo sin crear ni confirmar."""
    if not invoice.supplier_name:
        raise ValueError("La factura no indica el nombre del proveedor necesario para buscar en Odoo.")
    invoice_details = [line.detalle for line in invoice.lines]
    order = odoo_manager.find_purchase_order_by_similarity(
        invoice.supplier_name,
        invoice_details,
        invoice.total,
        supplier_rut=getattr(invoice, "supplier_rut", None),
    )
    if order:
        logger.info("Orden candidata en Odoo: %s (ID %s)", order.get("name"), order.get("id"))
    return order


def _load_purchase_order(invoice: InvoiceData) -> dict:
    """Busca la orden/cotización más parecida usando proveedor y detalle de productos.
    Si no existe, crea una nueva orden a partir de la factura."""
    created_from_invoice = False
    order = _find_purchase_order(invoice)
    if not order:
        logger.info("No se detectó una orden coincidente; creando una nueva en Odoo.")
        order = odoo_manager.create_purchase_order_from_invoice(invoice)
        created_from_invoice = True

    if not created_from_invoice and order.get("state") in {"draft", "sent"}:
        logger.info("La orden %s está en estado %s. Confirmando…", order.get("name"), order.get("state"))
        order = odoo_manager.confirm_purchase_order(order["id"])

    order["_created_from_invoice"] = created_from_invoice
    return order


def _finalize_order(
    order: dict,
    qty_by_product: Dict[int, float] | None = None,
    folio: str | None = None,
    fecha_emision: str | None = None,
) -> None:
    """Confirma la recepción y crea la factura."""
    status = odoo_manager.get_order_status(order["id"])
    state = status.get("state")
    if state not in {"purchase", "done"}:
        logger.info("Orden %s en estado %s; no se cerrará automáticamente.", order["name"], state)
        return
    logger.info(f"Finalizando orden {order['name']} (estado {state}).")
    picking_ids = status.get("picking_ids") or []
    if picking_ids:
        odoo_manager.confirm_order_receipt({"picking_ids": picking_ids}, qty_by_product=qty_by_product or {})
    else:
        logger.info("La orden %s no tiene recepciones pendientes en Odoo.", order["name"])
    odoo_manager.create_invoice_for_order(order["id"], ref=folio, invoice_date=fecha_emision)
    return




def _build_order_summary(order: dict) -> Dict[str, any]:
    """Obtiene desde Odoo los totales y las líneas de la orden (detalle, cantidad, precios)."""
    order_data = order
    if order.get("id"):
        # Relee la orden para asegurarse de tener montos y referencias actualizadas.
        fresh = odoo_manager.read_order(order["id"])
        if fresh:
            order_data = fresh

    lines = odoo_manager.read_order_lines(order_data.get("order_line", []))
    receipts = odoo_manager.read_receipts_for_order(order_data.get("picking_ids", []))
    return {
        "neto": order_data.get("amount_untaxed", 0.0),
        "iva_19": order_data.get("amount_tax", 0.0),
        "total": order_data.get("amount_total", 0.0),
        "lines": lines,
        "receipts": receipts,
    }


def _match_line(
    invoice_line: InvoiceLine,
    order_lines: List[dict],
    expected_product_id: int | None = None,
    used_line_ids: set[int] | None = None,
) -> dict | None:
    """Encuentra la línea de Odoo que corresponde a la línea de factura."""
    def _normalize(text: str) -> str:
        if not text:
            return ""
        text = text.lower()
        text = re.sub(r"\[[^\]]+\]", " ", text)
        text = "".join(
            ch
            for ch in unicodedata.normalize("NFD", text)
            if unicodedata.category(ch) != "Mn"
        )
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _accept(line: dict) -> dict:
        if used_line_ids is not None and line.get("id") is not None:
            used_line_ids.add(line["id"])
        return line

    def _score_line(line: dict) -> float:
        candidate_raw = (line.get("detalle") or "").strip()
        candidate = _normalize(candidate_raw)
        if not candidate:
            return 0.0
        if candidate_raw.lower() == target_raw.lower():
            return 1.0
        if target and candidate and (target in candidate or candidate in target):
            return 0.95
        return SequenceMatcher(None, target, candidate).ratio()

    target_raw = (invoice_line.detalle or "").strip()
    target = _normalize(target_raw)
    used_line_ids = used_line_ids or set()

    if expected_product_id is not None:
        candidates = [
            line
            for line in order_lines
            if line.get("product_id") == expected_product_id and line.get("id") not in used_line_ids
        ]
        if candidates:
            best_line = None
            best_key = None
            for line in candidates:
                qty_diff = abs(float(line.get("cantidad", 0.0)) - float(invoice_line.cantidad))
                score = _score_line(line)
                key = (qty_diff, -score)
                if best_key is None or key < best_key:
                    best_key = key
                    best_line = line
            if best_line is not None:
                return _accept(best_line)

    best_line = None
    best_score = 0.0
    for line in order_lines:
        if line.get("id") in used_line_ids:
            continue
        score = _score_line(line)
        if score > best_score:
            best_score = score
            best_line = line
        if score >= 0.95:
            return _accept(line)
    if best_line is not None and best_score >= 0.5:
        return _accept(best_line)
    return None


def _verify_line(
    invoice_line: InvoiceLine,
    order_lines: List[dict],
    matched_line: dict | None = None,
) -> ProcessedProduct:
    """Compara cantidad, precio unitario y subtotal de una línea contra lo que devuelve Odoo."""
    order_line = matched_line or _match_line(invoice_line, order_lines)
    if not order_line:
        return ProcessedProduct(
            detalle=invoice_line.detalle,
            cantidad_match=False,
            precio_match=False,
            subtotal_match=False,
            issues="Producto no encontrado en la orden de compra.",
        )

    cantidad_match = abs(order_line["cantidad"] - invoice_line.cantidad) <= 0.01
    precio_match = abs(order_line["precio_unitario"] - invoice_line.precio_unitario) <= 0.01
    subtotal_match = abs(order_line["subtotal"] - invoice_line.subtotal) <= 0.5

    issues = []
    if not cantidad_match:
        issues.append(
            f"CANT: factura {invoice_line.cantidad} vs Odoo {order_line['cantidad']}."
        )
    if not precio_match:
        issues.append(
            f"P. UNITARIO: factura {invoice_line.precio_unitario} vs Odoo {order_line['precio_unitario']}."
        )
    if not subtotal_match:
        issues.append(
            f"Subtotal: factura {invoice_line.subtotal} vs Odoo {order_line['subtotal']}."
        )

    return ProcessedProduct(
        detalle=invoice_line.detalle,
        cantidad_match=cantidad_match,
        precio_match=precio_match,
        subtotal_match=subtotal_match,
        issues=" ".join(issues) if issues else None,
    )


def process_invoice_file(
    image_path: str,
    allow_odoo_write: bool = False,
    split_plan: Optional[list[dict]] = None,
) -> InvoiceResponseModel:
    """
    Orquesta todo el flujo: extrae la factura, localiza/convierte la orden en Odoo,
    compara cabecera y líneas, verifica recepción y construye el resumen final.
    Si allow_odoo_write es False, funciona en modo lectura y solicita aprobacion.
    split_plan permite desglosar lineas antes de crear la OC.
    """
    invoice = extract_invoice_data(image_path)
    ocr_dudoso = getattr(invoice, "ocr_dudoso", False)
    ocr_warning = getattr(invoice, "ocr_warning", None)
    if not invoice.supplier_name or not invoice.supplier_rut:
        ocr_dudoso = True
        warn_missing = "Proveedor o RUT faltante/ilegible en OCR."
        ocr_warning = f"{ocr_warning}; {warn_missing}" if ocr_warning else warn_missing
    invoice = _apply_line_discounts(invoice)

    descuento_global_aplicado = False
    iva_incluido_detectado = False
    # Ajuste por descuento global: prorrateo equitativo en todas las líneas.
    descuento_global = getattr(invoice, "descuento_global", 0.0) or 0.0
    if descuento_global > 0 and invoice.lines:
        partes = len(invoice.lines)
        descuento_por_linea = descuento_global / partes if partes else 0.0
        adjusted_lines: List[InvoiceLine] = []
        for line in invoice.lines:
            nuevo_subtotal = max(line.subtotal - descuento_por_linea, 0.0)
            nuevo_precio = nuevo_subtotal / line.cantidad if line.cantidad else line.precio_unitario
            adjusted_lines.append(
                line.model_copy(
                    update={
                        "subtotal": nuevo_subtotal,
                        "precio_unitario": nuevo_precio,
                    }
                )
            )
        recalculated_neto = sum(l.subtotal for l in adjusted_lines)
        # Verifica que el neto ajustado cuadre con la cabecera; si no, detiene el flujo.
        if abs(recalculated_neto - invoice.neto) > 1.0:
            return InvoiceResponseModel(
                summary=(
                    f"No se pudo aplicar el descuento global correctamente: "
                    f"neto cabecera {invoice.neto} vs neto recalculado {recalculated_neto}."
                ),
                products=[],
                needs_follow_up=True,
                neto_match=False,
                iva_match=False,
                total_match=False,
            )
        iva_ajustado = recalculated_neto * 0.19
        total_ajustado = recalculated_neto + iva_ajustado
        invoice = invoice.model_copy(
            update={
                "lines": adjusted_lines,
                "neto": recalculated_neto,
                "iva_19": iva_ajustado,
                "total": total_ajustado,
            }
        )
        # Si ya prorrateamos descuento, no volver a ajustar por IVA incluido.
        skip_iva_incluido = True
        descuento_global_aplicado = True
    else:
        skip_iva_incluido = False

    # Caso especial: subtotales de líneas vienen con IVA incluido.
    # Si la suma de subtotales ≈ TOTAL y TOTAL ≈ NETO + IVA, recalculamos subtotales netos.
    if not skip_iva_incluido:
        sum_subtotales = sum(line.subtotal for line in invoice.lines)
        if (
            sum_subtotales > 0
            and abs(sum_subtotales - invoice.total) <= 1.0
            and abs((invoice.neto + invoice.iva_19) - invoice.total) <= 1.0
        ):
            factor = invoice.neto / sum_subtotales if sum_subtotales else 1.0
            adjusted_lines: List[InvoiceLine] = []
            for line in invoice.lines:
                subtotal_neto = line.subtotal * factor
                precio_unitario_neto = subtotal_neto / line.cantidad if line.cantidad else line.precio_unitario
                adjusted_lines.append(
                    line.model_copy(
                        update={
                            "subtotal": subtotal_neto,
                            "precio_unitario": precio_unitario_neto,
                        }
                    )
                )
            invoice = invoice.model_copy(update={"lines": adjusted_lines})
            iva_incluido_detectado = True

    # Normaliza líneas si subtotal y precio unitario no son consistentes.
    normalized_lines: List[InvoiceLine] = []
    for line in invoice.lines:
        expected_subtotal = line.precio_unitario * line.cantidad
        if abs(expected_subtotal - line.subtotal) > 1.0 and line.cantidad != 0:
            # Prefiere ajustar subtotal a cantidad x precio; precio_unitario ya es coherente.
            normalized_lines.append(
                line.model_copy(
                    update={
                        "subtotal": expected_subtotal,
                    }
                )
            )
        elif abs(expected_subtotal - line.subtotal) > 0.01 and line.cantidad != 0:
            # Ajuste leve: corrige solo precio para que cuadre con subtotal leído.
            corrected_price = line.subtotal / line.cantidad
            normalized_lines.append(
                line.model_copy(update={"precio_unitario": corrected_price})
            )
        else:
            normalized_lines.append(line)
    invoice = invoice.model_copy(update={"lines": normalized_lines})

    # Ajuste menor: si la suma de líneas difiere poco del neto de cabecera, corrige la última línea para cuadrar.
    sum_lines = sum(l.subtotal for l in invoice.lines)
    neto_diff = invoice.neto - sum_lines
    tolerance = max(20.0, abs(invoice.neto) * 0.001)  # $20 o 0.1% del neto
    if abs(neto_diff) > 0 and abs(neto_diff) <= tolerance and invoice.lines:
        last = invoice.lines[-1]
        new_subtotal = last.subtotal + neto_diff
        new_price = new_subtotal / last.cantidad if last.cantidad else last.precio_unitario
        adjusted = invoice.lines[:-1] + [
            last.model_copy(update={"subtotal": new_subtotal, "precio_unitario": new_price})
        ]
        invoice = invoice.model_copy(update={"lines": adjusted})
        sum_lines = sum(l.subtotal for l in invoice.lines)

    if iva_incluido_detectado:
        header_coherent = abs((invoice.neto + invoice.iva_19) - invoice.total) <= 1.0
        if header_coherent:
            ocr_dudoso = False
            normalize_warning = "Líneas venían con IVA incluido; se normalizaron a valores netos."
            ocr_warning = f"{ocr_warning}; {normalize_warning}" if ocr_warning else normalize_warning

    if descuento_global_aplicado:
        header_coherent = abs((invoice.neto + invoice.iva_19) - invoice.total) <= 1.0
        if header_coherent:
            ocr_dudoso = False
            dg_warning = "Se prorrateó descuento_global en las líneas; montos ajustados."
            ocr_warning = f"{ocr_warning}; {dg_warning}" if ocr_warning else dg_warning

    if not invoice.lines:
        raise ValueError("La factura no contiene líneas de productos para comparar.")

    resolved_supplier_id = odoo_manager._resolve_supplier_id(
        None,
        invoice.supplier_name,
        getattr(invoice, "supplier_rut", None),
    )
    explicit_map: List[Optional[dict]] = []
    if split_plan:
        invoice, explicit_map = _apply_split_plan(invoice, split_plan, resolved_supplier_id)
    pending_products: List[ProcessedProduct] = []
    expected_product_ids: List[Optional[int]] = []
    line_skus: List[Optional[str]] = []
    sku_cache: Dict[int, Optional[str]] = {}
    require_confirmation = _truthy_env("REQUIRE_PRODUCT_CONFIRMATION", default=True)
    for idx, line in enumerate(invoice.lines):
        explicit = explicit_map[idx] if explicit_map and idx < len(explicit_map) else None
        explicit_product_id = (
            odoo_manager._normalize_id(explicit.get("product_id")) if explicit else None
        )
        explicit_default_code = explicit.get("default_code") if explicit else None
        invoice_detail = line.detalle
        mapped_product_id = (
            explicit_product_id
            if explicit_product_id is not None
            else (
                odoo_manager.get_mapped_product_id(invoice_detail, resolved_supplier_id)
                if resolved_supplier_id is not None
                else None
            )
        )
        candidates = odoo_manager.get_product_candidates(
            invoice_detail,
            supplier_id=resolved_supplier_id,
            supplier_name=invoice.supplier_name,
            invoice_line=line,
        )
        best = candidates[0] if candidates else None
        best_score = float(best.get("score") or 0.0) if isinstance(best, dict) else 0.0
        best_default_code = best.get("default_code") if isinstance(best, dict) else None
        if best_default_code in (False, None, ""):
            best_default_code = None
        best_id = odoo_manager._normalize_id(best.get("id")) if isinstance(best, dict) else None
        if explicit_default_code and not best_default_code:
            best_default_code = explicit_default_code
        if explicit_product_id is not None and best_id is None:
            best_id = explicit_product_id
        expected_product_ids.append(mapped_product_id or best_id)
        sku_value = explicit_default_code
        mapped_norm = odoo_manager._normalize_id(mapped_product_id)
        if mapped_norm is not None:
            if mapped_norm in sku_cache:
                sku_value = sku_cache[mapped_norm]
            else:
                sku_value = odoo_manager.get_sku_by_product_id(mapped_norm)
                sku_cache[mapped_norm] = sku_value
        if not sku_value and best_default_code:
            sku_value = best_default_code
        if not sku_value and best_id is not None:
            if best_id in sku_cache:
                sku_value = sku_cache[best_id]
            else:
                sku_value = odoo_manager.get_sku_by_product_id(best_id)
                sku_cache[best_id] = sku_value
        line_skus.append(sku_value)

        needs_review = False
        if explicit_product_id is not None:
            needs_review = False
        elif mapped_product_id is None:
            if require_confirmation:
                needs_review = True
            elif not candidates:
                needs_review = True
            elif best_score < MIN_PRODUCT_CONFIDENCE:
                needs_review = True

        if needs_review:
            pending_products.append(
                ProcessedProduct(
                    detalle=invoice_detail,
                    invoice_detail=invoice_detail,
                    candidate_name=(best.get("name") if isinstance(best, dict) else None),
                    candidate_default_code=best_default_code,
                    supplier_name=invoice.supplier_name,
                    cantidad_match=False,
                    precio_match=False,
                    subtotal_match=False,
                    issues="Confirma el producto correcto antes de continuar.",
                    status="AMBIGUOUS",
                    candidates=candidates,
                )
            )

    if pending_products:
        summary_msg = "Flujo detenido: se requieren decisiones humanas para mapear productos antes de continuar."
        if ocr_warning:
            summary_msg = f"{summary_msg} {ocr_warning}"
        return InvoiceResponseModel(
            summary=summary_msg,
            products=pending_products,
            needs_follow_up=True,
            neto_match=None,
            iva_match=None,
            total_match=None,
            supplier_id=resolved_supplier_id,
            supplier_name=invoice.supplier_name,
            supplier_rut=getattr(invoice, "supplier_rut", None),
            status="WAITING_FOR_HUMAN",
        )
    order = None
    if allow_odoo_write:
        try:
            order = _load_purchase_order(invoice)
        except Exception as exc:
            warning = f"{ocr_warning}. " if ocr_warning else ""
            return InvoiceResponseModel(
                summary=f"No se pudo crear/abrir la orden en Odoo: {warning}{exc}",
                products=[],
                needs_follow_up=True,
                neto_match=False,
                iva_match=False,
                total_match=False,
            )
    else:
        try:
            order = _find_purchase_order(invoice)
        except Exception as exc:
            warning = f"{ocr_warning}. " if ocr_warning else ""
            return InvoiceResponseModel(
                summary=f"No se pudo buscar la orden en Odoo: {warning}{exc}",
                products=[],
                needs_follow_up=True,
                neto_match=False,
                iva_match=False,
                total_match=False,
            )
        if not order:
            summary_msg = _build_pre_approval_summary(
                invoice,
                image_path,
                order=None,
                supplier_id=resolved_supplier_id,
                issue_count=0,
                header_mismatch=False,
                ocr_warning=ocr_warning,
                line_skus=line_skus,
            )
            return InvoiceResponseModel(
                summary=summary_msg,
                products=[],
                needs_follow_up=True,
                neto_match=None,
                iva_match=None,
                total_match=None,
                supplier_id=resolved_supplier_id,
                supplier_name=invoice.supplier_name,
                supplier_rut=getattr(invoice, "supplier_rut", None),
                status="WAITING_FOR_APPROVAL",
            )

    summary = _build_order_summary(order)

    # Preprocesa recepciones solo para validar cantidades
    receipt_by_product: Dict[int, Dict[str, float]] = {}
    for rec in summary["receipts"]:
        product_id = rec.get("product_id")
        if not product_id:
            continue
        info = receipt_by_product.setdefault(product_id, {"quantity": 0.0})
        info["quantity"] = float(info["quantity"]) + float(rec.get("quantity_done", 0.0))

    products: List[ProcessedProduct] = []
    qty_by_product: Dict[int, float] = {}
    used_line_ids: set[int] = set()
    for idx, line in enumerate(invoice.lines):
        expected_product_id = expected_product_ids[idx] if idx < len(expected_product_ids) else None
        matched_line = _match_line(
            line,
            summary["lines"],
            expected_product_id=expected_product_id,
            used_line_ids=used_line_ids,
        )
        product_result = _verify_line(line, summary["lines"], matched_line=matched_line)

        if matched_line and matched_line.get("id"):
            prod_id = matched_line.get("product_id")
            if prod_id is not None:
                norm_pid = odoo_manager._normalize_id(prod_id)
                if norm_pid is not None:
                    qty_by_product[norm_pid] = line.cantidad
            desired_values: Dict[str, float] = {}
            if not product_result.cantidad_match:
                desired_values["product_qty"] = line.cantidad
            if not product_result.precio_match:
                desired_values["price_unit"] = line.precio_unitario
            # Si solo falla el subtotal, ajusta el precio unitario para que coincida con la factura.
            if not product_result.subtotal_match and "price_unit" not in desired_values:
                if line.cantidad != 0:
                    desired_values["price_unit"] = line.subtotal / line.cantidad

            if desired_values and allow_odoo_write:
                odoo_manager.update_order_line(matched_line["id"], desired_values)
                refreshed_line = odoo_manager.read_order_lines([matched_line["id"]])[0]
                product_result.cantidad_match = (
                    abs(refreshed_line["cantidad"] - line.cantidad) <= 0.01
                )
                product_result.precio_match = (
                    abs(refreshed_line["precio_unitario"] - line.precio_unitario) <= 0.01
                )
                product_result.subtotal_match = (
                    abs(refreshed_line["subtotal"] - line.subtotal) <= 0.5
                )
                if product_result.cantidad_match and product_result.precio_match and product_result.subtotal_match:
                    product_result.issues = None
                else:
                    product_result.issues = (
                        product_result.issues
                        or "No se pudieron ajustar los valores de la linea en Odoo."
                    )

            receipt_info = receipt_by_product.get(matched_line.get("product_id"), {})
        else:
            receipt_info = {}

        if receipt_info:
            qty_received = float(receipt_info.get("quantity", 0.0))
            if abs(qty_received - line.cantidad) > 0.01:
                extra = f"Recepción: factura {line.cantidad} vs recibido {qty_received}."
                product_result.issues = f"{product_result.issues or ''} {extra}".strip()

        if not product_result.issues:
            product_result = product_result.model_copy(update={"status": "MATCHED"})
        products.append(product_result)

    if allow_odoo_write:
        odoo_manager.recompute_order_amounts(order["id"])
        # Releer la orden tras las actualizaciones para usar totales y líneas frescos.
        summary = _build_order_summary(order)
    neto_match = abs(summary["neto"] - invoice.neto) <= 1.0
    iva_match = abs(summary["iva_19"] - invoice.iva_19) <= 1.0
    total_match = abs(summary["total"] - invoice.total) <= 1.0

    # Tolerancia de redondeo: si todas las líneas están OK y la diferencia en cabecera es <= 1 peso,
    # consideramos coincidencia de neto, IVA y total.
    header_neto_diff = abs(summary["neto"] - invoice.neto)
    header_iva_diff = abs(summary["iva_19"] - invoice.iva_19)
    header_total_diff = abs(summary["total"] - invoice.total)
    all_lines_ok = all(not p.issues for p in products)
    if all_lines_ok and header_neto_diff <= 1.0 and header_iva_diff <= 1.0 and header_total_diff <= 1.0:
        neto_match = True
        iva_match = True
        total_match = True

    needs_follow_up = any(p.issues for p in products) or not (neto_match and iva_match and total_match)
    if ocr_dudoso:
        needs_follow_up = True

    if not allow_odoo_write:
        issue_count = sum(1 for p in products if p.issues)
        header_mismatch = not (neto_match and iva_match and total_match)
        summary_msg = _build_pre_approval_summary(
            invoice,
            image_path,
            order=order,
            supplier_id=resolved_supplier_id,
            issue_count=issue_count,
            header_mismatch=header_mismatch,
            ocr_warning=ocr_warning,
            line_skus=line_skus,
        )
        return InvoiceResponseModel(
            summary=summary_msg,
            products=products,
            needs_follow_up=True,
            neto_match=neto_match,
            iva_match=iva_match,
            total_match=total_match,
            supplier_id=resolved_supplier_id,
            supplier_name=invoice.supplier_name,
            supplier_rut=getattr(invoice, "supplier_rut", None),
            status="WAITING_FOR_APPROVAL",
        )

    # Construye resumen breve según caso.
    if not needs_follow_up:
        if allow_odoo_write:
            action_summary = (
                "OC creada en Odoo."
                if order.get("_created_from_invoice")
                else "OC actualizada en Odoo."
            )
            try:
                finalize_summary = _finalize_and_post_order(
                    order,
                    folio=invoice.folio,
                    fecha_emision=invoice.fecha_emision,
                )
            except Exception as exc:
                error_summary = (
                    f"{action_summary} No se pudo finalizar el flujo en Odoo: {exc}"
                )
                return InvoiceResponseModel(
                    summary=error_summary,
                    products=products,
                    needs_follow_up=True,
                    neto_match=neto_match,
                    iva_match=iva_match,
                    total_match=total_match,
                    po_name=order.get("name"),
                    po_id=order.get("id"),
                )
            final_summary = f"{action_summary} {finalize_summary}"
            return InvoiceResponseModel(
                summary=final_summary,
                products=products,
                needs_follow_up=False,
                neto_match=neto_match,
                iva_match=iva_match,
                total_match=total_match,
                po_name=order.get("name"),
                po_id=order.get("id"),
            )
        error_summary = "No se pudo crear/editar OC, revisar Odoo manualmente."
        product_issues = [
            ProcessedProduct(
                detalle=p.detalle if hasattr(p, "detalle") else "desconocido",
                cantidad_match=getattr(p, "cantidad_match", False),
                precio_match=getattr(p, "precio_match", False),
                subtotal_match=getattr(p, "subtotal_match", False),
                issues=getattr(p, "issues", None),
            )
            for p in products
        ] or []
        return InvoiceResponseModel(
            summary=error_summary,
            products=product_issues,
            needs_follow_up=True,
            neto_match=neto_match,
            iva_match=iva_match,
            total_match=total_match,
        )

    # Construye resumen detallado de discrepancias.
    discrepancy_parts: List[str] = []
    if ocr_dudoso and ocr_warning:
        discrepancy_parts.append(f"OCR dudoso: {ocr_warning}.")
    elif ocr_warning:
        discrepancy_parts.append(f"Advertencia OCR: {ocr_warning}.")
    if not neto_match:
        discrepancy_parts.append(
            f"Neto no coincide (Factura {invoice.neto} vs Odoo {summary['neto']})."
        )
    if not iva_match:
        discrepancy_parts.append(
            f"IVA 19% no coincide (Factura {invoice.iva_19} vs Odoo {summary['iva_19']})."
        )
    if not total_match:
        discrepancy_parts.append(
            f"Total no coincide (Factura {invoice.total} vs Odoo {summary['total']})."
        )
    for product in products:
        if product.issues:
            discrepancy_parts.append(f"{product.detalle}: {product.issues}")

    if discrepancy_parts:
        discrepancy_msg = "No se pudo procesar la factura: " + " ".join(discrepancy_parts)
    else:
        discrepancy_msg = "No se pudo procesar la factura; revisa manualmente en Odoo."

    return InvoiceResponseModel(
        summary=discrepancy_msg,
        products=products,
        needs_follow_up=True,
        neto_match=neto_match,
        iva_match=iva_match,
        total_match=total_match,
    )
