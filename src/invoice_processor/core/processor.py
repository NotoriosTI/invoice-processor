from typing import Dict, List
import logging
import re
import unicodedata
from rich.logging import RichHandler
from rich.traceback import install
from difflib import SequenceMatcher

from ..infrastructure.services.odoo_connection_manager import odoo_manager
from .models import InvoiceData, InvoiceLine, ProcessedProduct, InvoiceResponseModel
from ..infrastructure.services.ocr import InvoiceOcrClient

install()
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[RichHandler(rich_tracebacks=True, markup=True)],
    )

logger = logging.getLogger(__name__)
ocr_client = InvoiceOcrClient()


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


def _load_purchase_order(invoice: InvoiceData) -> dict:
    """Busca la orden/cotización más parecida usando proveedor y detalle de productos.
    Si no existe, crea una nueva orden a partir de la factura."""
    if not invoice.supplier_name:
        raise ValueError("La factura no indica el nombre del proveedor necesario para buscar en Odoo.")
    invoice_details = [line.detalle for line in invoice.lines]

    created_from_invoice = False
    order = odoo_manager.find_purchase_order_by_similarity(
        invoice.supplier_name,
        invoice_details,
        invoice.total,
        supplier_rut=getattr(invoice, "supplier_rut", None),
    )
    if order:
        logger.info(f"Orden candidata en Odoo: {order.get('name')} (ID {order.get('id')})")
    if not order:
        logger.info("No se detectó una orden coincidente; creando una nueva en Odoo.")
        order = odoo_manager.create_purchase_order_from_invoice(invoice)
        created_from_invoice = True

    if not created_from_invoice and order.get("state") in {"draft", "sent"}:
        logger.info(f"La orden {order['name']} está en estado {order['state']}. Confirmando…")
        order = odoo_manager.confirm_purchase_order(order["id"])

    order["_created_from_invoice"] = created_from_invoice
    return order


def _finalize_order(order: dict, qty_by_product: Dict[int, float] | None = None) -> None:
    """Confirma la recepción y genera la factura sólo si la orden lo requiere."""
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
    invoice_ids = status.get("invoice_ids") or []
    if invoice_ids:
        logger.info("La orden %s ya tiene facturas (%s); se omite la creación automática.", order["name"], invoice_ids)
        return
    odoo_manager.create_invoice_for_order(order["id"])




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


def _match_line(invoice_line: InvoiceLine, order_lines: List[dict]) -> dict | None:
    """Encuentra la línea de Odoo cuyo DETALLE coincida con la línea de factura, con fallback por similitud."""
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

    target_raw = (invoice_line.detalle or "").strip()
    target = _normalize(target_raw)
    best_line = None
    best_score = 0.0
    for line in order_lines:
        candidate_raw = (line.get("detalle") or "").strip()
        candidate = _normalize(candidate_raw)
        if candidate_raw.lower() == target_raw.lower():
            return line
        if target and candidate and (target in candidate or candidate in target):
            best_line = line
            best_score = 1.0
            return line
        score = SequenceMatcher(None, target, candidate).ratio()
        if score > best_score:
            best_score = score
            best_line = line
    if best_score >= 0.5:
        return best_line
    return None


def _verify_line(invoice_line: InvoiceLine, order_lines: List[dict]) -> ProcessedProduct:
    """Compara cantidad, precio unitario y subtotal de una línea contra lo que devuelve Odoo."""
    order_line = _match_line(invoice_line, order_lines)
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


def process_invoice_file(image_path: str) -> InvoiceResponseModel:
    """
    Orquesta todo el flujo: extrae la factura, localiza/convierte la orden en Odoo,
    compara cabecera y líneas, verifica recepción y construye el resumen final.
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
            # Prefiere ajustar subtotal a cantidad x precio; recalcula precio unitario coherente.
            corrected_subtotal = expected_subtotal
            corrected_price = corrected_subtotal / line.cantidad
            normalized_lines.append(
                line.model_copy(
                    update={
                        "subtotal": corrected_subtotal,
                        "precio_unitario": corrected_price,
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

    resolved_supplier_id = odoo_manager._resolve_supplier_id(None, invoice.supplier_name)
    pending_products: List[ProcessedProduct] = []
    for line in invoice.lines:
        candidates = odoo_manager.get_product_candidates(
            line.detalle,
            supplier_id=resolved_supplier_id,
            supplier_name=invoice.supplier_name,
            invoice_line=line,
        )
        pending_products.append(
            ProcessedProduct(
                detalle=line.detalle,
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
            status="WAITING_FOR_HUMAN",
        )

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
    for line in invoice.lines:
        product_result = _verify_line(line, summary["lines"])
        matched_line = _match_line(line, summary["lines"])

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

            if desired_values:
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
                        or "No se pudieron ajustar los valores de la línea en Odoo."
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

    # Construye resumen breve según caso.
    if not needs_follow_up:
        try:
            _finalize_order(order, qty_by_product=qty_by_product)
            if order.get("_created_from_invoice"):
                final_summary = (
                    f"Se creó la orden de compra (ID {order['id']}). "
                    f"Se recepcionó y facturó correctamente."
                )
            else:
                final_summary = (
                    f"Se editó la orden de compra (ID {order['id']}). "
                    f"Se recepcionó y facturó correctamente."
                )
            return InvoiceResponseModel(
                summary=final_summary,
                products=products,
                needs_follow_up=False,
                neto_match=neto_match,
                iva_match=iva_match,
                total_match=total_match,
            )
        except Exception as exc:
            logger.error(f"No se pudo cerrar automáticamente la orden {order['name']}: {exc}")
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
