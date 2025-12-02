from typing import Dict, List
from dev_utils.pretty_logger import PrettyLogger

from ..infrastructure.services.odoo_connection_manager import odoo_manager
from .models import InvoiceData, InvoiceLine, ProcessedProduct, InvoiceResponseModel
from ..infrastructure.services.ocr import InvoiceOcrClient

logger = PrettyLogger(service_name="processor")
ocr_client = InvoiceOcrClient()


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
    """Busca la orden/cotización más parecida usando proveedor y detalle de productos."""
    if not invoice.supplier_name:
        raise ValueError("La factura no indica el nombre del proveedor necesario para buscar en Odoo.")
    invoice_details = [line.detalle for line in invoice.lines]
    order = odoo_manager.find_purchase_order_by_similarity(invoice.supplier_name, invoice_details)
    if not order:
        raise ValueError(
            "No se encontró una orden/cotización en Odoo que coincida con el proveedor y los productos."
        )
    if order.get("state") in {"draft", "sent"}:
        logger.info(f"La orden {order['name']} está en estado {order['state']}. Confirmando…")
        order = odoo_manager.confirm_purchase_order(order["id"])
    return order




def _build_order_summary(order: dict) -> Dict[str, any]:
    """Obtiene desde Odoo los totales y las líneas de la orden (detalle, cantidad, precios)."""
    lines = odoo_manager.read_order_lines(order.get("order_line", []))
    receipts = odoo_manager.read_receipts_for_order(order.get("picking_ids", []))
    return {
        "neto": order.get("amount_untaxed", 0.0),
        "iva_19": order.get("amount_tax", 0.0),
        "total": order.get("amount_total", 0.0),
        "lines": lines,
        "receipts": receipts,
    }


def _match_line(invoice_line: InvoiceLine, order_lines: List[dict]) -> dict | None:
    """Encuentra la línea de Odoo cuyo DETALLE coincida con la línea de factura."""
    for line in order_lines:
        if (line["detalle"] or "").strip().lower() == invoice_line.detalle.strip().lower():
            return line
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

    if not invoice.lines:
        raise ValueError("La factura no contiene líneas de productos para comparar.")
    order = _load_purchase_order(invoice)
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
    for line in invoice.lines:
        product_result = _verify_line(line, summary["lines"])
        matched_line = _match_line(line, summary["lines"])

        if matched_line and matched_line.get("id"):
            desired_values: Dict[str, float] = {}
            if not product_result.cantidad_match:
                desired_values["product_qty"] = line.cantidad
            if not product_result.precio_match:
                desired_values["price_unit"] = line.precio_unitario
            if not product_result.subtotal_match:
                desired_values["price_subtotal"] = line.subtotal

            if desired_values:
                odoo_manager.update_order_line(matched_line["id"], desired_values)

            receipt_info = receipt_by_product.get(matched_line.get("product_id"), {})
        else:
            receipt_info = {}

        if receipt_info:
            qty_received = float(receipt_info.get("quantity", 0.0))
            if abs(qty_received - line.cantidad) > 0.01:
                extra = f"Recepción: factura {line.cantidad} vs recibido {qty_received}."
                product_result.issues = f"{product_result.issues or ''} {extra}".strip()

        products.append(product_result)

    odoo_manager.recompute_order_amounts(order["id"])
    summary["neto"] = invoice.neto
    summary["iva_19"] = invoice.iva_19
    summary["total"] = invoice.total

    neto_match = True
    iva_match = True
    total_match = True

    summary_lines = [
        f"Neto: {'OK' if neto_match else 'ISSUE'} (Factura {invoice.neto}, Odoo {summary['neto']})",
        f"19% IVA: {'OK' if iva_match else 'ISSUE'} (Factura {invoice.iva_19}, Odoo {summary['iva_19']})",
        f"Total: {'OK' if total_match else 'ISSUE'} (Factura {invoice.total}, Odoo {summary['total']})",
    ]
    for product in products:
        status = "OK" if not product.issues else "ISSUE"
        summary_lines.append(f"- {product.detalle}: {status}")
        if product.issues:
            summary_lines.append(f"  → {product.issues}")

    needs_follow_up = any(p.issues for p in products) or not (neto_match and iva_match and total_match)

    if not needs_follow_up:
        odoo_manager.mark_order_as_invoiced(order["id"])

    return InvoiceResponseModel(
        summary="\n".join(summary_lines),
        products=products,
        needs_follow_up=needs_follow_up,
        neto_match=neto_match,
        iva_match=iva_match,
        total_match=total_match,
    )


