from typing import Dict, List
from dev_utils.pretty_logger import PrettyLogger

from .services.ocr import InvoiceOcrClient
from .services.odoo_client import OdooClient
from .models import InvoiceData, InvoiceLine, ProcessedProduct, InvoiceResponseModel


logger = PrettyLogger(service_name="processor")
ocr_client = InvoiceOcrClient()
odoo_client = OdooClient()


def extract_invoice_data(image_path: str) -> InvoiceData:
    """Lanza el OCR y valida que el resultado cumpla con nuestro esquema."""
    logger.info(f"Extrayendo datos de la factura {image_path}")
    raw = ocr_client.extract(image_path)
    if isinstance(raw, dict) and raw.get("error"):
        raise ValueError(f"OCR inválido: {raw['error']}")
    return (
        InvoiceData.model_validate_json(raw)
        if isinstance(raw, str)
        else InvoiceData.model_validate(raw)
    )



def _match_quote(invoice: InvoiceData) -> Dict:
    """Ubica la cotización/orden asociada usando proveedor + referencia del documento."""
    reference = invoice.document_reference or invoice.invoice_id
    quote = odoo_client.find_quote(invoice.supplier or "", invoice.invoice_id, reference)
    if not quote:
        raise ValueError("No pude encontrar una cotización relacionada con la factura.")
    logger.info("Se encontró la cotización %s", quote.get("name"))
    return quote


def _ensure_purchase_order(quote: Dict) -> Dict:
    """Confirma la cotización si sigue en borrador para trabajar con la orden de compra real."""
    if quote.get("state") == "purchase":
        logger.info("La cotización %s ya es orden de compra.", quote.get("name"))
        return quote
    logger.info("Confirmando cotización %s para generar orden de compra.", quote.get("name"))
    return odoo_client.confirm_quote(quote["id"])


def _collect_receipt_data(order: Dict) -> Dict[str, Dict]:
    """Consulta las recepciones vinculadas a la orden y las indexa por producto."""
    receipt = odoo_client.get_receipts_for_order(order["id"])
    result: Dict[str, Dict] = {}
    for line in receipt.get("lines", []):
        product_id = line["product_id"]
        result[product_id] = {
            "product_id": product_id,
            "product_name": line["product_name"],
            "sku": line.get("sku") or line.get("default_code"),
            "received_qty": line.get("received_qty", 0.0),
            "location": line.get("location"),
            "product_type": odoo_client.get_product_type(product_id),
        }
    return result


def _match_product(line: InvoiceLine, receipt_map: Dict[str, Dict]) -> Dict | None:
    """Hace matching por nombre o SKU para alinear la línea de factura con la recepción."""
    for data in receipt_map.values():
        if data["product_name"] == line.product_name:
            return data
        if line.sku and data["sku"] == line.sku:
            return data
    return None


def _verify_line(line: InvoiceLine, receipt_map: Dict[str, Dict]) -> ProcessedProduct:
    """Compara cantidad y ubicación del producto entre la factura y la recepción Odoo."""
    matched_receipt = _match_product(line, receipt_map)

    if not matched_receipt:
        return ProcessedProduct(
            action="receipt_issue",
            product_name=line.product_name,
            sku=line.sku or "N/A",
            invoice_quantity=line.quantity,
            issues="No se encontró recepción para este producto.",
        )

    odoo_qty = matched_receipt["received_qty"]
    product_type = matched_receipt["product_type"]
    expected_location = "Materia Prima" if product_type == "materia_prima" else "Productos Terminados"
    location_found = matched_receipt.get("location")

    issues = []
    if abs(odoo_qty - line.quantity) > 0.01:
        issues.append(f"Cantidad en Odoo ({odoo_qty}) no coincide con la factura ({line.quantity}).")
    if location_found and location_found != expected_location:
        issues.append(f"Ubicación actual '{location_found}' no coincide con '{expected_location}'.")

    action = "receipt_verified" if not issues else "receipt_issue"

    return ProcessedProduct(
        action=action,
        product_name=line.product_name,
        sku=matched_receipt["sku"] or line.sku or "N/A",
        invoice_quantity=line.quantity,
        odoo_quantity=odoo_qty,
        location_expected=expected_location,
        location_found=location_found,
        issues=" ".join(issues) if issues else None,
    )


def process_invoice_file(image_path: str) -> InvoiceResponseModel:
    """Flujo principal: OCR → match de cotización → validación de recepción y resumen final."""
    invoice = extract_invoice_data(image_path)
    quote = _match_quote(invoice)
    purchase_order = _ensure_purchase_order(quote)
    receipt_map = _collect_receipt_data(purchase_order)

    products = [_verify_line(line, receipt_map) for line in invoice.lines]

    summary_lines = [
        f"Cotización: {quote.get('name')} | Orden de compra: {purchase_order.get('name')}"
    ]
    for product in products:
        status = "OK" if not product.issues else "ISSUE"
        summary_lines.append(
            f"- {product.product_name} ({product.sku}): {status} "
            f"(Factura: {product.invoice_quantity}, Odoo: {product.odoo_quantity})"
        )
        if product.issues:
            summary_lines.append(f"  → {product.issues}")

    summary = "\n".join(summary_lines)

    return InvoiceResponseModel(
        summary=summary,
        quote_id=quote.get("name"),
        purchase_order_id=purchase_order.get("name"),
        products=products,
        needs_follow_up=any(prod.issues for prod in products),
    )
