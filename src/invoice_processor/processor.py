from typing import List, Dict
from dev_utils.pretty_logger import PrettyLogger

from .services.ocr import InvoiceOcrClient
from .services.odoo_client import OdooClient
from .models import InvoiceData, InvoiceLine, ProcessedProduct, InvoiceResponseModel


logger = PrettyLogger(service_name="processor")
ocr_client = InvoiceOcrClient()
odoo_client = OdooClient()


def extract_invoice_data(image_path: str) -> InvoiceData:
    """
    Ejecuta OCR sobre la imagen y valida el resultado contra InvoiceData.
    """
    logger.info(f"Extrayendo datos de la factura {image_path}")
    raw = ocr_client.extract(image_path)
    return (
        InvoiceData.model_validate_json(raw)
        if isinstance(raw, str)
        else InvoiceData.model_validate(raw)
    )


def _match_quote(invoice: InvoiceData) -> Dict:
    reference = invoice.lines[0].quote_reference if invoice.lines else None
    quote = odoo_client.find_quote(invoice.supplier or "", invoice.invoice_id, reference)
    if not quote:
        raise ValueError("No pude encontrar una cotización relacionada con la factura.")
    logger.info("Se encontró la cotización %s", quote.get("name"))
    return quote


def _ensure_purchase_order(quote: Dict) -> Dict:
    if quote.get("state") == "purchase":
        logger.info("La cotización %s ya es orden de compra.", quote.get("name"))
        return quote
    logger.info("Confirmando cotización %s para generar orden de compra.", quote.get("name"))
    return odoo_client.confirm_quote(quote["id"])


def _collect_receipt_data(order: Dict) -> Dict[str, Dict]:
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


def _verify_line(line: InvoiceLine, receipt_map: Dict[str, Dict]) -> ProcessedProduct:
    matched_receipt = next(
        (
            data
            for data in receipt_map.values()
            if data["product_name"] == line.product_name
            or data["sku"] == line.sku
        ),
        None,
    )

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
    """
    Orquesta la lectura de la factura, identificación de cotización y validación de recepciones.
    """
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
