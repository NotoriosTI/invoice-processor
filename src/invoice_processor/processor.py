from typing import List
from dev_utils.pretty_logger import PrettyLogger

from .services.ocr import InvoiceOcrClient
from .services.odoo_client import OdooClient
from .models import InvoiceData, InvoiceLine, ProcessedProduct

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


def _create_product(line: InvoiceLine) -> dict:
    payload = {
        "name": line.product_name,
        "default_code": line.sku,
        "categ_id": line.category,
    }
    return odoo_client.create_product(payload)


def _upsert(line: InvoiceLine) -> ProcessedProduct:
    product = odoo_client.find_product(line.sku, line.product_name)
    action = "Updated"

    if not product:
        product = _create_product(line)
        action = "Created"

    result = odoo_client.update_inventory(
        product_id=product["id"], quantity_delta=line.quantity
    )
    return ProcessedProduct(
        action=action.lower(),
        product_name=product["name"],
        sku=product.get("default_code") or line.sku or "N/A",
        quantity=result["new_quantity"],
        category=product.get("categ_id"),
    )


def process_invoice_file(image_path: str) -> List[ProcessedProduct]:
    """
    Realiza la lectura de la factura y la actualización del inventario en Odoo.
    """
    invoice = extract_invoice_data(image_path)
    processed: List[ProcessedProduct] = []

    for line in invoice.lines:
        try:
            processed.append(_upsert(line))
        except Exception as exc:
            logger.exception(f"No se pudo procesar {line.product_name}: {exc}")

    return processed
