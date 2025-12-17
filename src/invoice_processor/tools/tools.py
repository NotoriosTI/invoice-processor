from langchain_core.tools import tool
from pydantic import BaseModel, Field

from ..core.models import InvoiceData, InvoiceResponseModel
from ..core.processor import extract_invoice_data, process_invoice_file


class ParseInvoiceArgs(BaseModel):
    image_path: str = Field(..., description="Ruta local del archivo PNG/JPG que contiene la factura")


class ProcessInvoiceArgs(BaseModel):
    image_path: str = Field(..., description="Ruta local del archivo a procesar")


@tool("parse_invoice_image", args_schema=ParseInvoiceArgs)
def parse_invoice_image(image_path: str) -> InvoiceData:
    """Extrae proveedor, RUT, descuento_global, descuentos por línea, CANT, DETALLE, P. UNITARIO, subtotal por línea, NETO, IVA 19% y TOTAL."""
    return extract_invoice_data(image_path)


@tool("process_invoice_purchase_flow", args_schema=ProcessInvoiceArgs)
def process_invoice_purchase_flow(image_path: str) -> InvoiceResponseModel:
    """
    Compara los campos de la factura con los de Odoo. Devuelve coincidencias/diferencias
    en cabecera y por producto. Si el OCR falla, informa al usuario.
    """
    try:
        return process_invoice_file(image_path)
    except ValueError as exc:
        return InvoiceResponseModel(
            summary=f"No se pudo interpretar la factura: {exc}",
            products=[],
            needs_follow_up=True,
        )
