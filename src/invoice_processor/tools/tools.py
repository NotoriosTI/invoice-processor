from langchain_core.tools import tool
from pydantic import BaseModel, Field

from ..core.models import InvoiceData, InvoiceResponseModel
from ..core.processor import extract_invoice_data, process_invoice_file
from ..infrastructure.services.odoo_connection_manager import odoo_manager


class ParseInvoiceArgs(BaseModel):
    image_path: str = Field(..., description="Ruta local del archivo PNG/JPG que contiene la factura")


class ProcessInvoiceArgs(BaseModel):
    image_path: str = Field(..., description="Ruta local del archivo a procesar")


class MapProductDecisionArgs(BaseModel):
    invoice_detail: str = Field(..., description="Nombre del producto tal como aparece en la factura")
    odoo_product_id: int | None = Field(default=None, description="ID del producto seleccionado en Odoo (si se conoce)")
    supplier_id: int = Field(..., description="ID del proveedor en Odoo")
    default_code: str | None = Field(default=None, description="Código interno (SKU) del producto en Odoo")


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


@tool("map_product_decision_tool", args_schema=MapProductDecisionArgs)
def map_product_decision_tool(invoice_detail: str, odoo_product_id: int | None, supplier_id: int, default_code: str | None = None) -> str:
    """Guarda la decisión humana del producto, enlazando el nombre de factura con el ID de Odoo."""
    odoo_manager.map_product_decision(invoice_detail, odoo_product_id, supplier_id, default_code=default_code)
    return "Decisión registrada. Reintenta el procesamiento de la factura."
