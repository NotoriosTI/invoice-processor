
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from .models import InvoiceData, InvoiceResponseModel
from .processor import extract_invoice_data, process_invoice_file

class ParseInvoiceArgs(BaseModel):
    image_path: str = Field(..., description = "Ruta local del archivo PNG/JPG")

class ProcessInvoiceArgs(BaseModel):
    image_path: str = Field(..., description = "ruta local del archivo a procesar")

@tool("parse_invoice_image", args_schema  = ParseInvoiceArgs)
def parse_invoice_image(image_path: str) -> InvoiceData:
    """Extrae proveedor, referencia y detalle de productos desde la factura"""
    return extract_invoice_data(image_path)

@tool("process_invoice_purchase_flow", args_schema=ProcessInvoiceArgs)
def process_invoice_purchase_flow(image_path: str) -> InvoiceResponseModel:
    """
    Procesa la factura completa; si el OCR falla, retorna una respuesta con summary explicativo.
    """
    try:
        return process_invoice_file(image_path)
    except ValueError as exc:
        return InvoiceResponseModel(
            summary=f"No se pudo interpretar la factura: {exc}",
            products=[],
            needs_follow_up=True,
        )


@tool("request_human_input")
def request_human_input(question: str) -> str:
    """Solicita informacion adicional al usuario cuando falten datos clave"""
    return f"HUMAN_INPUT_REQUIRED: {question}"