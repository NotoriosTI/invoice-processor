from typing import List
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from .models import InvoiceData, ProcessedProduct
from .processor import extract_invoice_data, process_invoice_file

class ParseInvoiceArgs(BaseModel):
    image_path: str = Field(..., description = "Ruta local del archivo PNG/JPG")

class UpsertInvoiceArgs(BaseModel):
    image_path: str = Field(..., description = "Ruta local del archivo a procesar")


@tool("parse_invoice_image", args_schema = ParseInvoiceArgs)
def parse_invoice_image(image_path: str) -> InvoiceData:
    """Extrae los datos estructurados de la factura"""
    return extract_invoice_data(image_path)

@tool("upsert_invoice_inventory", args_schema = UpsertInvoiceArgs)
def upsert_invoice_inventory(image_path: str) -> List[ProcessedProduct]:
    """Crea/actualiza productos en Odoo y ajusta inventario segùn la factura."""
    return process_invoice_file(image_path)

@tool("request_human_input")
def request_human_input(question: str) -> str:
    """Solicita informaciòn adicional al usuario cuando falte algun dato critico."""
    return f"HUMAN_INPUT_REQUIRED: {question}"