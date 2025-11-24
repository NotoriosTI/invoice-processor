from typing import List
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from .models import InvoiceData, InvoiceResponseModel
from .processor import extract_invoice_data, process_invoice_file

class ParseInvoiceArgs(BaseModel):
    image_path: str = Field(..., description = "Ruta local del archivo PNG/JPG")

class ProcessInvoiceArgs(BaseModel):
    image_path: str = Field(..., description = "Ruta local del archivo a procesar")


@tool("parse_invoice_image", args_schema = ParseInvoiceArgs)
def parse_invoice_image(image_path: str) -> InvoiceData:
    """Extrae los datos estructurados de la factura"""
    return extract_invoice_data(image_path)

@tool("process_invoice_purchase_flow", args_schema = ProcessInvoiceArgs)
def process_invoice_purchase_flow(image_path: str) -> InvoiceResponseModel:
    """Procesa la factura completa: identifica cotización, genera orden de compra y valida recepciones"""
    return process_invoice_file(image_path)

@tool("request_human_input")
def request_human_input(question: str) -> str:
    """Solicita informaciòn adicional al usuario cuando falte algun dato critico."""
    return f"HUMAN_INPUT_REQUIRED: {question}"