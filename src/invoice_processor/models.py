from typing import List, Optional
from pydantic import BaseModel, Field


class InvoiceLine(BaseModel):
    product_name: str = Field(..., description="Nombre detectado en la factura")
    quantity: float = Field(..., gt=0, description="Cantidad del producto")
    sku: Optional[str] = None
    category: Optional[str] = None


class InvoiceData(BaseModel):
    supplier: Optional[str] = None
    invoice_id: Optional[str] = None
    delivery_date: Optional[str] = None
    lines: List[InvoiceLine]


class ProcessedProduct(BaseModel):
    action: str = Field(..., description="created | updated")
    product_name: str
    sku: str
    quantity: float
    category: Optional[str] = None


class InvoiceResponseModel(BaseModel):
    summary: str
    products: List[ProcessedProduct]
    needs_follow_up: bool = False
