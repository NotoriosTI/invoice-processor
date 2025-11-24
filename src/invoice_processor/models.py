from typing import List, Optional, Literal
from pydantic import BaseModel, Field


class InvoiceLine(BaseModel):
    product_name: str = Field(..., description="Nombre detectado en la factura")
    quantity: float = Field(..., gt=0, description="Cantidad del producto")
    sku: Optional[str] = None
    category: Optional[str] = None
    quote_reference: Optional[str] = Field(
        default=None,
        description="Referencia de cotización/OC detectada en la factura",
    )


class InvoiceData(BaseModel):
    supplier: Optional[str] = None
    invoice_id: Optional[str] = None
    delivery_date: Optional[str] = None
    lines: List[InvoiceLine]


class ProcessedProduct(BaseModel):
    action: Literal["quote_matched", "po_created", "receipt_verified", "receipt_issue"]
    product_name: str
    sku: str
    invoice_quantity: float
    odoo_quantity: Optional[float] = None
    location_expected: Optional[str] = None
    location_found: Optional[str] = None
    issues: Optional[str] = None


class InvoiceResponseModel(BaseModel):
    summary: str
    quote_id: Optional[str] = None
    purchase_order_id: Optional[str] = None
    products: List[ProcessedProduct]
    needs_follow_up: bool = False
