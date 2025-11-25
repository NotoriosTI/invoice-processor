from typing import List, Optional, Literal
from pydantic import BaseModel, Field


class InvoiceLine(BaseModel):
    product_name: str = Field(..., description="Nombre detectado en la factura")
    description: Optional[str] = Field(
        default=None, description="Descripción del producto en la factura"
        )
    quantity: float = Field(..., gt=0, description="Cantidad facturada")
    unit_price: Optional[float] = Field(default=None, description="Precio unitario")
    sku: Optional[str] = None
    category: Optional[str] = None
    quote_reference: Optional[str] = Field(
        default=None,
        description="Referencia de cotización/OC detectada en la factura",
    )


class InvoiceData(BaseModel):
    supplier: Optional[str] = None
    invoice_id: Optional[str] = None
    document_reference: Optional[str] = Field(
        default=None, description="Folio o referencia principal de la factura"
    )
    delivery_date: Optional[str] = None
    invoice_date: Optional[str] = Field(default=None, description="Fecha del documento")
    delivery_address: Optional[str] = None
    tax_percent: Optional[float] = Field(default=None, description="Porcentaje de IVA")
    total_amount: Optional[float] = Field(default=None, description="Total facturado")
    lines: List[InvoiceLine]


class ProcessedProduct(BaseModel):
    action: Literal["quote_matched", "po_created", "receipt_verified", "receipt_issue"]
    product_name: str
    sku: str
    invoice_quantity: float
    odoo_quantity: Optional[float] = None
    price_match: Optional[bool] = None
    tax_match: Optional[bool] = None
    location_expected: Optional[str] = None
    location_found: Optional[str] = None
    issues: Optional[str] = None


class InvoiceResponseModel(BaseModel):
    summary: str
    quote_id: Optional[str] = None
    purchase_order_id: Optional[str] = None
    products: List[ProcessedProduct]
    needs_follow_up: bool = False
    date_match: Optional[bool] = None
    reference_match: Optional[bool] = None
    total_match: Optional[bool] = None
    supplier_match: Optional[bool] = None
    delivery_address_match: Optional[bool] = None
