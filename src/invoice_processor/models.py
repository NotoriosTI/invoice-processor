from typing import List, Optional
from pydantic import BaseModel, Field


class InvoiceLine(BaseModel):
    detalle: str = Field(..., description="Producto tal como aparece en la factura (DETALLE)")
    cantidad: float = Field(..., gt=0, description="Cantidad facturada (CANT)")
    precio_unitario: float = Field(..., description="Precio unitario (P. UNITARIO)")
    subtotal: float = Field(..., description="Total por línea (columna TOTAL junto a P. UNITARIO)")


class InvoiceData(BaseModel):
    neto: float = Field(..., description="Monto NETO del documento")
    iva_19: float = Field(..., description="Impuesto 19% I.V.A")
    total: float = Field(..., description="TOTAL general de la factura")
    lines: List[InvoiceLine]


class ProcessedProduct(BaseModel):
    detalle: str
    cantidad_match: bool
    precio_match: bool
    subtotal_match: bool
    issues: Optional[str] = None


class InvoiceResponseModel(BaseModel):
    summary: str
    products: List[ProcessedProduct]
    needs_follow_up: bool = False
    neto_match: Optional[bool] = None
    iva_match: Optional[bool] = None
    total_match: Optional[bool] = None
