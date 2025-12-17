from typing import List, Optional
from pydantic import BaseModel, Field


class InvoiceLine(BaseModel):
    detalle: str = Field(..., description="Producto tal como aparece en la factura (DETALLE)")
    cantidad: float = Field(..., gt=0, description="Cantidad facturada (CANT)")
    precio_unitario: float = Field(..., description="Precio unitario (P. UNITARIO)")
    subtotal: float = Field(..., description="Total por línea (columna TOTAL junto a P. UNITARIO)")
    unidad: Optional[str] = Field(
        default=None,
        description="Unidad de medida leída en la factura (kg, g, ml, unidad, etc.)",
    )
    descuento_pct: Optional[float] = Field(
        default=None, description="Descuento porcentual aplicado a la línea (% DCTO), si existe"
    )
    descuento_monto: Optional[float] = Field(
        default=None, description="Descuento en CLP aplicado a la línea, si existe"
    )


class InvoiceData(BaseModel):
    supplier_name: Optional[str] = Field(
        default=None, description="Nombre del proveedor según la factura"
    )
    supplier_rut: Optional[str] = Field(
        default=None, description="RUT/tax_id del proveedor según la factura"
    )
    ocr_dudoso: bool = Field(
        default=False, description="True si el OCR detecta incoherencias internas en montos"
    )
    ocr_warning: Optional[str] = Field(
        default=None, description="Mensaje de advertencia generado por el OCR"
    )
    descuento_global: float = Field(
        default=0.0, description="Descuento global aplicado al documento (CLP)"
    )
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
