from typing import List, Optional, Literal
from pydantic import BaseModel, Field, ConfigDict, field_validator


class InvoiceLine(BaseModel):
    model_config = ConfigDict(extra="forbid")
    detalle: str = Field(..., description="Producto tal como aparece en la factura (DETALLE)")
    default_code: Optional[str] = Field(
        default=None, description="SKU/default_code del producto en Odoo (si está disponible)"
    )
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
    model_config = ConfigDict(extra="forbid")
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


class ProductCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: int = Field(..., description="ID del producto en Odoo")
    name: Optional[str] = Field(default=None, description="Nombre del producto en Odoo")
    score: Optional[float] = Field(default=None, description="Score de similitud")
    default_code: Optional[str] = Field(default=None, description="Código interno (SKU) del producto en Odoo")

    @field_validator("default_code", mode="before")
    @classmethod
    def _normalize_default_code(cls, value):
        if value in (False, None, ""):
            return None
        return value


class ProcessedProduct(BaseModel):
    model_config = ConfigDict(extra="forbid")
    detalle: str
    invoice_detail: Optional[str] = Field(
        default=None, description="Nombre del producto tal como aparece en la factura"
    )
    candidate_name: Optional[str] = Field(
        default=None, description="Nombre del producto propuesto en Odoo"
    )
    candidate_default_code: Optional[str] = Field(
        default=None, description="SKU/default_code del producto propuesto en Odoo"
    )
    supplier_name: Optional[str] = Field(
        default=None, description="Nombre del proveedor detectado en la factura"
    )
    cantidad_match: bool
    precio_match: bool
    subtotal_match: bool
    issues: Optional[str] = None
    status: Optional[Literal["MATCHED", "NEW_CREATED", "AMBIGUOUS"]] = Field(
        default=None, description="Estado de resolución de producto"
    )
    candidates: Optional[List[ProductCandidate]] = Field(
        default=None, description="Candidatos sugeridos cuando el match es ambiguo"
    )


class InvoiceResponseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    summary: str
    products: List[ProcessedProduct]
    needs_follow_up: bool = False
    neto_match: Optional[bool] = None
    iva_match: Optional[bool] = None
    total_match: Optional[bool] = None
    po_name: Optional[str] = Field(
        default=None, description="Nombre de la OC asociada al flujo (ej: PO000123)"
    )
    po_id: Optional[int] = Field(
        default=None, description="ID interno de la OC en Odoo"
    )
    supplier_id: Optional[int] = Field(
        default=None, description="ID de proveedor usado para mapear productos"
    )
    supplier_name: Optional[str] = Field(
        default=None, description="Nombre del proveedor detectado en la factura"
    )
    supplier_rut: Optional[str] = Field(
        default=None, description="RUT/tax_id del proveedor detectado en la factura"
    )
    status: Optional[Literal["WAITING_FOR_HUMAN", "WAITING_FOR_APPROVAL"]] = Field(
        default=None, description="Estado global del flujo de la factura"
    )
