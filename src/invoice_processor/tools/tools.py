from langchain_core.tools import tool
from pydantic import BaseModel, Field

from ..core.models import InvoiceData, InvoiceResponseModel
from ..core.processor import extract_invoice_data, process_invoice_file
from ..infrastructure.services.odoo_connection_manager import odoo_manager


class ParseInvoiceArgs(BaseModel):
    image_path: str = Field(..., description="Ruta local del archivo PNG/JPG que contiene la factura")


class SplitPlanLineItem(BaseModel):
    product_search_term: str = Field(
        ..., description="SKU o nombre del producto para desglosar."
    )
    qty: float = Field(..., description="Cantidad para el item desglosado.")


class SplitPlanItem(BaseModel):
    original_line_keyword: str = Field(
        ..., description="Texto para identificar la línea a desglosar."
    )
    new_items: list[SplitPlanLineItem] = Field(
        ..., description="Items del desglose con product_search_term y qty."
    )


class ProcessInvoiceArgs(BaseModel):
    image_path: str = Field(..., description="Ruta local del archivo a procesar")
    allow_odoo_write: bool = Field(
        default=False,
        description="Si es True, permite escribir en Odoo (crear/editar OC, recepcionar, facturar).",
    )
    split_plan: list[SplitPlanItem] | None = Field(
        default=None,
        description="Plan de desglose antes de crear la OC. Lista de {original_line_keyword, new_items}.",
    )


class MapProductDecisionArgs(BaseModel):
    invoice_detail: str = Field(..., description="Nombre del producto tal como aparece en la factura")
    odoo_product_id: int | None = Field(default=None, description="ID del producto seleccionado en Odoo (si se conoce)")
    supplier_id: int | None = Field(default=None, description="ID del proveedor en Odoo")
    supplier_name: str | None = Field(
        default=None,
        description="Nombre del proveedor tal como aparece en la factura/Odoo (si falta supplier_id)",
    )
    supplier_rut: str | None = Field(
        default=None, description="RUT/tax_id del proveedor (si falta supplier_id)"
    )
    default_code: str | None = Field(default=None, description="Código interno (SKU) del producto en Odoo")


@tool("parse_invoice_image", args_schema=ParseInvoiceArgs)
def parse_invoice_image(image_path: str) -> InvoiceData:
    """Extrae proveedor, RUT, descuento_global, descuentos por línea, CANT, DETALLE, P. UNITARIO, subtotal por línea, NETO, IVA 19% y TOTAL."""
    return extract_invoice_data(image_path)


@tool("process_invoice_purchase_flow", args_schema=ProcessInvoiceArgs)
def process_invoice_purchase_flow(
    image_path: str, allow_odoo_write: bool = False, split_plan: list[SplitPlanItem] | None = None
) -> InvoiceResponseModel:
    """
    Compara los campos de la factura con los de Odoo. Devuelve coincidencias/diferencias
    en cabecera y por producto. Si el OCR falla, informa al usuario.
    Si allow_odoo_write es False, no escribe en Odoo y solicita aprobacion.
    split_plan permite desglosar lineas antes de crear la OC.
    """
    try:
        return process_invoice_file(image_path, allow_odoo_write=allow_odoo_write, split_plan=split_plan)
    except ValueError as exc:
        return InvoiceResponseModel(
            summary=f"No se pudo interpretar la factura: {exc}",
            products=[],
            needs_follow_up=True,
        )


@tool("map_product_decision_tool", args_schema=MapProductDecisionArgs)
def map_product_decision_tool(
    invoice_detail: str,
    odoo_product_id: int | None,
    supplier_id: int | None,
    default_code: str | None = None,
    supplier_name: str | None = None,
    supplier_rut: str | None = None,
) -> str:
    """Registra el mapeo en `product.supplierinfo` (memoria de mapeo) usando odoo_product_id o default_code (SKU). No modifica `purchase.order`."""
    if supplier_id is None:
        supplier_id = odoo_manager._resolve_supplier_id(None, supplier_name, supplier_rut)
    if supplier_id is None:
        raise ValueError("No se pudo resolver supplier_id. Provee supplier_id o supplier_name/supplier_rut válidos.")
    odoo_manager.map_product_decision(invoice_detail, odoo_product_id, supplier_id, default_code=default_code)
    resolved_product_id = odoo_manager._normalize_id(odoo_product_id) if odoo_product_id is not None else None
    if resolved_product_id is None and default_code:
        try:
            resolved_product_id = odoo_manager._resolve_product_by_default_code(
                default_code, odoo_manager._normalize_id(supplier_id)
            )
        except Exception:
            resolved_product_id = None
    product_name = None
    product_sku = None
    if resolved_product_id is not None:
        try:
            recs = odoo_manager._execute_kw(
                "product.product",
                "search_read",
                [[["id", "=", resolved_product_id]]],
                {"fields": ["name", "default_code"], "limit": 1},
            )
            if recs:
                product_name = recs[0].get("name")
                sku = recs[0].get("default_code")
                product_sku = sku if sku not in (False, None, "") else None
        except Exception:
            product_name = None
            product_sku = None
    target = product_name or (product_sku or resolved_product_id or default_code or "producto")
    sku_info = f" (SKU: {product_sku})" if product_sku else ""
    return f"Mapeo registrado: '{invoice_detail}' ahora apunta a '{target}'{sku_info}. Revisa la lista nuevamente."
