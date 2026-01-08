import logging
import math
import re
import xmlrpc.client
from datetime import datetime
from typing import List

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from ..infrastructure.services.odoo_connection_manager import odoo_manager

logger = logging.getLogger(__name__)


class SplitPurchaseLineItem(BaseModel):
    product_search_term: str = Field(
        ..., description="SKU o nombre para buscar el producto en Odoo."
    )
    qty: float = Field(..., description="Cantidad para la nueva linea.")


class SplitPurchaseLineArgs(BaseModel):
    po_name: str = Field(..., description="Nombre de la Orden de Compra (ej: PO00123).")
    original_line_keyword: str = Field(
        ..., description="Texto para identificar la linea original."
    )
    new_items: List[SplitPurchaseLineItem] = Field(
        ..., description="Lista de items a crear, cada uno con product_search_term y qty."
    )


class UpdateLineQuantityArgs(BaseModel):
    po_name: str = Field(..., description="Nombre de la Orden de Compra.")
    product_keyword: str = Field(..., description="Texto para identificar la linea.")
    new_qty: float = Field(..., description="Nueva cantidad (> 0).")


class FinalizeInvoiceWorkflowArgs(BaseModel):
    po_name: str = Field(..., description="Nombre de la Orden de Compra.")
    block_if_line_keyword_present: str | None = Field(
        default=None,
        description="Si se indica, bloquea el cierre si existe una linea que contenga este texto.",
    )


def _get_unique_order_by_name(po_name: str) -> dict:
    po_name = (po_name or "").strip()
    if not po_name:
        raise ValueError("po_name es requerido.")
    order_ids = odoo_manager._execute_kw(
        "purchase.order",
        "search",
        [[["name", "=", po_name]]],
        {"limit": 2},
    )
    order_ids = odoo_manager._normalize_id_list(order_ids or [])
    if not order_ids:
        raise ValueError(f"No se encontro una OC con nombre '{po_name}'.")
    if len(order_ids) > 1:
        raise ValueError(f"Se encontraron multiples OCs con nombre '{po_name}': {order_ids}.")
    order = odoo_manager.read_order(order_ids[0])
    if not order:
        raise ValueError(f"No se pudo leer la OC '{po_name}'.")
    return order


def _describe_lines(lines: List[dict]) -> str:
    if not lines:
        return "sin lineas"
    parts = []
    for line in lines:
        line_id = line.get("id")
        detail = line.get("detalle") or line.get("line_name") or "sin_detalle"
        sku = line.get("sku")
        qty = line.get("cantidad")
        label = f"[{line_id}] {detail}"
        if sku:
            label += f" (SKU {sku})"
        if qty is not None:
            label += f" qty {qty}"
        parts.append(label)
    return "; ".join(parts)


def _find_matching_lines(lines: List[dict], keyword: str) -> List[dict]:
    keyword_norm = (keyword or "").strip().lower()
    if not keyword_norm:
        raise ValueError("El texto de busqueda no puede estar vacio.")
    matches = []
    for line in lines:
        line_name = (line.get("line_name") or "").lower()
        detalle = (line.get("detalle") or "").lower()
        if keyword_norm in line_name or keyword_norm in detalle:
            matches.append(line)
    return matches


def _looks_like_sku(value: str) -> bool:
    cleaned = re.sub(r"[^A-Za-z0-9_-]", "", value or "")
    return bool(cleaned) and (" " not in (value or ""))


def _resolve_product_by_term(term: str, supplier_id: int | None = None) -> dict:
    term_norm = (term or "").strip()
    if not term_norm:
        raise ValueError("product_search_term no puede estar vacio.")
    raw = term_norm.strip("`'\"").strip()
    sku_hint = raw
    if raw.lower().startswith("sku:"):
        sku_hint = raw.split(":", 1)[1].strip()
    product_id = None
    if sku_hint and _looks_like_sku(sku_hint):
        try:
            product_id = odoo_manager._resolve_product_by_default_code(sku_hint, supplier_id)
        except Exception:
            product_id = None
    if product_id:
        recs = odoo_manager._execute_kw(
            "product.product",
            "read",
            [[int(product_id)]],
            {"fields": ["id", "name", "default_code", "uom_po_id", "uom_id"]},
        )
        if recs:
            return recs[0]

    domain = [["name", "ilike", raw], ["purchase_ok", "=", True]]
    products = odoo_manager._execute_kw(
        "product.product",
        "search_read",
        [domain],
        {"fields": ["id", "name", "default_code", "uom_po_id", "uom_id"], "limit": 5},
    )
    products = products or []
    if not products:
        raise ValueError(f"No se encontro producto para '{term_norm}'.")
    if len(products) > 1:
        candidates = []
        for prod in products:
            prod_id = odoo_manager._normalize_id(prod.get("id"))
            name = prod.get("name") or "sin_nombre"
            sku = prod.get("default_code")
            sku_label = f" SKU {sku}" if sku else ""
            candidates.append(f"[{prod_id}] {name}{sku_label}")
        raise ValueError(
            f"Busqueda ambigua para '{term_norm}'. Candidatos: {', '.join(candidates)}."
        )
    return products[0]


@tool("split_purchase_line", args_schema=SplitPurchaseLineArgs)
def split_purchase_line(
    po_name: str, original_line_keyword: str, new_items: List[SplitPurchaseLineItem]
) -> str:
    """Reemplaza una linea generica por varias lineas especificas sin alterar el total."""
    try:
        order = _get_unique_order_by_name(po_name)
        order_state = order.get("state")
        if order_state not in {"draft", "sent"}:
            raise ValueError(
                f"Estado de OC no permitido para modificar: '{order_state}'."
            )
        lines = odoo_manager.read_order_lines(order.get("order_line", []))
        matches = _find_matching_lines(lines, original_line_keyword)
        if not matches:
            raise ValueError(
                f"No se encontro linea que contenga '{original_line_keyword}'. "
                f"Lineas disponibles: {_describe_lines(lines)}."
            )
        if len(matches) > 1:
            raise ValueError(
                f"Se encontraron multiples lineas con '{original_line_keyword}': "
                f"{_describe_lines(matches)}."
            )
        target_line = matches[0]
        line_id = odoo_manager._normalize_id(target_line.get("id"))
        if line_id is None:
            raise ValueError("No se pudo resolver line_id de la linea objetivo.")
        supplier_id = odoo_manager._normalize_id(order.get("partner_id"))

        original_qty = float(target_line.get("cantidad") or 0.0)
        original_price_unit = float(target_line.get("precio_unitario") or 0.0)
        original_tax_ids = odoo_manager._normalize_id_list(target_line.get("tax_ids") or [])
        original_uom = odoo_manager._normalize_id(target_line.get("product_uom"))

        line_extra = odoo_manager._execute_kw(
            "purchase.order.line",
            "read",
            [[line_id]],
            {"fields": ["date_planned", "name"]},
        )
        line_extra = line_extra[0] if line_extra else {}
        date_planned = line_extra.get("date_planned") or datetime.utcnow().strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        original_name = (
            line_extra.get("name")
            or target_line.get("line_name")
            or target_line.get("detalle")
        )

        if not new_items:
            raise ValueError("new_items no puede estar vacio.")
        resolved_items = []
        total_new_qty = 0.0
        for item in new_items:
            qty = float(item.qty or 0.0)
            if qty <= 0:
                raise ValueError("Cada qty en new_items debe ser > 0.")
            total_new_qty += qty
            product = _resolve_product_by_term(item.product_search_term, supplier_id=supplier_id)
            product_id = odoo_manager._normalize_id(product.get("id"))
            if product_id is None:
                raise ValueError(
                    f"No se pudo resolver product_id para '{product.get('name')}'."
                )
            product_uom = odoo_manager._normalize_id(
                product.get("uom_po_id") or product.get("uom_id")
            )
            if product_uom is None:
                product_uom = original_uom
            if product_uom is None:
                raise ValueError(
                    f"No se pudo resolver product_uom para '{product.get('name')}'."
                )
            line_name = product.get("name") or original_name
            if not line_name:
                raise ValueError("No se pudo resolver el nombre de la linea nueva.")
            resolved_items.append(
                {
                    "qty": qty,
                    "product_id": product_id,
                    "product_uom": product_uom,
                    "line_name": line_name,
                }
            )

        if not math.isclose(
            total_new_qty, original_qty, rel_tol=1e-6, abs_tol=1e-6
        ):
            raise ValueError(
                f"La suma de cantidades ({total_new_qty}) no coincide con la linea original ({original_qty})."
            )

        odoo_manager._execute_kw("purchase.order.line", "unlink", [[line_id]])

        created_lines = []
        order_id = odoo_manager._normalize_id(order.get("id"))
        for item in resolved_items:
            payload = {
                "order_id": order_id,
                "product_id": item["product_id"],
                "product_qty": float(item["qty"]),
                "price_unit": original_price_unit,
                "taxes_id": [(6, 0, original_tax_ids)],
                "product_uom": item["product_uom"],
                "date_planned": date_planned,
                "name": item["line_name"],
            }
            odoo_manager._execute_kw("purchase.order.line", "create", [[payload]])
            created_lines.append(f"{item['line_name']} x {item['qty']}")

        odoo_manager.recompute_order_amounts(order_id)
        created_summary = ", ".join(created_lines)
        return (
            f"OC {order.get('name') or po_name}: linea reemplazada por {len(created_lines)} "
            f"lineas. Items: {created_summary}."
        )
    except xmlrpc.client.Fault as exc:
        logger.exception("Odoo Fault en split_purchase_line: %s", exc)
        raise RuntimeError(
            f"Odoo devolvio un error en split_purchase_line: {exc}"
        ) from exc


@tool("update_line_quantity", args_schema=UpdateLineQuantityArgs)
def update_line_quantity(po_name: str, product_keyword: str, new_qty: float) -> str:
    """Actualiza la cantidad de una linea puntual en la OC."""
    try:
        if new_qty is None or float(new_qty) <= 0:
            raise ValueError("new_qty debe ser > 0.")
        order = _get_unique_order_by_name(po_name)
        order_state = order.get("state")
        if order_state not in {"draft", "sent"}:
            raise ValueError(
                f"Estado de OC no permitido para modificar: '{order_state}'."
            )
        lines = odoo_manager.read_order_lines(order.get("order_line", []))
        matches = _find_matching_lines(lines, product_keyword)
        if not matches:
            raise ValueError(
                f"No se encontro linea que contenga '{product_keyword}'. "
                f"Lineas disponibles: {_describe_lines(lines)}."
            )
        if len(matches) > 1:
            raise ValueError(
                f"Se encontraron multiples lineas con '{product_keyword}': "
                f"{_describe_lines(matches)}."
            )
        target_line = matches[0]
        line_id = odoo_manager._normalize_id(target_line.get("id"))
        if line_id is None:
            raise ValueError("No se pudo resolver line_id de la linea objetivo.")
        odoo_manager.update_order_line(line_id, {"product_qty": float(new_qty)})
        order_id = odoo_manager._normalize_id(order.get("id"))
        odoo_manager.recompute_order_amounts(order_id)
        price_unit = float(target_line.get("precio_unitario") or 0.0)
        subtotal = float(new_qty) * price_unit
        detail = (
            target_line.get("detalle")
            or target_line.get("line_name")
            or target_line.get("product_name")
            or "linea"
        )
        return (
            f"OC {order.get('name') or po_name}: linea '{detail}' actualizada a qty "
            f"{new_qty}. Subtotal estimado: {subtotal:.2f}."
        )
    except xmlrpc.client.Fault as exc:
        logger.exception("Odoo Fault en update_line_quantity: %s", exc)
        raise RuntimeError(
            f"Odoo devolvio un error en update_line_quantity: {exc}"
        ) from exc


@tool("finalize_invoice_workflow", args_schema=FinalizeInvoiceWorkflowArgs)
def finalize_invoice_workflow(po_name: str, block_if_line_keyword_present: str | None = None) -> str:
    """Confirma OC y recepciona (sin tocar facturas)."""
    try:
        order = _get_unique_order_by_name(po_name)
        order_id = odoo_manager._normalize_id(order.get("id"))
        if order_id is None:
            raise ValueError("No se pudo resolver el id de la OC.")
        if block_if_line_keyword_present:
            lines = odoo_manager.read_order_lines(order.get("order_line", []))
            pending = _find_matching_lines(lines, block_if_line_keyword_present)
            if pending:
                raise RuntimeError(
                    f"Bloqueado: la OC aun contiene lineas con '{block_if_line_keyword_present}'. "
                    "Aplica el desglose antes de recepcionar."
                )
        order_state = order.get("state")
        actions = []
        if order_state in {"draft", "sent"}:
            order = odoo_manager.confirm_purchase_order(order_id)
            actions.append("confirmada")
        elif order_state in {"purchase", "done"}:
            actions.append("ya_confirmada")
        elif order_state == "cancel":
            raise ValueError("La OC esta cancelada. No se puede continuar.")
        else:
            raise ValueError(f"Estado de OC no soportado: '{order_state}'.")

        odoo_manager.confirm_order_receipt(order)
        picking_ids = odoo_manager._normalize_id_list(order.get("picking_ids", []))
        picking_summaries = []
        if picking_ids:
            pickings = odoo_manager._execute_kw(
                "stock.picking",
                "read",
                [picking_ids],
                {"fields": ["id", "name", "state"]},
            )
            for picking in pickings:
                name = picking.get("name") or str(picking.get("id"))
                state = picking.get("state")
                picking_summaries.append(f"{name}:{state}")
                if state != "done":
                    raise RuntimeError(
                        f"Picking {name} quedo en estado '{state}'. Se requiere intervencion manual."
                    )

        pickings_info = (
            f" Pickings: {', '.join(picking_summaries)}."
            if picking_summaries
            else " Sin pickings."
        )
        action_summary = ", ".join(actions) if actions else "ejecutado"
        return (
            f"OC {order.get('name') or po_name}: flujo {action_summary}. "
            f"Recepcion confirmada.{pickings_info}"
        )
    except xmlrpc.client.Fault as exc:
        logger.exception("Odoo Fault en finalize_invoice_workflow: %s", exc)
        raise RuntimeError(
            f"Odoo devolvio un error en finalize_invoice_workflow: {exc}"
        ) from exc
