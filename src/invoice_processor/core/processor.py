from typing import Dict, List
from dev_utils.pretty_logger import PrettyLogger
from rich.traceback import install

from ..infrastructure.services.odoo_connection_manager import odoo_manager
from .models import InvoiceData, InvoiceLine, ProcessedProduct, InvoiceResponseModel
from ..infrastructure.services.ocr import InvoiceOcrClient

install()

logger = PrettyLogger(service_name="processor")
ocr_client = InvoiceOcrClient()


def extract_invoice_data(image_path: str) -> InvoiceData:
    """Ejecuta OCR sobre la imagen y valida que tenga los campos esperados."""
    logger.info(f"Extrayendo datos de la factura {image_path}")
    raw = ocr_client.extract(image_path)
    if isinstance(raw, dict) and raw.get("error"):
        raise ValueError(f"OCR inválido: {raw['error']}")
    return (
        InvoiceData.model_validate_json(raw)
        if isinstance(raw, str)
        else InvoiceData.model_validate(raw)
    )


def _load_purchase_order(invoice: InvoiceData) -> dict:
    """Busca la orden/cotización más parecida usando proveedor y detalle de productos.
    Si no existe, crea una nueva orden a partir de la factura."""
    if not invoice.supplier_name:
        raise ValueError("La factura no indica el nombre del proveedor necesario para buscar en Odoo.")
    invoice_details = [line.detalle for line in invoice.lines]

    created_from_invoice = False
    order = odoo_manager.find_purchase_order_by_similarity(
        invoice.supplier_name,
        invoice_details,
        invoice.total,
    )
    if order:
        logger.info(f"Orden candidata en Odoo: {order.get('name')} (ID {order.get('id')})")
    if not order:
        logger.info("No se detectó una orden coincidente; creando una nueva en Odoo.")
        order = odoo_manager.create_purchase_order_from_invoice(invoice)
        created_from_invoice = True

    if not created_from_invoice and order.get("state") in {"draft", "sent"}:
        logger.info(f"La orden {order['name']} está en estado {order['state']}. Confirmando…")
        order = odoo_manager.confirm_purchase_order(order["id"])

    order["_created_from_invoice"] = created_from_invoice
    return order


def _finalize_order(order: dict) -> None:
    """Confirma la recepción y genera la factura sólo si la orden lo requiere."""
    status = odoo_manager.get_order_status(order["id"])
    state = status.get("state")
    if state not in {"purchase", "done"}:
        logger.info("Orden %s en estado %s; no se cerrará automáticamente.", order["name"], state)
        return
    logger.info(f"Finalizando orden {order['name']} (estado {state}).")
    picking_ids = status.get("picking_ids") or []
    if picking_ids:
        odoo_manager.confirm_order_receipt({"picking_ids": picking_ids})
    else:
        logger.info("La orden %s no tiene recepciones pendientes en Odoo.", order["name"])
    invoice_ids = status.get("invoice_ids") or []
    if invoice_ids:
        logger.info("La orden %s ya tiene facturas (%s); se omite la creación automática.", order["name"], invoice_ids)
        return
    odoo_manager.create_invoice_for_order(order["id"])




def _build_order_summary(order: dict) -> Dict[str, any]:
    """Obtiene desde Odoo los totales y las líneas de la orden (detalle, cantidad, precios)."""
    order_data = order
    if order.get("id"):
        # Relee la orden para asegurarse de tener montos y referencias actualizadas.
        fresh = odoo_manager.read_order(order["id"])
        if fresh:
            order_data = fresh

    lines = odoo_manager.read_order_lines(order_data.get("order_line", []))
    receipts = odoo_manager.read_receipts_for_order(order_data.get("picking_ids", []))
    return {
        "neto": order_data.get("amount_untaxed", 0.0),
        "iva_19": order_data.get("amount_tax", 0.0),
        "total": order_data.get("amount_total", 0.0),
        "lines": lines,
        "receipts": receipts,
    }


def _match_line(invoice_line: InvoiceLine, order_lines: List[dict]) -> dict | None:
    """Encuentra la línea de Odoo cuyo DETALLE coincida con la línea de factura."""
    for line in order_lines:
        if (line["detalle"] or "").strip().lower() == invoice_line.detalle.strip().lower():
            return line
    return None


def _verify_line(invoice_line: InvoiceLine, order_lines: List[dict]) -> ProcessedProduct:
    """Compara cantidad, precio unitario y subtotal de una línea contra lo que devuelve Odoo."""
    order_line = _match_line(invoice_line, order_lines)
    if not order_line:
        return ProcessedProduct(
            detalle=invoice_line.detalle,
            cantidad_match=False,
            precio_match=False,
            subtotal_match=False,
            issues="Producto no encontrado en la orden de compra.",
        )

    cantidad_match = abs(order_line["cantidad"] - invoice_line.cantidad) <= 0.01
    precio_match = abs(order_line["precio_unitario"] - invoice_line.precio_unitario) <= 0.01
    subtotal_match = abs(order_line["subtotal"] - invoice_line.subtotal) <= 0.5

    issues = []
    if not cantidad_match:
        issues.append(
            f"CANT: factura {invoice_line.cantidad} vs Odoo {order_line['cantidad']}."
        )
    if not precio_match:
        issues.append(
            f"P. UNITARIO: factura {invoice_line.precio_unitario} vs Odoo {order_line['precio_unitario']}."
        )
    if not subtotal_match:
        issues.append(
            f"Subtotal: factura {invoice_line.subtotal} vs Odoo {order_line['subtotal']}."
        )

    return ProcessedProduct(
        detalle=invoice_line.detalle,
        cantidad_match=cantidad_match,
        precio_match=precio_match,
        subtotal_match=subtotal_match,
        issues=" ".join(issues) if issues else None,
    )


def process_invoice_file(image_path: str) -> InvoiceResponseModel:
    """
    Orquesta todo el flujo: extrae la factura, localiza/convierte la orden en Odoo,
    compara cabecera y líneas, verifica recepción y construye el resumen final.
    """
    invoice = extract_invoice_data(image_path)
    # Normaliza líneas si subtotal y precio unitario no son consistentes.
    normalized_lines: List[InvoiceLine] = []
    for line in invoice.lines:
        expected_subtotal = line.precio_unitario * line.cantidad
        if abs(expected_subtotal - line.subtotal) > 0.01 and line.cantidad != 0:
            corrected_price = line.subtotal / line.cantidad
            normalized_lines.append(
                line.model_copy(update={"precio_unitario": corrected_price})
            )
        else:
            normalized_lines.append(line)
    invoice = invoice.model_copy(update={"lines": normalized_lines})

    if not invoice.lines:
        raise ValueError("La factura no contiene líneas de productos para comparar.")
    order = _load_purchase_order(invoice)
    summary = _build_order_summary(order)

    # Preprocesa recepciones solo para validar cantidades
    receipt_by_product: Dict[int, Dict[str, float]] = {}
    for rec in summary["receipts"]:
        product_id = rec.get("product_id")
        if not product_id:
            continue
        info = receipt_by_product.setdefault(product_id, {"quantity": 0.0})
        info["quantity"] = float(info["quantity"]) + float(rec.get("quantity_done", 0.0))

    products: List[ProcessedProduct] = []
    for line in invoice.lines:
        product_result = _verify_line(line, summary["lines"])
        matched_line = _match_line(line, summary["lines"])

        if matched_line and matched_line.get("id"):
            desired_values: Dict[str, float] = {}
            if not product_result.cantidad_match:
                desired_values["product_qty"] = line.cantidad
            if not product_result.precio_match:
                desired_values["price_unit"] = line.precio_unitario
            # Si solo falla el subtotal, ajusta el precio unitario para que coincida con la factura.
            if not product_result.subtotal_match and "price_unit" not in desired_values:
                if line.cantidad != 0:
                    desired_values["price_unit"] = line.subtotal / line.cantidad

            if desired_values:
                odoo_manager.update_order_line(matched_line["id"], desired_values)
                refreshed_line = odoo_manager.read_order_lines([matched_line["id"]])[0]
                product_result.cantidad_match = (
                    abs(refreshed_line["cantidad"] - line.cantidad) <= 0.01
                )
                product_result.precio_match = (
                    abs(refreshed_line["precio_unitario"] - line.precio_unitario) <= 0.01
                )
                product_result.subtotal_match = (
                    abs(refreshed_line["subtotal"] - line.subtotal) <= 0.5
                )
                if product_result.cantidad_match and product_result.precio_match and product_result.subtotal_match:
                    product_result.issues = None
                else:
                    product_result.issues = (
                        product_result.issues
                        or "No se pudieron ajustar los valores de la línea en Odoo."
                    )

            receipt_info = receipt_by_product.get(matched_line.get("product_id"), {})
        else:
            receipt_info = {}

        if receipt_info:
            qty_received = float(receipt_info.get("quantity", 0.0))
            if abs(qty_received - line.cantidad) > 0.01:
                extra = f"Recepción: factura {line.cantidad} vs recibido {qty_received}."
                product_result.issues = f"{product_result.issues or ''} {extra}".strip()

        products.append(product_result)

    odoo_manager.recompute_order_amounts(order["id"])
    # Releer la orden tras las actualizaciones para usar totales y líneas frescos.
    summary = _build_order_summary(order)
    neto_match = abs(summary["neto"] - invoice.neto) <= 0.5
    iva_match = abs(summary["iva_19"] - invoice.iva_19) <= 0.5
    total_match = abs(summary["total"] - invoice.total) <= 0.5

    summary_lines = [
        f"Neto: {'OK' if neto_match else 'ISSUE'} (Factura {invoice.neto}, Odoo {summary['neto']})",
        f"19% IVA: {'OK' if iva_match else 'ISSUE'} (Factura {invoice.iva_19}, Odoo {summary['iva_19']})",
        f"Total: {'OK' if total_match else 'ISSUE'} (Factura {invoice.total}, Odoo {summary['total']})",
    ]
    for product in products:
        status = "OK" if not product.issues else "ISSUE"
        summary_lines.append(f"- {product.detalle}: {status}")
        if product.issues:
            summary_lines.append(f"  → {product.issues}")

    needs_follow_up = any(p.issues for p in products) or not (neto_match and iva_match and total_match)

    if not needs_follow_up:
        try:
            _finalize_order(order)
            summary_lines.append("✅ La orden fue recepcionada y facturada automáticamente en Odoo.")
        except Exception as exc:
            logger.error(f"No se pudo cerrar automáticamente la orden {order['name']}: {exc}")
            summary_lines.append("⚠️ No se pudo cerrar automáticamente la orden; revisa Odoo manualmente.")
            needs_follow_up = True

    return InvoiceResponseModel(
        summary="\n".join(summary_lines),
        products=products,
        needs_follow_up=needs_follow_up,
        neto_match=neto_match,
        iva_match=iva_match,
        total_match=total_match,
    )
