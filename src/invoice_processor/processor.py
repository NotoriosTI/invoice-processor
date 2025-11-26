from typing import Dict, List
from dev_utils.pretty_logger import PrettyLogger
from pydantic import ValidationError

from .services.ocr import InvoiceOcrClient
from .services.odoo_client import OdooClient
from .models import InvoiceData, InvoiceLine, ProcessedProduct, InvoiceResponseModel

logger = PrettyLogger(service_name="processor")
ocr_client = InvoiceOcrClient()
odoo_client = OdooClient()


def extract_invoice_data(image_path: str) -> InvoiceData:
    logger.info(f"Extrayendo datos de la factura {image_path}")
    raw = ocr_client.extract(image_path)
    if isinstance(raw, dict) and raw.get("error"):
        raise ValueError(f"OCR inválido: {raw['error']}")
    try:
        return (
            InvoiceData.model_validate_json(raw)
            if isinstance(raw, str)
            else InvoiceData.model_validate(raw)
        )
    except ValidationError as exc:
        raise ValueError(f"OCR no coincide con el formato esperado: {exc}") from exc


def _fetch_purchase_summary(invoice: InvoiceData) -> Dict:
    """
    Solicita a Odoo los datos equivalentes (cabecera y líneas).
    Se espera que el endpoint retorne:
    {
        "neto": float,
        "iva_19": float,
        "total": float,
        "lines": [
            {"detalle": str, "cantidad": float, "precio_unitario": float, "subtotal": float}
        ]
    }
    """
    payload = {
        "neto": invoice.neto,
        "iva_19": invoice.iva_19,
        "total": invoice.total,
        "lines": [line.model_dump() for line in invoice.lines],
    }
    return odoo_client.fetch_purchase_summary(payload)


def _match_line(invoice_line: InvoiceLine, odoo_lines: List[Dict]) -> Dict | None:
    for line in odoo_lines:
        if line["detalle"].strip().lower() == invoice_line.detalle.strip().lower():
            return line
    return None


def _verify_line(invoice_line: InvoiceLine, odoo_lines: List[Dict]) -> ProcessedProduct:
    odoo_line = _match_line(invoice_line, odoo_lines)
    if not odoo_line:
        return ProcessedProduct(
            detalle=invoice_line.detalle,
            cantidad_match=False,
            precio_match=False,
            subtotal_match=False,
            issues="Producto no encontrado en la orden de compra.",
        )

    cantidad_match = abs(odoo_line["cantidad"] - invoice_line.cantidad) <= 0.01
    precio_match = abs(odoo_line["precio_unitario"] - invoice_line.precio_unitario) <= 0.01
    subtotal_match = abs(odoo_line["subtotal"] - invoice_line.subtotal) <= 0.5

    issues: list[str] = []
    if not cantidad_match:
        issues.append(
            f"CANT: factura {invoice_line.cantidad} vs Odoo {odoo_line['cantidad']}."
        )
    if not precio_match:
        issues.append(
            f"P. UNITARIO: factura {invoice_line.precio_unitario} vs Odoo {odoo_line['precio_unitario']}."
        )
    if not subtotal_match:
        issues.append(
            f"Subtotal línea: factura {invoice_line.subtotal} vs Odoo {odoo_line['subtotal']}."
        )

    return ProcessedProduct(
        detalle=invoice_line.detalle,
        cantidad_match=cantidad_match,
        precio_match=precio_match,
        subtotal_match=subtotal_match,
        issues=" ".join(issues) if issues else None,
    )


def process_invoice_file(image_path: str) -> InvoiceResponseModel:
    invoice = extract_invoice_data(image_path)
    purchase_data = _fetch_purchase_summary(invoice)

    products = [
        _verify_line(line, purchase_data.get("lines", []))
        for line in invoice.lines
    ]

    neto_match = abs(purchase_data["neto"] - invoice.neto) <= 0.5
    iva_match = abs(purchase_data["iva_19"] - invoice.iva_19) <= 0.5
    total_match = abs(purchase_data["total"] - invoice.total) <= 0.5

    summary_lines = [
        f"Neto: {'OK' if neto_match else 'ISSUE'} (Factura {invoice.neto}, Odoo {purchase_data['neto']})",
        f"19% IVA: {'OK' if iva_match else 'ISSUE'} (Factura {invoice.iva_19}, Odoo {purchase_data['iva_19']})",
        f"Total: {'OK' if total_match else 'ISSUE'} (Factura {invoice.total}, Odoo {purchase_data['total']})",
    ]
    for product in products:
        status = "OK" if not product.issues else "ISSUE"
        summary_lines.append(f"- {product.detalle}: {status}")
        if product.issues:
            summary_lines.append(f"  → {product.issues}")

    return InvoiceResponseModel(
        summary="\n".join(summary_lines),
        products=products,
        needs_follow_up=any(p.issues for p in products) or not (neto_match and iva_match and total_match),
        neto_match=neto_match,
        iva_match=iva_match,
        total_match=total_match,
    )
