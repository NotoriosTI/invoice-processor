from typing import Any, Dict, Optional
import requests

from ..config import get_settings


class OdooClient:
    def __init__(self):
        settings = get_settings()
        self.base_url = settings.odoo_url.rstrip("/")
        self.db = settings.odoo_db
        self.username = settings.odoo_username
        self.password = settings.odoo_password
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})
        self._auth_token: Optional[str] = None

    def _authenticate(self):
        if self._auth_token:
            return
        resp = self.session.post(
            f"{self.base_url}/api/auth",
            json={"db": self.db, "login": self.username, "password": self.password},
            timeout=30,
        )
        resp.raise_for_status()
        token = resp.json()["token"]
        self._auth_token = token
        self.session.headers.update({"Authorization": f"Bearer {token}"})

    def find_quote(self, supplier: str, invoice_id: Optional[str], reference: Optional[str]) -> Optional[Dict[str, Any]]:
        """
        Busca una cotización u orden de compra en Odoo usando proveedor y referencias.
        Prioriza coincidencias exactas por referencia; como fallback usa proveedor + fecha/ID.
        """
        self._authenticate()
        payload = {
            "supplier": supplier,
            "invoice_id": invoice_id,
            "reference": reference,
        }
        resp = self.session.post(f"{self.base_url}/api/purchase/find_quote", json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        return data if data else None

    def confirm_quote(self, quote_id: int) -> Dict[str, Any]:
        """
        Confirma una cotización para convertirla en orden de compra (si aún está en draft).
        """
        self._authenticate()
        resp = self.session.post(
            f"{self.base_url}/api/purchase/{quote_id}/confirm",
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()

    def get_receipts_for_order(self, order_id: int) -> Dict[str, Any]:
        """
        Obtiene la información de recepción (stock.picking) asociada a una orden de compra,
        incluyendo cantidades recibidas por producto y ubicación.
        """
        self._authenticate()
        resp = self.session.get(
            f"{self.base_url}/api/purchase/{order_id}/receipts",
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()

    def get_product_type(self, product_id: int) -> str:
        """
        Devuelve el tipo de producto para determinar la ubicación esperada (materia prima / producto).
        """
        self._authenticate()
        resp = self.session.get(
            f"{self.base_url}/api/product/{product_id}/type",
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json().get("type", "producto")

    def fetch_purchase_summary(self, invoice_payload: dict) -> Dict[str, Any]:
        """
        Recibe un payload con los campos de la factura y devuelve la informacion equivalente en Odoo
        El endpoint debe responder con rl neto, iva, total y una lista de lineas(detalle, cantidad, precio_unitario, subtotal)
        """
        self._authenticate()
        resp = self.session.post(
            f"{self.base_url}/api/purchase/match_invoice",
            json=invoice_payload,
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()