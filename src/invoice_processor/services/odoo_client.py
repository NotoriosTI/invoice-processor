from typing import Any, Dict, Optional
import requests
from ..config import get_settings

class OdooClient:
    def __init__(self):
        settings = get_settings()
        self.base_url = settings.odoo_url.rstrip('/')
        self.db = settings.odoo_db
        self.username = settings.odoo_username
        self.password = settings.odoo_password
        self.sesion = requests.Session()
        self.session.headers.update({'Content-Type': 'application/json'})
        self._auth_token: Optional[str] = None

    def authenticate(self):
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
        self.session.headers.update["Authorization"] = f"Bearer {token}"

    def find_product(self, sku: Optional[str], name: str) -> Optional[Dict[str, Any]]:
        self._authenticate()
        params = {"default_code": sku} if sku else {"name": name}
        resp = self.session.get(f"{self.base_url}/api/product", params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        return data[0] if data else None

    def create_product(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        self._authenticate()
        resp = self.session.post(f"{self.base_url}/api/product", json=payload, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def update_inventory(self, product_id: int, quantity_delta: float) -> Dict[str, Any]:
        self._authenticate()
        resp = self.session.post(
            f"{self.base_url}/api/inventory/{product_id}/adjust",
            json={"quantity_delta": quantity_delta},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()