import logging
from typing import Optional, Dict, Any
from odoo_api import OdooProduct, OdooWarehouse
from ..config import get_settings

logger = logging.getLogger(__name__)

class OdooConnectionManager:
    """Manager centralizado para conexiones a Odoo con soporte para multiples APIs."""

    _instance: Optional["OdooConnectionManager"] = None
    _product_client: Optional[OdooProduct] = None
    _warehouse_client: Optional[OdooWarehouse] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if getattr(self, "_initialized", False):
            return
        settings = get_settings()
        self._connection_config = {
            "url": settings.odoo_url,
            "db": settings.odoo_db,
            "username": settings.odoo_username,
            "password": sttings.odoo_password,
        }
        self._initialized = True

    def get_product_client(self) -> OdooProduct:
        if self._product_client is None:
            logger.info("Creando conexión OdooProduct...")
            self.get_product_client = OdooProduct(**self._connection_config)
        return self._product_client

    def get_warehouse_client(self) -> OdooWarehouse:
        if self._warehouse_client is None:
            logger.info("Creando conexión OdooWarehouse...")
            self._warehouse_client = OdooWarehouse(**self._connection_config)

    def close_product_connection(self):
        if self._product_client:
            logger.info("Cerrando conexión OdooProduct...")
            if hasattr(self._product_client, "close"):
                self._product_client.close()
            self._product_client = None

    def close_warehouse_connection(self):
        if self._warehouse_client:
            logger.info("Cerrando conexión OdooWarehouse...")
            if hasattr(self._warehouse_client, "close"):
                self._warehouse_client.close()
            self._warehouse_client = None

    def close_all_connections(self):
        loger.info("Cerrando todas las conexiones Odoo...")
        self.close_product_connection()
        self.close_warehouse_connection()
    
    def get_connection_status(self)-> Dict[str, Any]:
        return {
            "product_connected": self._product_client is not None,
            "warehouse_connected": self._warehouse_client is not None,
            "total_connections": sum(
                [
                    1 if self._product_client else 0,
                    1 if self._warehouse_client else 0,
                ]
            ),
        }
    
    def test_connections(self) -> Dict[str, bool]:
        results = {}
        try:
            results["product"] = self._product_client is not None
        except Exception as exc:
            logger.error(f"Error testing OdooProduct connection: {exc}")
            results["product"] = False
        try:
            results["warehouse"] = self._warehouse_client is not None
        except Exception as exc:
            logger.error(f"Error testing OdooWarehouse connection: {exc}")
            results["warehouse"] = False
        return results
        
# --- Nuevas utilidades sobre purchase orders ---

    def _execute_kw(self, model, method, args=None, kwargs=None):
        """Atajo para llamar XML-RPC usando el product_client actual."""
        client = self.get_product_client()
        return client.models.execute_kw(
            client.db,
            client.uid,
            client.password,
            model,
            method,
            args or [],
            kwargs or {},
        )

    def find_purchase_order(self, reference: str) -> dict | None:
        """Busca una orden/cotización por su nombre (folio en Odoo)."""
        if not reference:
            return None
        order_ids = self._execute_kw(
            "purchase.order",
            "search",
            [[["name", "=", reference]]],
            {"limit": 1},
        )
        if not order_ids:
            return None
        order = self._execute_kw(
            "purchase.order",
            "read",
            [order_ids],
            {"fields": ["id", "name", "state", "amount_untaxed", "amount_tax", "amount_total", "order_line", "picking_ids"]},
        )[0]
        return order

    def confirm_purchase_order(self, order_id: int) -> dict:
        """Confirma una cotización y devuelve la orden actualizada."""
        self._execute_kw("purchase.order", "button_confirm", [[order_id]])
        order = self._execute_kw(
            "purchase.order",
            "read",
            [[order_id]],
            {"fields": ["id", "name", "state", "amount_untaxed", "amount_tax", "amount_total", "order_line", "picking_ids"]},
        )[0]
        return order

    def read_order_lines(self, order_line_ids: list[int]) -> list[dict]:
        """Devuelve las líneas de la orden con SKU, descripción, cantidad y precios."""
        if not order_line_ids:
            return []
        fields = ["id", "product_id", "name", "product_qty", "price_unit", "price_subtotal"]
        lines = self._execute_kw("purchase.order.line", "read", [order_line_ids], {"fields": fields})
        # Enriquecer con SKU/descripcion desde product.product
        product_ids = [line["product_id"][0] for line in lines if line.get("product_id")]
        products = {}
        if product_ids:
            product_records = self._execute_kw(
                "product.product",
                "read",
                [list(set(product_ids))],
                {"fields": ["id", "default_code", "name"]},
            )
            products = {prod["id"]: prod for prod in product_records}
        parsed = []
        for line in lines:
            product_info = products.get(line["product_id"][0]) if line.get("product_id") else {}
            parsed.append(
                {
                    "detalle": product_info.get("name") or line.get("name"),
                    "sku": product_info.get("default_code"),
                    "cantidad": line.get("product_qty", 0.0),
                    "precio_unitario": line.get("price_unit", 0.0),
                    "subtotal": line.get("price_subtotal", 0.0),
                }
            )
        return parsed

    def read_receipts_for_order(self, picking_ids: list[int]) -> list[dict]:
        """Lee los pickings asociados y devuelve cantidades recepcionadas y ubicación."""
        if not picking_ids:
            return []
        pickings = self._execute_kw(
            "stock.picking",
            "read",
            [picking_ids],
            {"fields": ["id", "name", "state", "location_dest_id", "move_ids_without_package"]},
        )
        move_ids = []
        for picking in pickings:
            move_ids.extend(picking.get("move_ids_without_package", []))
        moves = []
        if move_ids:
            move_records = self._execute_kw(
                "stock.move",
                "read",
                [move_ids],
                {"fields": ["product_id", "product_uom_qty", "quantity_done", "location_dest_id"]},
            )
            for move in move_records:
                moves.append(
                    {
                        "product_id": move["product_id"][0] if move.get("product_id") else None,
                        "quantity_done": move.get("quantity_done", 0.0),
                        "location": move["location_dest_id"][1] if move.get("location_dest_id") else None,
                    }
                )
        return moves

    def get_product_type(self, product_id: int) -> str:
        """Devuelve el tipo de producto (materia prima vs producto terminado)."""
        product = self._execute_kw(
            "product.product",
            "read",
            [[product_id]],
            {"fields": ["type", "categ_id"]},
        )[0]
        # Personaliza la lógica según tu clasificación
        category_name = product["categ_id"][1] if product.get("categ_id") else ""
        return "materia_prima" if "Materia Prima" in category_name else "producto"


odoo_manager = OdooConnectionManager()