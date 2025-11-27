import logging
from difflib import SequenceMatcher
from typing import Optional, Dict, Any, List
from odoo_api import OdooProduct, OdooWarehouse
from ..config import get_settings

logger = logging.getLogger(__name__)
MIN_ORDER_SIMILARITY = 0.5


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
            "password": settings.odoo_password,
        }
        self._initialized = True

    def get_product_client(self) -> OdooProduct:
        if self._product_client is None:
            logger.info("Creando conexión OdooProduct...")
            self._product_client = OdooProduct(**self._connection_config)
        return self._product_client

    def get_warehouse_client(self) -> OdooWarehouse:
        if self._warehouse_client is None:
            logger.info("Creando conexión OdooWarehouse...")
            self._warehouse_client = OdooWarehouse(**self._connection_config)
        return self._warehouse_client

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
        logger.info("Cerrando todas las conexiones Odoo...")
        self.close_product_connection()
        self.close_warehouse_connection()

    def get_connection_status(self) -> Dict[str, Any]:
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

    # --- Utilidades de similitud para purchase orders ---

    @staticmethod
    def _string_similarity(left: Optional[str], right: Optional[str]) -> float:
        if not left or not right:
            return 0.0
        return SequenceMatcher(None, left.lower(), right.lower()).ratio()

    def _score_products(self, invoice_details: List[str], order_lines: List[dict]) -> float:
        if not invoice_details or not order_lines:
            return 0.0
        total = 0.0
        for detail in invoice_details:
            best = 0.0
            for line in order_lines:
                best = max(best, self._string_similarity(detail, line.get("detalle")))
            total += best
        return total / max(len(invoice_details), 1)

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

    def find_purchase_order_by_similarity(
        self,
        supplier_name: Optional[str],
        invoice_details: List[str],
        limit: int = 20,
    ) -> Optional[dict]:
        """Busca la OC más parecida combinando proveedor (nombre) y líneas."""
        domain: List[list] = [["state", "in", ["draft", "sent", "purchase", "done"]]]
        partner_ids: List[int] = []
        if supplier_name:
            partner_ids = self._execute_kw(
                "res.partner",
                "search",
                [[[ "name", "ilike", supplier_name ]]],
                {"limit": 5},
            )
        if partner_ids:
            domain.append(["partner_id", "in", partner_ids])


        order_ids = self._execute_kw(
            "purchase.order",
            "search",
            [domain],
            {"limit": limit, "order": "create_date desc"},
        )
        if not order_ids:
            return None

        fields = [
            "id",
            "name",
            "state",
            "amount_untaxed",
            "amount_tax",
            "amount_total",
            "order_line",
            "picking_ids",
            "partner_id",
        ]
        orders = self._execute_kw("purchase.order", "read", [order_ids], {"fields": fields})

        best_order: Optional[dict] = None
        best_score = 0.0
        for order in orders:
            lines = self.read_order_lines(order.get("order_line", []))
            product_score = self._score_products(invoice_details, lines)
            supplier_score = self._string_similarity(
                supplier_name,
                order["partner_id"][1] if order.get("partner_id") else None,
            )
            score = product_score * 0.7 + supplier_score * 0.3
            if score > best_score:
                best_score = score
                best_order = order

        if best_score < MIN_ORDER_SIMILARITY:
            logger.warning(
                "Similitud insuficiente para vincular la factura (score=%.2f).",
                best_score,
            )
            return None

        return best_order

    # --- Resto de utilidades existentes (confirmación, lectura, etc.) ---

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
                    "product_id": line["product_id"][0] if line.get("product_id") else None,
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
            {"fields": ["id", "state", "move_ids_without_package"]},
        )
        move_ids: List[int] = []
        for picking in pickings:
            move_ids.extend(picking.get("move_ids_without_package", []))
        if not move_ids:
            return []
        moves = self._execute_kw(
            "stock.move",
            "read",
            [move_ids],
            {"fields": ["product_id", "quantity_done", "location_dest_id"]},
        )
        parsed = []
        for move in moves:
            parsed.append(
                {
                    "product_id": move["product_id"][0] if move.get("product_id") else None,
                    "quantity_done": move.get("quantity_done", 0.0),
                    "location": move["location_dest_id"][1] if move.get("location_dest_id") else None,
                }
            )
        return parsed

    def get_product_type(self, product_id: int) -> str:
        """Devuelve el tipo de producto (materia prima vs producto terminado)."""
        product = self._execute_kw(
            "product.product",
            "read",
            [[product_id]],
            {"fields": ["type", "categ_id"]},
        )[0]
        category_name = product["categ_id"][1] if product.get("categ_id") else ""
        return "materia_prima" if "Materia Prima" in category_name else "producto"

odoo_manager = OdooConnectionManager()
