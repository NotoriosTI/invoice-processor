import logging
from difflib import SequenceMatcher
from typing import Optional, Dict, Any, List, Tuple, TYPE_CHECKING
from odoo_api import OdooProduct, OdooWarehouse
from datetime import datetime
from ...config import get_settings

if TYPE_CHECKING:
    from ...core.models import InvoiceLine

logger = logging.getLogger(__name__)
MIN_ORDER_SIMILARITY = 0.5
MIN_PRODUCT_LINE_SIMILARITY = 0.6


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

    def _score_products(self, invoice_details: List[str], order_lines: List[dict]) -> Tuple[float, List[float]]:
        if not invoice_details or not order_lines:
            return 0.0, []
        scores: List[float] = []
        total = 0.0
        for detail in invoice_details:
            best = 0.0
            for line in order_lines:
                best = max(best, self._string_similarity(detail, line.get("detalle")))
            scores.append(best)
            total += best
        return total / max(len(invoice_details), 1), scores

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
        best_line_scores: List[float] = []
        best_line_count = 0
        for order in orders:
            lines = self.read_order_lines(order.get("order_line", []))
            product_score, line_scores = self._score_products(invoice_details, lines)
            supplier_score = self._string_similarity(
                supplier_name,
                order["partner_id"][1] if order.get("partner_id") else None,
            )
            score = product_score * 0.7 + supplier_score * 0.3
            if score > best_score:
                best_score = score
                best_order = order
                best_line_scores = line_scores
                best_line_count = len(lines)

        if best_score < MIN_ORDER_SIMILARITY or not best_order:
            logger.warning(
                "Similitud insuficiente para vincular la factura (score=%.2f).",
                best_score,
            )
            return None

        if not best_line_count:
            logger.warning(
                "La orden candidata %s no contiene líneas; se descartará.",
                best_order.get("name") if best_order else "",
            )
            return None

        max_line_similarity = max(best_line_scores) if best_line_scores else 0.0
        if invoice_details and max_line_similarity < MIN_PRODUCT_LINE_SIMILARITY:
            logger.warning(
                "La orden candidata %s no coincide con los productos de la factura (máxima similitud=%.2f).",
                best_order.get("name") if best_order else "",
                max_line_similarity,
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
                    "id": line["id"],
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
        try:
            moves = self._execute_kw(
                "stock.move",
                "read",
                [move_ids],
                {"fields": ["product_id", "product_uom_qty", "quantity_done", "location_dest_id"]},
            )
        except Exception as exc:  # algunas versiones no tienen quantity_done
            if "quantity_done" not in str(exc):
                raise
            moves = self._execute_kw(
                "stock.move",
                "read",
                [move_ids],
                {"fields": ["product_id", "product_uom_qty", "location_dest_id"]},
            )
        parsed = []
        for move in moves:
            qty_done = move.get("quantity_done")
            if qty_done is None:
                qty_done = move.get("product_uom_qty", 0.0)
            parsed.append(
                {
                    "product_id": move["product_id"][0] if move.get("product_id") else None,
                    "quantity_done": qty_done,
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

    def update_order_line(self, line_id: int, values: dict) -> None:
        self._execute_kw("purchase.order.line", "write", [[line_id], values])

    def recompute_order_amounts(self, order_id: int) -> None:
        """Recalcula los totales usando la lógica estándar de Odoo."""
        self._execute_kw(
            "purchase.order",
            "write",
            [[order_id], {}],
        )

    def _find_product_candidate(self, detail: str) -> Optional[int]:
        product_ids = self._execute_kw(
            "product.product",
            "search",
            [[["name", "ilike", detail]]],
            {"limit": 5},
        )
        if not product_ids:
            return None
        products = self._execute_kw(
            "product.product",
            "read",
            [product_ids],
            {"fields": ["id", "name"]},
        )
        if not products:
            return None
        best_product = max(
            products,
            key=lambda prod: self._string_similarity(detail, prod.get("name", "")),
        )
        logger.info(
            "Producto detectado por similitud: %s (ID %s)",
            best_product.get("name"),
            best_product.get("id"),
        )
        return best_product.get("id")

    def _ensure_supplier_on_product(self, product_id: int, supplier_id: int, price: float) -> None:
        product = self._execute_kw(
            "product.product",
            "read",
            [[product_id]],
            {"fields": ["seller_ids", "product_tmpl_id"]},
        )[0]
        seller_ids = product.get("seller_ids") or []
        template_raw = product.get("product_tmpl_id")
        template_id = template_raw[0] if template_raw else None
        if template_id is None:
            logger.warning("El producto %s no posee plantilla asociada en Odoo.", product_id)
            return
        if not seller_ids:
            self._execute_kw(
                "product.supplierinfo",
                "create",
                [[{"name": supplier_id, "product_tmpl_id": template_id, "price": price}]],
            )
            return
        supplier_infos = self._execute_kw(
            "product.supplierinfo",
            "read",
            [seller_ids],
            {"fields": ["id", "name"]},
        )
        if any(info.get("name") and info["name"][0] == supplier_id for info in supplier_infos):
            return
        self._execute_kw(
            "product.supplierinfo",
            "create",
            [[{"name": supplier_id, "product_tmpl_id": template_id, "price": price}]],
        )

    def ensure_product_for_supplier(self, invoice_line: "InvoiceLine", supplier_id: int) -> int:
        """Garantiza que exista un product_id compatible con el proveedor."""
        product_id = self._find_product_candidate(invoice_line.detalle)
        if product_id:
            self._ensure_supplier_on_product(product_id, supplier_id, invoice_line.precio_unitario)
            return product_id

        logger.info("Creando producto nuevo para %s", invoice_line.detalle)
        product_id = self._execute_kw(
            "product.product",
            "create",
            [[
                {
                    "name": invoice_line.detalle,
                    "type": "product",
                    "purchase_ok": True,
                    "seller_ids": [
                        (
                            0,
                            0,
                            {
                                "name": supplier_id,
                                "price": invoice_line.precio_unitario,
                            },
                        )
                    ],
                }
            ]],
        )
        return product_id


    def _select_supplier_candidate(self, supplier_name: str, candidate_ids: List[int]) -> Optional[int]:
        if not candidate_ids:
            return None
        partners = self._execute_kw(
            "res.partner",
            "read",
            [candidate_ids],
            {"fields": ["id", "name"]},
        )
        best_partner: Optional[dict] = None
        best_score = 0.0
        for partner in partners:
            score = self._string_similarity(supplier_name, partner.get("name"))
            if score > best_score:
                best_partner = partner
                best_score = score
        if best_partner:
            logger.info(
                "Proveedor detectado por similitud: %s (ID %s, score %.2f)",
                best_partner.get("name"),
                best_partner.get("id"),
                best_score,
            )
            return best_partner.get("id")
        return None

    def _prompt_supplier_id(self, supplier_name: str) -> Optional[int]:
        message = (
            f"No se encontró un proveedor que coincida con '{supplier_name}'.\n"
            "Ingresa el ID numérico de un proveedor existente en Odoo o deja vacío para crear uno nuevo: "
        )
        try:
            response = input(message)
        except (EOFError, KeyboardInterrupt):
            return None
        response = (response or "").strip()
        if not response:
            return None
        if not response.isdigit():
            logger.warning("El ID de proveedor proporcionado no es numérico: %s", response)
            return None
        supplier_id = int(response)
        existing = self._execute_kw(
            "res.partner",
            "search",
            [[["id", "=", supplier_id]]],
            {"limit": 1},
        )
        if existing:
            partner = self._execute_kw(
                "res.partner",
                "read",
                [[supplier_id]],
                {"fields": ["id", "name"]},
            )[0]
            logger.info(
                "Se utilizará el proveedor proporcionado manualmente: %s (ID %s)",
                partner.get("name"),
                supplier_id,
            )
            return supplier_id
        logger.warning("El ID %s no corresponde a un proveedor existente en Odoo.", supplier_id)
        return None

    def _confirm_supplier_creation(self, supplier_name: str) -> bool:
        prompt = (
            f"¿Deseas crear un nuevo proveedor en Odoo con el nombre '{supplier_name}'? [S/n]: "
        )
        try:
            answer = input(prompt)
        except (EOFError, KeyboardInterrupt):
            return False
        normalized = (answer or "").strip().lower()
        return normalized in {"", "s", "si", "sí"}

    def _find_or_create_supplier(self, supplier_name: str) -> int:
        supplier_label = (supplier_name or "Proveedor Desconocido").strip()
        partner_ids = self._execute_kw(
            "res.partner",
            "search",
            [[["name", "ilike", supplier_label]]],
            {"limit": 10},
        )
        selected = self._select_supplier_candidate(supplier_label, partner_ids)
        if selected:
            return selected

        manual_id = self._prompt_supplier_id(supplier_label)
        if manual_id:
            return manual_id

        if not self._confirm_supplier_creation(supplier_label):
            raise ValueError(
                "Se requiere un proveedor válido en Odoo para continuar. Ingresa el ID existente o acepta crear uno nuevo."
            )

        logger.info("Creando un nuevo proveedor en Odoo: %s", supplier_label)
        return self._execute_kw(
            "res.partner",
            "create",
            [[{"name": supplier_label, "company_type": "company"}]],
        )

    def create_purchase_order_from_invoice(self, invoice) -> dict:
        """
        Crea una orden de compra en Odoo usando la información extraída de la factura.
        Retorna la orden confirmada.
        """
        partner_id = self._find_or_create_supplier(invoice.supplier_name or "Proveedor Desconocido")
        today = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        lines = []
        for line in invoice.lines:
            product_id = self.ensure_product_for_supplier(line, partner_id)
            lines.append(
                (
                    0,
                    0,
                    {
                        "name": line.detalle,
                        "product_id": product_id,
                        "product_qty": line.cantidad,
                        "price_unit": line.precio_unitario,
                        "date_planned": today,
                    },
                )
            )


        try:
            order_id = self._execute_kw(
                "purchase.order",
                "create",
                [[
                    {
                        "partner_id": partner_id,
                        "date_order": today,
                        "origin": invoice.supplier_name,
                        "order_line": lines,
                    }
                ]],
            )
        except ValueError:
            raise
        except Exception as exc:
            logger.error("Error al crear la orden de compra en Odoo: %s", exc)
            raise RuntimeError(
                "Odoo rechazó la creación de la orden. Confirma que el proveedor es correcto o intenta ingresar su ID manual."
            ) from exc

        order = self.confirm_purchase_order(order_id)
        self.confirm_order_receipt(order)
        self.create_invoice_for_order(order_id)
        return order

    
    def confirm_order_receipt(self, order: dict) -> None:
        """Confirma la recepcion de todos los pickings asociados a la orden"""
        picking_ids = order.get("picking_ids", [])
        if not picking_ids:
            return
        self._execute_kw("stock.picking", "action_confirm", [picking_ids])
        self._execute_kw("stock.picking", "button_validate", [picking_ids])
    
    def create_invoice_for_order(self, order_id: int) -> None:
        """Genera la factura desde la orden de compra."""
        self._execute_kw("purchase.order", "action_create_invoice", [[order_id]])



    
odoo_manager = OdooConnectionManager()
