import logging
from difflib import SequenceMatcher
from typing import Optional, Dict, Any, List, Tuple, TYPE_CHECKING
from odoo_api import OdooProduct, OdooWarehouse, OdooSupply
from datetime import datetime
from ...config import get_settings
from rich.traceback import install

install()

if TYPE_CHECKING:
    from ...core.models import InvoiceLine

logger = logging.getLogger(__name__)
MIN_ORDER_SIMILARITY = 0.5
MIN_PRODUCT_LINE_SIMILARITY = 0.6
MIN_LINE_MATCH_RATIO = 0.6
MAX_TOTAL_MISMATCH_RATIO = 0.05


class OdooConnectionManager:
    """Manager centralizado para conexiones a Odoo con soporte para multiples APIs."""

    _instance: Optional["OdooConnectionManager"] = None
    _product_client: Optional[OdooProduct] = None
    _warehouse_client: Optional[OdooWarehouse] = None
    _supply_client: Optional[OdooSupply] = None

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
        raw_tax_ids = settings.default_purchase_tax_ids
        if isinstance(raw_tax_ids, str):
            raw_list = [x.strip() for x in raw_tax_ids.split(",") if x.strip()]
        else:
            raw_list = raw_tax_ids or []
        self._default_purchase_tax_ids = self._normalize_id_list(raw_list)
        self._default_uom_id: Optional[int] = None
        self._default_category_id: Optional[int] = None
        self._initialized = True
        self._supplierinfo_fields: Optional[Dict[str, dict]] = None

    @staticmethod
    def _normalize_id(value: Any) -> Optional[int]:
        """
        Convierte valores devueltos por Odoo a int seguro:
        - Si es lista/tupla anidada, toma el primer elemento numérico.
        - Si es (id, name), usa id.
        - Si es str numérica o float, castea a int.
        - Si no hay valor numérico, devuelve None.
        """
        if value is None:
            return None
        if isinstance(value, (list, tuple, set)):
            for item in value:
                norm = OdooConnectionManager._normalize_id(item)
                if norm is not None:
                    return norm
            return None
        if isinstance(value, (int, float)):
            return int(value)
        if isinstance(value, str):
            try:
                return int(float(value))
            except Exception:
                return None
        return None

    @staticmethod
    def _normalize_id_list(values: List[Any]) -> List[int]:
        """Aplana y convierte cualquier lista de IDs a enteros."""
        flat: List[int] = []
        for val in values or []:
            norm = OdooConnectionManager._normalize_id(val)
            if norm is not None:
                flat.append(norm)
        return flat

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

    def get_supply_client(self) -> OdooSupply:
        if self._supply_client is None:
            logger.info("Creando conexión OdooSupply...")
            self._supply_client = OdooSupply(**self._connection_config)
        return self._supply_client

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

    def _score_products(self, invoice_details: List[str], order_lines: List[dict]) -> Tuple[float, List[float], int]:
        if not invoice_details or not order_lines:
            return 0.0, [], 0
        scores: List[float] = []
        total = 0.0
        matched_lines = 0
        for detail in invoice_details:
            best = 0.0
            for line in order_lines:
                best = max(best, self._string_similarity(detail, line.get("detalle")))
            scores.append(best)
            total += best
            if best >= MIN_PRODUCT_LINE_SIMILARITY:
                matched_lines += 1
        return total / max(len(invoice_details), 1), scores, matched_lines

    def _execute_kw(self, model, method, args=None, kwargs=None):
        """Atajo para llamar XML-RPC usando el product_client actual.
        Si es una lectura, aplana posibles listas anidadas de IDs para evitar errores de hash.
        """
        client = self.get_product_client()
        safe_args = args or []
        if method == "read" and safe_args and isinstance(safe_args[0], list):
            safe_args = [self._normalize_id_list(safe_args[0])] + list(safe_args[1:])
        return client.models.execute_kw(
            client.db,
            client.uid,
            client.password,
            model,
            method,
            safe_args,
            kwargs or {},
        )

    def find_purchase_order_by_similarity(
        self,
        supplier_name: Optional[str],
        invoice_details: List[str],
        invoice_total: float,
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
        best_matched_lines = 0
        best_total = 0.0
        required_matches = max(1, int(len(invoice_details) * MIN_LINE_MATCH_RATIO))
        invoice_total_value = float(invoice_total or 0.0)
        for order in orders:
            lines = self.read_order_lines(order.get("order_line", []))
            product_score, line_scores, matched_lines = self._score_products(invoice_details, lines)
            supplier_score = self._string_similarity(
                supplier_name,
                order["partner_id"][1] if order.get("partner_id") else None,
            )
            order_total = order.get("amount_total") or 0.0
            if invoice_total_value <= 0.0:
                total_ratio = 0.0
            else:
                total_ratio = abs(order_total - invoice_total_value) / max(invoice_total_value, 1.0)
            total_match = 1.0 - min(total_ratio, 1.0)

            if total_ratio > MAX_TOTAL_MISMATCH_RATIO:
                logger.debug(
                    "Descartando orden %s por diferencia de total (factura %.2f vs Odoo %.2f).",
                    order.get("name"),
                    invoice_total_value,
                    order_total,
                )
                continue

            if matched_lines < required_matches:
                logger.debug(
                    "Descartando orden %s: solo %s líneas coinciden (requiere %s).",
                    order.get("name"),
                    matched_lines,
                    required_matches,
                )
                continue

            score = product_score * 0.6 + supplier_score * 0.3 + total_match * 0.1
            if score > best_score:
                best_score = score
                best_order = order
                best_line_scores = line_scores
                best_line_count = len(lines)
                best_matched_lines = matched_lines
                best_total = order_total

        if best_score < MIN_ORDER_SIMILARITY or not best_order:
            logger.warning(
                "Similitud insuficiente para vincular la factura (score=%.2f).",
                best_score,
            )
            return None

        if not best_order:
            logger.warning("No se encontró una orden compatible tras aplicar los criterios estrictos.")
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

        logger.info(
            "La orden %s fue seleccionada (score %.2f, líneas coincidentes %s/%s, total Odoo %.2f).",
            best_order.get("name"),
            best_score,
            best_matched_lines,
            len(invoice_details) or 1,
            best_total,
        )

        return best_order

    # --- Resto de utilidades existentes (confirmación, lectura, etc.) ---

    def confirm_purchase_order(self, order_id: int) -> dict:
        """Confirma una cotización y devuelve la orden actualizada."""
        order_id = self._normalize_id(order_id)
        self._execute_kw("purchase.order", "button_confirm", [[order_id]])
        order = self._execute_kw(
            "purchase.order",
            "read",
            [[order_id]],
            {"fields": ["id", "name", "state", "amount_untaxed", "amount_tax", "amount_total", "order_line", "picking_ids"]},
        )[0]
        return order

    def get_order_status(self, order_id: int) -> dict:
        """Obtiene estado, recepciones e invoices de una orden."""
        order_id = self._normalize_id(order_id)
        result = self._execute_kw(
            "purchase.order",
            "read",
            [[order_id]],
            {"fields": ["state", "picking_ids", "invoice_ids"]},
        )
        return result[0] if result else {}

    def read_order_lines(self, order_line_ids: list[int]) -> list[dict]:
        """Devuelve las líneas de la orden con SKU, descripción, cantidad y precios."""
        order_line_ids = self._normalize_id_list(order_line_ids)
        if not order_line_ids:
            return []
        fields = ["id", "product_id", "name", "product_qty", "price_unit", "price_subtotal", "taxes_id", "product_uom"]
        lines = self._execute_kw("purchase.order.line", "read", [order_line_ids], {"fields": fields})
        product_ids = [line["product_id"][0] for line in lines if line.get("product_id")]
        products = {}
        if product_ids:
            product_records = self._execute_kw(
                "product.product",
                "read",
                [self._normalize_id_list(list(set(product_ids)))],
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
                    "tax_ids": self._normalize_id_list(line.get("taxes_id") or []),
                    "product_uom": self._normalize_id(line.get("product_uom")),
                }
            )
        return parsed

    def read_order(self, order_id: int) -> dict:
        """Lee la orden de compra con sus totales y referencias clave."""
        order_id = self._normalize_id(order_id)
        result = self._execute_kw(
            "purchase.order",
            "read",
            [[order_id]],
            {"fields": ["id", "name", "state", "amount_untaxed", "amount_tax", "amount_total", "order_line", "picking_ids", "partner_id"]},
        )
        return result[0] if result else {}

    def read_receipts_for_order(self, picking_ids: list[int]) -> list[dict]:
        """Lee los pickings asociados y devuelve cantidades recepcionadas y ubicación."""
        picking_ids = self._normalize_id_list(picking_ids)
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
            move_ids.extend(self._normalize_id_list(picking.get("move_ids_without_package", [])))
        move_ids = self._normalize_id_list(move_ids)
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
            [[int(product_id)]],
            {"fields": ["type", "categ_id"]},
        )[0]
        category_name = product["categ_id"][1] if product.get("categ_id") else ""
        return "materia_prima" if "Materia Prima" in category_name else "producto"

    def update_order_line(self, line_id: int, values: dict) -> None:
        self._execute_kw("purchase.order.line", "write", [[int(line_id)], values])

    def recompute_order_amounts(self, order_id: int) -> None:
        """Recalcula los totales usando la lógica estándar de Odoo."""
        self._execute_kw(
            "purchase.order",
            "write",
            [[order_id], {}],
        )

    def _get_default_uom_id(self) -> int:
        if self._default_uom_id:
            return self._default_uom_id
        uom_ids = self._execute_kw("uom.uom", "search", [[]], {"limit": 1})
        if not uom_ids:
            raise RuntimeError("No se encontró una unidad de medida en Odoo para crear productos.")
        self._default_uom_id = uom_ids[0]
        return self._default_uom_id

    def _get_product_details(self, product_id: int) -> dict:
        """Lee datos básicos del producto necesarios para crear líneas."""
        product_id = self._normalize_id(product_id)
        products = self._execute_kw(
            "product.product",
            "read",
            [[product_id]],
            {"fields": ["uom_po_id", "uom_id", "supplier_taxes_id"]},
        )
        return products[0] if products else {}

    def _get_line_taxes(self, product_taxes: List[Any]) -> List[int]:
        taxes = self._normalize_id_list(product_taxes or [])
        if taxes:
            return taxes
        if self._default_purchase_tax_ids:
            return self._default_purchase_tax_ids
        return []

    def _get_default_category_id(self) -> int:
        if self._default_category_id:
            return self._default_category_id
        category_ids = self._execute_kw("product.category", "search", [[]], {"limit": 1})
        if not category_ids:
            raise RuntimeError("No se encontró una categoría de producto en Odoo para crear productos.")
        self._default_category_id = category_ids[0]
        return self._default_category_id

    def _get_supplierinfo_fields(self) -> Dict[str, dict]:
        if self._supplierinfo_fields is None:
            fields = self._execute_kw(
                "product.supplierinfo",
                "fields_get",
                [],
                {"attributes": ["required"]},
            )
            self._supplierinfo_fields = fields or {}
        return self._supplierinfo_fields

    def _supplierinfo_partner_field(self) -> str:
        fields = self._get_supplierinfo_fields()
        if "partner_id" in fields:
            return "partner_id"
        if "name" in fields:
            return "name"
        raise RuntimeError("No hay un campo para asociar el proveedor en product.supplierinfo.")

    def _supplierinfo_product_field(self) -> str:
        fields = self._get_supplierinfo_fields()
        if "product_tmpl_id" in fields:
            return "product_tmpl_id"
        if "product_id" in fields:
            return "product_id"
        raise RuntimeError("No hay un campo para asociar el producto en product.supplierinfo.")

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
            [[self._normalize_id(product_id)]],
            {"fields": ["seller_ids", "product_tmpl_id"]},
        )[0]
        seller_ids = self._normalize_id_list(product.get("seller_ids") or [])
        template_raw = product.get("product_tmpl_id")
        template_id = self._normalize_id(template_raw)
        if template_id is None:
            logger.warning("El producto %s no posee plantilla asociada en Odoo.", product_id)
            return
        partner_field = self._supplierinfo_partner_field()
        fields_to_read = ["id", partner_field]
        supplier_infos = self._execute_kw(
            "product.supplierinfo",
            "read",
            [self._normalize_id_list(seller_ids)],
            {"fields": fields_to_read},
        )
        for info in supplier_infos:
            partner_ref = info.get(partner_field)
            partner_ref_id = self._normalize_id(partner_ref)
            if partner_ref_id and partner_ref_id == self._normalize_id(supplier_id):
                return
        self._create_supplierinfo_record(template_id, product_id, supplier_id, price)

    def _create_supplierinfo_record(
        self,
        template_id: Optional[int],
        product_id: int,
        supplier_id: int,
        price: float,
    ) -> None:
        fields = self._get_supplierinfo_fields()
        partner_field = self._supplierinfo_partner_field()
        product_field = self._supplierinfo_product_field()

        if partner_field == "name":
            partner_data = self._execute_kw(
                "res.partner",
                "read",
                [[self._normalize_id(supplier_id)]],
                {"fields": ["name"]},
            )[0]
            partner_value: Any = [self._normalize_id(supplier_id), partner_data.get("name")]
        else:
            partner_value = self._normalize_id(supplier_id)

        values: Dict[str, Any] = {"price": price, "min_qty": 1, partner_field: partner_value}

        if product_field == "product_tmpl_id":
            if not template_id:
                raise RuntimeError(
                    "No se pudo determinar product_tmpl_id para crear la relación proveedor-producto."
                )
            values["product_tmpl_id"] = template_id
        else:
            values["product_id"] = product_id

        if "delay" in fields and "delay" not in values:
            values["delay"] = 1

        if "company_id" in fields and fields["company_id"].get("required"):
            partner = self._execute_kw(
                "res.partner",
                "read",
                [[self._normalize_id(supplier_id)]],
                {"fields": ["company_id"]},
            )[0]
            company_ref = partner.get("company_id")
            if company_ref:
                company_id = self._normalize_id(company_ref)
                if company_id is not None:
                    values["company_id"] = company_id

        self._execute_kw("product.supplierinfo", "create", [[values]])
        logger.info(
            "Se creó la relación proveedor-producto usando %s y %s",
            partner_field,
            product_field,
        )

    def ensure_product_for_supplier(self, invoice_line: "InvoiceLine", supplier_id: int) -> int:
        """Garantiza que exista un product_id compatible con el proveedor."""
        product_id = self._find_product_candidate(invoice_line.detalle)
        if product_id:
            self._ensure_supplier_on_product(product_id, supplier_id, invoice_line.precio_unitario)
            return product_id

        logger.info("Creando producto nuevo para %s", invoice_line.detalle)
        uom_id = self._get_default_uom_id()
        category_id = self._get_default_category_id()
        product_id = self._execute_kw(
            "product.product",
            "create",
            [[
                {
                    "name": invoice_line.detalle,
                    "type": "product",
                    "purchase_ok": True,
                    "sale_ok": False,
                    "uom_id": uom_id,
                    "uom_po_id": uom_id,
                    "categ_id": category_id,
                }
            ]],
        )
        self._ensure_supplier_on_product(product_id, supplier_id, invoice_line.precio_unitario)
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

    def _find_or_create_supplier(self, supplier_name: str) -> int:
        supplier_label = (supplier_name or "Proveedor Desconocido").strip()
        partner_ids = self._execute_kw(
            "res.partner",
            "search",
            [[["name", "ilike", supplier_label]]],
            {"limit": 10},
        )
        selected = self._select_supplier_candidate(supplier_label, self._normalize_id_list(partner_ids))
        if selected:
            # Si existe pero no está marcado como proveedor, se actualiza supplier_rank=1.
            try:
                selected_id = self._normalize_id(selected)
                partner = self._execute_kw(
                    "res.partner",
                    "read",
                    [[selected_id]],
                    {"fields": ["supplier_rank", "name"]},
                )[0]
                if (partner.get("supplier_rank") or 0) < 1:
                    logger.info(
                        "Actualizando supplier_rank=1 para el proveedor existente '%s' (ID %s).",
                        partner.get("name"),
                        selected_id,
                    )
                    self._execute_kw(
                        "res.partner",
                        "write",
                        [[selected_id], {"supplier_rank": 1}],
                    )
            except Exception as exc:
                logger.warning(
                    "No se pudo verificar/actualizar supplier_rank del proveedor %s: %s",
                    selected_id,
                    exc,
                )
            return selected
        logger.info(
            "No se encontró proveedor para '%s'. Se creará automáticamente uno nuevo.",
            supplier_label,
        )
        try:
            return self._execute_kw(
                "res.partner",
                "create",
                [[{"name": supplier_label, "company_type": "company", "supplier_rank": 1}]],
            )
        except Exception as exc:
            logger.error("No fue posible crear el proveedor %s: %s", supplier_label, exc)
            raise RuntimeError(
                "Odoo rechazó la creación automática del proveedor. Configura el proveedor manualmente en Odoo e intenta nuevamente."
            ) from exc

    def create_purchase_order_from_invoice(self, invoice) -> dict:
        """
        Crea una orden de compra en Odoo usando la información extraída de la factura.
        Retorna la orden confirmada.
        """
        partner_id = self._find_or_create_supplier(invoice.supplier_name or "Proveedor Desconocido")
        today = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        line_values: List[dict] = []
        for line in invoice.lines:
            product_id = self.ensure_product_for_supplier(line, partner_id)
            product_data = self._get_product_details(product_id)
            taxes = self._get_line_taxes(product_data.get("supplier_taxes_id", []))
            uom = self._normalize_id(product_data.get("uom_po_id") or product_data.get("uom_id") or self._get_default_uom_id())
            line_values.append(
                {
                    "name": line.detalle,
                    "product_id": product_id,
                    "product_qty": line.cantidad,
                    "price_unit": line.precio_unitario,
                    "product_uom": uom,
                    "taxes_id": [[6, 0, taxes]] if taxes else [],
                    "date_planned": today,
                }
            )

        supply = self.get_supply_client()
        logger.info(
            "Intentando crear orden en Odoo (OdooSupply) para proveedor %s con %s líneas.",
            partner_id,
            len(line_values),
        )
        try:
            result = supply.create_rfq(
                vendor_id=partner_id,
                order_lines=line_values,
                rfq_values={"date_order": today, "origin": invoice.supplier_name},
                confirm=True,
            )
            order_id = result.get("id")
        except ValueError:
            raise
        except Exception as exc:
            logger.error("Error al crear la orden en Odoo usando OdooSupply: %s", exc)
            raise RuntimeError(
                "Odoo rechazó la creación de la orden. Revisa los datos del proveedor/producto e intenta nuevamente."
            ) from exc

        order = self._execute_kw(
            "purchase.order",
            "read",
            [[order_id]],
            {"fields": ["id", "name", "state", "amount_untaxed", "amount_tax", "amount_total", "order_line", "picking_ids", "partner_id"]},
        )[0]
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
