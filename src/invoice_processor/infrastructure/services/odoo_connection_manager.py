from __future__ import annotations
import logging
import os
import re
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path
from types import SimpleNamespace
from typing import Optional, Dict, Any, List, Tuple, TYPE_CHECKING
from odoo_api import OdooProduct, OdooWarehouse, OdooSupply
from datetime import datetime
from ...config import get_settings
from .embedding_service import EmbeddingService, choose_best_match_with_fallback
from rich.traceback import install

install()

if TYPE_CHECKING:
    from ...core.models import InvoiceLine

logger = logging.getLogger(__name__)
MIN_ORDER_SIMILARITY = 0.5
MIN_PRODUCT_LINE_SIMILARITY = 0.6
MIN_LINE_MATCH_RATIO = 0.6
MAX_TOTAL_MISMATCH_RATIO = 0.05
MIN_PRODUCT_EMBEDDING_SCORE = 0.7
PRODUCT_EMBEDDING_CACHE_FILE = "product_embeddings.pkl"
MIN_PRODUCT_CONFIDENCE = 0.9
MAX_PRODUCT_CANDIDATES = 5

SYNONYM_MAP: Dict[str, str] = {
    "cj": "caja",
    "und": "unidad",
    "un": "unidad",
    "pza": "pieza",
    "lt": "litro",
    "gr": "gramo",
}

STOPWORDS: set[str] = {
    "mp",
    "aceite",
    "aceites",
    "esencial",
    "esenciales",
    "exfoliantes",
    "semillas",
    "harinas",
    "producto",
    "materia",
    "prima",
    "efervescentes",
    "sales",
    "sello",
    "tapa",
    "vidrio",
    "transparente",
    "pote",
    "crema",
    "blanca",
    "mesh",
    "saco",
    "kg",
    "g",
    "gr",
    "ml",
    "l",
    "lt",
    "lote",
    "vto",
    "fv",
    "neto",
    "cod",
}


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
        self._settings = settings
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
        self._tag_cache: Dict[int, str] = {}
        self._location_cache: Dict[str, Optional[int]] = {}
        self._has_product_tag_field: Optional[bool] = None
        self._product_tag_by_product: Dict[int, List[int]] = {}
        self._embedding_cache_path = settings.data_path / PRODUCT_EMBEDDING_CACHE_FILE
        self._embedding_cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._embedding_service: Optional[EmbeddingService] = None
        self._product_embeddings_cache: Optional[Dict[str, Any]] = None
        self._product_embeddings_cache_by_supplier: Dict[int, Dict[str, Any]] = {}

    def _resolve_supplier_id(
        self,
        supplier_id: Optional[int],
        supplier_name: Optional[str],
        supplier_rut: Optional[str] = None,
    ) -> Optional[int]:
        if supplier_id:
            return int(supplier_id)
        supplier_label = (supplier_name or "").strip()
        normalized_rut = self._normalize_vat_strict(supplier_rut)
        raw_rut = supplier_rut.strip() if isinstance(supplier_rut, str) else None

        conditions: List[Tuple[str, str, Any]] = []
        if supplier_label:
            conditions.append(("name", "ilike", supplier_label))
        if normalized_rut:
            conditions.append(("vat", "ilike", normalized_rut))
        if raw_rut and raw_rut != normalized_rut:
            conditions.append(("vat", "ilike", raw_rut))

        if not conditions:
            return None

        try:
            candidates = self._execute_kw(
                "res.partner",
                "search_read",
                [self._build_or_domain(conditions)],
                {"fields": ["id", "name", "vat"], "limit": 5},
            )
        except Exception:
            return None

        if not candidates:
            return None

        if normalized_rut:
            for cand in candidates:
                cand_vat = self._normalize_vat_strict(cand.get("vat"))
                if cand_vat and cand_vat == normalized_rut:
                    return int(cand.get("id"))

        if supplier_label:
            best = None
            best_score = 0.0
            for cand in candidates or []:
                score = self._string_similarity(supplier_label, cand.get("name"))
                if score > best_score:
                    best_score = score
                    best = cand
            if best:
                return int(best.get("id"))

        return int(candidates[0].get("id"))

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
    def _sanitize_default_code(value: Any) -> Optional[str]:
        """Normaliza un SKU/default_code (quita espacios/puntuación típica de texto libre)."""
        if value in (None, False):
            return None
        text = str(value).strip()
        if not text:
            return None
        # Remueve wrappers típicos en Slack/Markdown.
        text = text.strip("`'\"").strip()
        # Remueve espacios internos.
        text = re.sub(r"\s+", "", text)
        # Remueve puntuación/ruido al inicio/fin (ej: "MP026.", "(MP026)").
        text = re.sub(r"^[^A-Za-z0-9]+", "", text)
        text = re.sub(r"[^A-Za-z0-9_-]+$", "", text)
        text = text.upper()
        return text or None

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

    def _get_embedding_service(self) -> EmbeddingService:
        if self._embedding_service is None:
            self._embedding_service = EmbeddingService(
                model_name=getattr(self._settings, "embedding_model", "text-embedding-3-small"),
                api_key=self._settings.openai_api_key,
                cache_path=self._embedding_cache_path,
            )
        return self._embedding_service

    def _embedding_cache_path_for_supplier(self, supplier_id: Optional[int]) -> Path:
        if supplier_id is None:
            return self._embedding_cache_path
        return self._settings.data_path / f"product_embeddings_supplier_{supplier_id}.pkl"

    def _refresh_product_embeddings(
        self,
        force: bool = False,
        supplier_id: Optional[int] = None,
        supplier_name: Optional[str] = None,
    ) -> None:
        """Genera o carga embeddings; si supplier_id/nombre están presentes, usa caché filtrada por proveedor."""
        service = self._get_embedding_service()
        resolved_supplier_id = self._resolve_supplier_id(supplier_id, supplier_name)
        cache_path = self._embedding_cache_path_for_supplier(resolved_supplier_id)

        target_cache = (
            self._product_embeddings_cache_by_supplier.get(int(resolved_supplier_id))
            if resolved_supplier_id is not None
            else self._product_embeddings_cache
        )

        if target_cache and not force:
            return

        if not force:
            cached = service.load_cache(cache_path)
            if cached:
                if resolved_supplier_id is None:
                    self._product_embeddings_cache = cached
                else:
                    self._product_embeddings_cache_by_supplier[int(resolved_supplier_id)] = cached
                return

        domain = None
        if resolved_supplier_id is not None:
            domain = [["seller_ids.partner_id", "=", int(resolved_supplier_id)]]
        elif supplier_name:
            domain = [["seller_ids.name", "ilike", supplier_name]]

        supplier_product_ids: List[int] = []
        if resolved_supplier_id is not None:
            try:
                supplier_product_ids = self._execute_kw(
                    "product.product",
                    "search",
                    [[["seller_ids.partner_id", "=", int(resolved_supplier_id)]]],
                    {"limit": 1000},
                ) or []
            except Exception:
                supplier_product_ids = []
        elif supplier_name:
            try:
                supplier_product_ids = self._execute_kw(
                    "product.product",
                    "search",
                    [[["seller_ids.name", "ilike", supplier_name]]],
                    {"limit": 1000},
                ) or []
            except Exception:
                supplier_product_ids = []

        try:
            df_products = self.read_products_for_embeddings(domain=domain)
        except TypeError:
            logger.info("read_products_for_embeddings no acepta domain; se usará sin filtro.")
            try:
                df_products = self.read_products_for_embeddings(domain=None)
            except Exception as exc:
                logger.warning("No se pudieron leer productos para embeddings: %s", exc)
                return
        except Exception as exc:
            logger.warning("No se pudieron leer productos para embeddings: %s", exc)
            return
        if df_products is None:
            logger.warning("read_products_for_embeddings devolvió None.")
            return

        try:
            if hasattr(df_products, "__getitem__") and hasattr(df_products, "to_dict"):
                texts = df_products["text_for_embedding"].fillna("").astype(str).tolist()
                ids = [int(x) for x in df_products["id"].tolist()]
                if supplier_product_ids:
                    supplier_ids_set = {
                        sid for sid in (self._normalize_id(x) for x in supplier_product_ids) if sid is not None
                    }
                    mask = df_products["id"].astype(int).isin(supplier_ids_set)
                    texts = df_products.loc[mask, "text_for_embedding"].fillna("").astype(str).tolist()
                    ids = [int(x) for x in df_products.loc[mask, "id"].tolist()]
            else:
                texts = [str(item.get("text_for_embedding", "")) for item in df_products]
                ids = [int(item.get("id")) for item in df_products if item.get("id") is not None]
                if supplier_product_ids:
                    supplier_ids_set = {
                        sid for sid in (self._normalize_id(x) for x in supplier_product_ids) if sid is not None
                    }
                    filtered = [
                        (text, pid)
                        for text, pid in zip(texts, ids)
                        if pid in supplier_ids_set
                    ]
                    texts = [t for t, _ in filtered]
                    ids = [pid for _, pid in filtered]
        except Exception as exc:
            logger.warning("Formato inesperado en productos para embeddings: %s", exc)
            return

        if not ids or not texts or len(ids) != len(texts):
            logger.warning(
                "Productos para embeddings incompletos para supplier_id=%s; ids=%s, textos=%s",
                supplier_id,
                len(ids),
                len(texts),
            )
            return
        if supplier_product_ids and not ids:
            logger.info(
                "No se encontraron productos del proveedor %s en dataset embeddings; se usará índice global.",
                resolved_supplier_id or supplier_name,
            )

        try:
            vectors = service.embed_documents(texts)
        except Exception as exc:
            logger.warning("Error generando embeddings de productos: %s", exc)
            return

        service.save_cache(ids, vectors, path=cache_path)
        payload = {"ids": ids, "vectors": vectors}
        if supplier_id is None:
            self._product_embeddings_cache = payload
        else:
            self._product_embeddings_cache_by_supplier[int(supplier_id)] = payload

    def _find_product_candidate_by_embedding(
        self,
        invoice_line: "InvoiceLine",
        supplier_id: Optional[int] = None,
        supplier_name: Optional[str] = None,
        return_score: bool = False,
    ):
        detail = getattr(invoice_line, "detalle", None) or ""
        if not detail.strip():
            return None
        try:
            if supplier_id or supplier_name:
                self._refresh_product_embeddings(force=False, supplier_id=supplier_id, supplier_name=supplier_name)
            self._refresh_product_embeddings(force=False)
        except Exception as exc:
            logger.warning("No se pudo refrescar la caché de embeddings: %s", exc)
            return None

        caches = []
        resolved_supplier_id = self._resolve_supplier_id(supplier_id, supplier_name)
        if resolved_supplier_id:
            caches.append(self._product_embeddings_cache_by_supplier.get(int(resolved_supplier_id)))
        caches.append(self._product_embeddings_cache)

        try:
            match = choose_best_match_with_fallback(
                self._get_embedding_service(),
                detail,
                caches,
                MIN_PRODUCT_EMBEDDING_SCORE,
            )
        except Exception as exc:
            logger.warning("No se pudo realizar la búsqueda por embeddings: %s", exc)
            return None

        if not match:
            return None
        product_id, score = match
        logger.info("Producto detectado por embeddings: ID %s (score %.2f)", product_id, score)
        if return_score:
            return product_id, score
        return product_id

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

    @staticmethod
    def _build_or_domain(conditions: List[Tuple[str, str, Any]]) -> List[Any]:
        if not conditions:
            return []
        if len(conditions) == 1:
            return [conditions[0]]
        return ["|"] * (len(conditions) - 1) + conditions

    def _normalize_vat_strict(self, vat: Optional[str]) -> Optional[str]:
        if not vat:
            return None
        cleaned = vat.strip().upper()
        if cleaned.startswith("CL"):
            cleaned = cleaned[2:]
        cleaned = "".join(ch for ch in cleaned if ch.isalnum())
        return cleaned or None

    def _normalize_vat(self, vat: Optional[str]) -> Optional[str]:
        return self._normalize_vat_strict(vat)

    def find_purchase_order_by_similarity(
        self,
        supplier_name: Optional[str],
        invoice_details: List[str],
        invoice_total: float,
        limit: int = 20,
        supplier_rut: Optional[str] = None,
    ) -> Optional[dict]:
        """Busca la OC más parecida combinando proveedor (nombre) y líneas."""
        domain: List[list] = [["state", "in", ["draft", "sent", "purchase", "done"]]]
        partner_ids: List[int] = []
        normalized_rut = self._normalize_vat(supplier_rut)
        if normalized_rut:
            partner_ids = self._execute_kw(
                "res.partner",
                "search",
                [[["vat", "=", normalized_rut]]],
                {"limit": 5},
            )
        if supplier_name and not partner_ids:
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
            "invoice_status",
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
            invoice_status = order.get("invoice_status")
            if invoice_status and str(invoice_status).lower() in {"invoiced", "facturado"}:
                logger.debug("Descartando orden %s por estado de facturación=%s.", order.get("name"), invoice_status)
                continue
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
        fields = [
            "id",
            "product_id",
            "name",
            "product_qty",
            "price_unit",
            "price_subtotal",
            "taxes_id",
            "product_uom",
        ]
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
            line_name = line.get("name")
            parsed.append(
                {
                    "id": line["id"],
                    "detalle": line_name or product_info.get("name"),
                    "line_name": line_name,
                    "product_name": product_info.get("name"),
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
            {"fields": ["id", "state", "move_ids_without_package", "location_dest_id"]},
        )
        parsed = []
        for picking in pickings:
            move_ids = self._normalize_id_list(picking.get("move_ids_without_package", []))
            if not move_ids:
                continue
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
            for move in moves:
                qty_done = move.get("quantity_done")
                if qty_done is None:
                    qty_done = move.get("product_uom_qty", 0.0)
                parsed.append(
                    {
                        "product_id": move["product_id"][0] if move.get("product_id") else None,
                        "quantity_done": qty_done,
                        "location": move["location_dest_id"][1] if move.get("location_dest_id") else None,
                        "location_id": self._normalize_id(move.get("location_dest_id")),
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

    def _get_location_id_by_name(self, name: str) -> Optional[int]:
        if name in self._location_cache:
            return self._location_cache[name]
        try:
            ids = self._execute_kw(
                "stock.location",
                "search",
                [[["complete_name", "ilike", name]]],
                {"limit": 1},
            )
            loc_id = self._normalize_id(ids[0]) if ids else None
            self._location_cache[name] = loc_id
            return loc_id
        except Exception as exc:
            logger.warning("No se pudo resolver ubicación '%s': %s", name, exc)
            self._location_cache[name] = None
            return None

    def _product_tags_for_ids(self, product_ids: List[int]) -> Dict[int, List[int]]:
        """Devuelve tags por product_id leyendo product.product; cachea resultados."""
        result: Dict[int, List[int]] = {}
        if self._has_product_tag_field is False:
            return result
        if self._has_product_tag_field is None:
            try:
                fields = self._execute_kw(
                    "product.product",
                    "fields_get",
                    [],
                    {"attributes": ["type"]},
                )
                self._has_product_tag_field = "product_tag_ids" in fields
            except Exception:
                self._has_product_tag_field = False
        if not self._has_product_tag_field:
            return result

        to_fetch = []
        for pid in product_ids or []:
            pid_norm = self._normalize_id(pid)
            if pid_norm is None:
                continue
            if pid_norm in self._product_tag_by_product:
                result[pid_norm] = self._product_tag_by_product.get(pid_norm, [])
            else:
                to_fetch.append(pid_norm)
        if not to_fetch:
            return result
        try:
            records = self._execute_kw(
                "product.product",
                "read",
                [to_fetch],
                {"fields": ["id", "product_tag_ids"]},
            )
            for rec in records or []:
                pid = self._normalize_id(rec.get("id"))
                tags = self._normalize_id_list(rec.get("product_tag_ids") or [])
                if pid is not None:
                    self._product_tag_by_product[pid] = tags
                    result[pid] = tags
        except Exception:
            pass
        return result

    # --- Helpers de producto basados en OdooProduct (solo lectura/utilidad) ---
    def product_exists(self, sku: str) -> bool:
        """Valida existencia de producto por SKU usando OdooProduct si está disponible; fallback a search."""
        client = self.get_product_client()
        try:
            if hasattr(client, "product_exists"):
                return bool(client.product_exists(sku))
        except Exception:
            pass
        ids = self._execute_kw(
            "product.product",
            "search",
            [[["default_code", "=", str(sku).strip()]]],
            {"limit": 1},
        )
        return bool(ids)

    def get_product_id_by_sku(self, sku: str) -> Optional[int]:
        """Devuelve product_id por SKU; fallback a search si el helper no está disponible."""
        client = self.get_product_client()
        try:
            if hasattr(client, "get_id_by_sku"):
                res = client.get_id_by_sku(sku)
                if isinstance(res, dict):
                    return self._normalize_id(res.get("product_id"))
        except Exception:
            pass
        ids = self._execute_kw(
            "product.product",
            "search",
            [[["default_code", "=", str(sku).strip()]]],
            {"limit": 1},
        )
        return self._normalize_id(ids[0]) if ids else None

    def get_sku_by_product_id(self, product_id: int) -> Optional[str]:
        """Devuelve SKU (default_code) por product_id; fallback a search_read."""
        client = self.get_product_client()
        try:
            if hasattr(client, "get_sku_by_id"):
                res = client.get_sku_by_id(product_id)
                if isinstance(res, str):
                    return res
        except Exception:
            pass
        recs = self._execute_kw(
            "product.product",
            "search_read",
            [[["id", "=", self._normalize_id(product_id)]]],
            {"fields": ["default_code"], "limit": 1},
        )
        if recs and isinstance(recs, list) and recs[0].get("default_code"):
            return recs[0]["default_code"]
        return None

    def read_all_product_tags_cached(self) -> Dict[int, str]:
        """Lee todas las etiquetas de producto y retorna un mapa id->name; usa OdooProduct si está disponible."""
        if self._tag_cache:
            return {k: v for k, v in self._tag_cache.items() if v}
        client = self.get_product_client()
        tags = []
        try:
            if hasattr(client, "read_all_product_tags"):
                df = client.read_all_product_tags()
                try:
                    tags = df.to_dict(orient="records")  # si viene DataFrame
                except Exception:
                    tags = df
        except Exception:
            pass
        if not tags:
            try:
                tags = self._execute_kw(
                    "product.tag",
                    "search_read",
                    [[]],
                    {"fields": ["id", "name"]},
                )
            except Exception:
                tags = []
        for tag in tags or []:
            tid = self._normalize_id(tag.get("id"))
            tname = tag.get("name")
            if tid:
                self._tag_cache[tid] = tname
        return {k: v for k, v in self._tag_cache.items() if v}

    def get_active_skus(self) -> set[str]:
        """Retorna SKUs activos usando helper de OdooProduct si existe; fallback a search_read."""
        client = self.get_product_client()
        try:
            if hasattr(client, "get_active_skus"):
                skus = client.get_active_skus()
                return set(skus) if skus else set()
        except Exception:
            pass
        try:
            products = self._execute_kw(
                "product.product",
                "search_read",
                [[["active", "=", True], ["default_code", "!=", False]]],
                {"fields": ["default_code"]},
            )
            return {
                prod["default_code"]
                for prod in products
                if isinstance(prod, dict) and prod.get("default_code")
            }
        except Exception:
            return set()

    def read_products_for_embeddings(self, domain: Optional[list] = None):
        """Proxy a OdooProduct.read_products_for_embeddings si está disponible; caso contrario, devuelve None."""
        client = self.get_product_client()
        try:
            if hasattr(client, "read_products_for_embeddings"):
                return client.read_products_for_embeddings(domain=domain)
        except Exception:
            return None
        return None
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

    def _search_product_native_optimized(
        self,
        query: str,
        supplier_id: Optional[int] = None,
        limit: int = 10,
    ) -> List[dict]:
        if not query or not query.strip():
            return []
        raw_query = query.strip()
        cleaned = re.sub(r"\s+", " ", raw_query.lower())
        tokens = [
            SYNONYM_MAP.get(token, token)
            for token in cleaned.split()
            if len(token) >= 3 and token not in STOPWORDS
        ]
        conditions: List[Tuple[str, str, str]] = [
            ("default_code", "=ilike", raw_query),
            ("default_code", "ilike", raw_query),
        ]
        for token in tokens:
            conditions.append(("name", "ilike", token))

        if not conditions:
            return []

        if len(conditions) == 1:
            domain: List[Any] = [conditions[0]]
        else:
            domain = ["|"] * (len(conditions) - 1) + conditions

        resolved_supplier_id = self._normalize_id(supplier_id) if supplier_id is not None else None
        if resolved_supplier_id is not None:
            domain = ["|", ("seller_ids.partner_id", "=", resolved_supplier_id)] + domain

        try:
            return self._execute_kw(
                "product.product",
                "search_read",
                [domain],
                {"fields": ["id", "name", "default_code", "seller_ids"], "limit": limit},
            )
        except Exception as exc:
            logger.warning("Búsqueda nativa falló para '%s': %s", query, exc)
            return []

    def _find_product_candidate(
        self,
        invoice_line: "InvoiceLine",
        supplier_id: Optional[int] = None,
        supplier_name: Optional[str] = None,
    ) -> Optional[int]:
        def _normalize(text: str) -> str:
            if not text:
                return ""
            text = text.lower()
            text = re.sub(r"\[[^\]]+\]", " ", text)
            text = "".join(
                ch
                for ch in unicodedata.normalize("NFD", text)
                if unicodedata.category(ch) != "Mn"
            )
            text = re.sub(r"[^a-z0-9\s]", " ", text)
            text = re.sub(r"\s+", " ", text).strip()
            return text

        def _tokens(text: str) -> List[str]:
            normalized = _normalize(text)
            if not normalized:
                return []
            words: List[str] = []
            for word in normalized.split():
                if not word or word in STOPWORDS:
                    continue
                words.append(SYNONYM_MAP.get(word, word))
            return words

        def _extract_quantity(text: str) -> Optional[Tuple[float, str]]:
            pattern = re.compile(r"(\d+(?:[\.,]\d+)?)\s*(kg|g|gr|gramos?|ml|l|lt|cc)", re.IGNORECASE)
            match = pattern.search(text or "")
            if not match:
                return None
            raw_val = match.group(1).replace(",", ".")
            try:
                val = float(raw_val)
            except ValueError:
                return None
            unit = match.group(2).lower()
            if unit in {"kg"}:
                return val * 1000.0, "g"
            if unit in {"l", "lt"}:
                return val * 1000.0, "ml"
            if unit == "cc":
                return val, "cc"
            return val, unit

        def _normalize_unit(value: Optional[str]) -> Optional[str]:
            if not value:
                return None
            unit = value.strip().lower()
            mapping = {
                "kg": {"kg", "kilo", "kilos"},
                "g": {"g", "gr", "gramo", "gramos"},
                "l": {"l", "lt", "litro", "litros"},
                "ml": {"ml", "mililitro", "mililitros"},
                "unidad": {"unidad", "unidades", "ud", "uds", "un"},
                "saco": {"saco", "sacos"},
                "caja": {"caja", "cajas"},
            }
            for norm, variants in mapping.items():
                if unit in variants:
                    return norm
            return unit

        def _score_candidate(
            target_norm: str,
            cand_norm: str,
            target_tokens: List[str],
            cand_tokens: List[str],
            target_qty,
            cand_qty,
            target_unit: Optional[str],
        ) -> float:
            base = SequenceMatcher(None, target_norm, cand_norm).ratio()
            if target_tokens:
                overlap = len(set(target_tokens) & set(cand_tokens))
                token_score = overlap / max(len(target_tokens), 1)
            else:
                token_score = 0.0
            qty_bonus = 0.0
            if target_qty and cand_qty:
                t_val, t_unit = target_qty
                c_val, c_unit = cand_qty
                if t_unit == c_unit and t_val > 0:
                    diff = abs(t_val - c_val) / t_val
                    if diff <= 0.1:
                        qty_bonus = 0.1
            unit_bonus = 0.0
            cand_unit = cand_qty[1] if cand_qty else None
            if target_unit and cand_unit and target_unit == cand_unit:
                unit_bonus = 0.1
            return base * 0.2 + token_score * 0.6 + qty_bonus + unit_bonus

        detail = invoice_line.detalle if hasattr(invoice_line, "detalle") else ""
        if not detail:
            return None
        target_norm = _normalize(detail)
        target_tokens = _tokens(detail)
        target_qty = _extract_quantity(detail)
        target_unit = _normalize_unit(getattr(invoice_line, "unidad", None))

        resolved_supplier_id = self._resolve_supplier_id(supplier_id, supplier_name)
        candidates_native = self._search_product_native_optimized(detail, supplier_id=resolved_supplier_id, limit=10)
        supplier_id_norm = self._normalize_id(resolved_supplier_id) if resolved_supplier_id is not None else None
        supplier_match_ids: set[int] = set()
        if supplier_id_norm is not None and candidates_native:
            seller_ids: set[int] = set()
            for prod in candidates_native:
                seller_ids.update(self._normalize_id_list(prod.get("seller_ids") or []))
            if seller_ids:
                try:
                    partner_field = self._supplierinfo_partner_field()
                    supplier_infos = self._execute_kw(
                        "product.supplierinfo",
                        "read",
                        [self._normalize_id_list(list(seller_ids))],
                        {"fields": ["id", partner_field]},
                    )
                    for info in supplier_infos:
                        partner_ref = info.get(partner_field)
                        partner_ref_id = self._normalize_id(partner_ref)
                        if partner_ref_id == supplier_id_norm:
                            info_id = self._normalize_id(info.get("id"))
                            if info_id is not None:
                                supplier_match_ids.add(info_id)
                except Exception as exc:
                    logger.warning("No se pudo validar seller_ids contra proveedor: %s", exc)

        best_product = None
        best_score = 0.0
        for prod in candidates_native:
            cand_name = prod.get("name") or ""
            cand_norm = _normalize(cand_name)
            cand_tokens = _tokens(cand_name)
            cand_qty = _extract_quantity(cand_name)
            score = _score_candidate(target_norm, cand_norm, target_tokens, cand_tokens, target_qty, cand_qty, target_unit)
            if score > best_score:
                best_score = score
                best_product = prod

        if best_product:
            seller_ids = self._normalize_id_list(best_product.get("seller_ids") or [])
            supplier_hit = bool(
                supplier_id_norm is not None
                and supplier_match_ids
                and any(sid in supplier_match_ids for sid in seller_ids)
            )
            if best_score > 0.90 or (best_score > 0.80 and supplier_hit):
                logger.info(
                    "Producto encontrado por Match Nativo (Score: %.2f): [%s] %s",
                    best_score,
                    best_product.get("id"),
                    best_product.get("name"),
                )
                return best_product.get("id")

        embedding_match = None
        try:
            embedding_match = self._find_product_candidate_by_embedding(
                invoice_line,
                supplier_id=resolved_supplier_id,
                supplier_name=supplier_name,
                return_score=True,
            )
        except Exception as exc:
            logger.warning("Búsqueda por embeddings falló: %s", exc)

        embedding_product = None
        if embedding_match:
            embedding_id, _score = embedding_match
            try:
                embedding_records = self._execute_kw(
                    "product.product",
                    "read",
                    [self._normalize_id_list([embedding_id])],
                    {"fields": ["id", "name", "default_code"]},
                )
                embedding_product = embedding_records[0] if embedding_records else None
            except Exception as exc:
                logger.warning("No se pudo leer el producto del embedding: %s", exc)
                embedding_product = None

        if embedding_product:
            logger.info(
                "Producto encontrado por Búsqueda Vectorial (Fallback): [%s] %s",
                embedding_product.get("id"),
                embedding_product.get("name"),
            )

        combined_candidates: List[dict] = list(candidates_native)
        if embedding_product:
            emb_id = self._normalize_id(embedding_product.get("id"))
            if emb_id is not None and not any(self._normalize_id(p.get("id")) == emb_id for p in combined_candidates):
                combined_candidates.append(embedding_product)

        best_final = None
        best_final_score = 0.0
        for prod in combined_candidates:
            cand_name = prod.get("name") or ""
            cand_norm = _normalize(cand_name)
            cand_tokens = _tokens(cand_name)
            cand_qty = _extract_quantity(cand_name)
            score = _score_candidate(target_norm, cand_norm, target_tokens, cand_tokens, target_qty, cand_qty, target_unit)
            if score > best_final_score:
                best_final_score = score
                best_final = prod

        if best_final and best_final_score > MIN_PRODUCT_CONFIDENCE:
            return best_final.get("id")
        return None

    def get_product_candidates(
        self,
        description: str,
        supplier_id: Optional[int] = None,
        supplier_name: Optional[str] = None,
        invoice_line: "InvoiceLine" | None = None,
        limit: int = MAX_PRODUCT_CANDIDATES,
    ) -> List[Dict[str, Any]]:
        """Retorna candidatos de producto con score usando embeddings y fuzzy search."""
        detail = (description or "").strip()
        if not detail:
            return []

        def _normalize(text: str) -> str:
            if not text:
                return ""
            text = text.lower()
            text = re.sub(r"\[[^\]]+\]", " ", text)
            text = "".join(
                ch
                for ch in unicodedata.normalize("NFD", text)
                if unicodedata.category(ch) != "Mn"
            )
            text = re.sub(r"[^a-z0-9\s]", " ", text)
            text = re.sub(r"\s+", " ", text).strip()
            return text

        def _tokens(text: str) -> List[str]:
            normalized = _normalize(text)
            if not normalized:
                return []
            words: List[str] = []
            for word in normalized.split():
                if not word or word in STOPWORDS:
                    continue
                words.append(SYNONYM_MAP.get(word, word))
            return words

        def _extract_quantity(text: str) -> Optional[Tuple[float, str]]:
            pattern = re.compile(r"(\d+(?:[\.,]\d+)?)\s*(kg|g|gr|gramos?|ml|l|lt|cc)", re.IGNORECASE)
            match = pattern.search(text or "")
            if not match:
                return None
            raw_val = match.group(1).replace(",", ".")
            try:
                val = float(raw_val)
            except ValueError:
                return None
            unit = match.group(2).lower()
            if unit in {"kg"}:
                return val * 1000.0, "g"
            if unit in {"l", "lt"}:
                return val * 1000.0, "ml"
            if unit == "cc":
                return val, "cc"
            return val, unit

        def _normalize_unit(value: Optional[str]) -> Optional[str]:
            if not value:
                return None
            unit = value.strip().lower()
            mapping = {
                "kg": {"kg", "kilo", "kilos"},
                "g": {"g", "gr", "gramo", "gramos"},
                "l": {"l", "lt", "litro", "litros"},
                "ml": {"ml", "mililitro", "mililitros"},
                "unidad": {"unidad", "unidades", "ud", "uds", "un"},
                "saco": {"saco", "sacos"},
                "caja": {"caja", "cajas"},
            }
            for norm, variants in mapping.items():
                if unit in variants:
                    return norm
            return unit

        def _score_candidate(
            target_norm: str,
            cand_norm: str,
            target_tokens: List[str],
            cand_tokens: List[str],
            target_qty,
            cand_qty,
            target_unit: Optional[str],
        ) -> float:
            base = SequenceMatcher(None, target_norm, cand_norm).ratio()
            if target_tokens:
                overlap = len(set(target_tokens) & set(cand_tokens))
                token_score = overlap / max(len(target_tokens), 1)
            else:
                token_score = 0.0
            qty_bonus = 0.0
            if target_qty and cand_qty:
                t_val, t_unit = target_qty
                c_val, c_unit = cand_qty
                if t_unit == c_unit and t_val > 0:
                    diff = abs(t_val - c_val) / t_val
                    if diff <= 0.1:
                        qty_bonus = 0.1
            unit_bonus = 0.0
            cand_unit = cand_qty[1] if cand_qty else None
            if target_unit and cand_unit and target_unit == cand_unit:
                unit_bonus = 0.1
            return base * 0.2 + token_score * 0.6 + qty_bonus + unit_bonus
        unidad = getattr(invoice_line, "unidad", None) if invoice_line is not None else None
        target_norm = _normalize(detail)
        target_tokens = _tokens(detail)
        target_qty = _extract_quantity(detail)
        target_unit = _normalize_unit(unidad)

        resolved_supplier_id = self._resolve_supplier_id(supplier_id, supplier_name)
        product_records = self._search_product_native_optimized(detail, supplier_id=resolved_supplier_id, limit=10)
        candidates: List[Dict[str, Any]] = []

        for prod in product_records or []:
            cand_name = prod.get("name") or ""
            cand_norm = _normalize(cand_name)
            cand_tokens = _tokens(cand_name)
            cand_qty = _extract_quantity(cand_name)
            score = _score_candidate(target_norm, cand_norm, target_tokens, cand_tokens, target_qty, cand_qty, target_unit)
            dc = prod.get("default_code")
            dc = dc if dc not in (False, None, "") else None
            candidates.append(
                {
                    "id": prod.get("id"),
                    "name": cand_name,
                    "score": float(score),
                    "default_code": dc,
                }
            )

        line_for_embedding = invoice_line or SimpleNamespace(detalle=detail, unidad=unidad)
        embedding_match = None
        try:
            embedding_match = self._find_product_candidate_by_embedding(
                line_for_embedding,
                supplier_id=resolved_supplier_id,
                supplier_name=supplier_name,
                return_score=True,
            )
        except Exception as exc:
            logger.warning("Búsqueda por embedding falló: %s", exc)

        if embedding_match:
            emb_id, _emb_score = embedding_match
            emb_norm = self._normalize_id(emb_id)
            if emb_norm is not None and not any(self._normalize_id(c.get("id")) == emb_norm for c in candidates):
                try:
                    emb_records = self._execute_kw(
                        "product.product",
                        "read",
                        [self._normalize_id_list([emb_norm])],
                        {"fields": ["id", "name", "default_code"]},
                    )
                    if emb_records:
                        emb_prod = emb_records[0]
                        cand_name = emb_prod.get("name") or ""
                        cand_norm = _normalize(cand_name)
                        cand_tokens = _tokens(cand_name)
                        cand_qty = _extract_quantity(cand_name)
                        score = _score_candidate(
                            target_norm,
                            cand_norm,
                            target_tokens,
                            cand_tokens,
                            target_qty,
                            cand_qty,
                            target_unit,
                        )
                        dc = emb_prod.get("default_code")
                        dc = dc if dc not in (False, None, "") else None
                        candidates.append(
                            {
                                "id": emb_prod.get("id"),
                                "name": cand_name,
                                "score": float(score),
                                "default_code": dc,
                            }
                        )
                except Exception as exc:
                    logger.warning("No se pudo leer producto embedding para candidatos: %s", exc)

        best_by_id: Dict[int, Dict[str, Any]] = {}
        for cand in candidates:
            pid = self._normalize_id(cand.get("id"))
            if pid is None:
                continue
            existing = best_by_id.get(pid)
            cand_score = float(cand.get("score") or 0.0)
            raw_default_code = cand.get("default_code")
            if raw_default_code in (False, None, ""):
                raw_default_code = None
            payload = {
                "id": pid,
                "name": cand.get("name"),
                "default_code": raw_default_code,
                "score": round(cand_score, 3),
            }
            if not existing or cand_score > existing.get("score", 0.0):
                best_by_id[pid] = payload

        sorted_candidates = sorted(best_by_id.values(), key=lambda x: x.get("score", 0.0), reverse=True)
        if limit and limit > 0:
            return sorted_candidates[:limit]
        return sorted_candidates

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
        product_name: Optional[str] = None,
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
        if product_name and "product_name" in fields:
            values["product_name"] = product_name

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
        if product_name:
            logger.info("Nombre de factura mapeado a supplierinfo: %s", product_name)

    def _product_id_from_supplierinfo_record(self, record: dict) -> Optional[int]:
        """Extrae product_id desde un registro de supplierinfo, resolviendo plantillas si aplica."""
        product_field = self._supplierinfo_product_field()
        product_ref = record.get(product_field)
        prod_id = self._normalize_id(product_ref)
        if prod_id is None:
            return None
        if product_field == "product_tmpl_id":
            try:
                prod_ids = self._execute_kw(
                    "product.product",
                    "search",
                    [[["product_tmpl_id", "=", prod_id]]],
                    {"limit": 1},
                )
                return self._normalize_id(prod_ids[0]) if prod_ids else None
            except Exception:
                return None
        return prod_id

    def get_mapped_product_id(self, invoice_detail: str, supplier_id: Optional[int], partial: bool = False) -> Optional[int]:
        """Devuelve product_id mapeado por supplierinfo.product_name e ID de proveedor."""
        if not invoice_detail or supplier_id is None:
            return None
        fields = self._get_supplierinfo_fields()
        if "product_name" not in fields:
            return None
        partner_field = self._supplierinfo_partner_field()
        op = "ilike" if partial else "="
        domain = [
            [partner_field, "=", self._normalize_id(supplier_id)],
            ["product_name", op, invoice_detail],
        ]
        try:
            records = self._execute_kw(
                "product.supplierinfo",
                "search_read",
                [domain],
                {"fields": ["id", partner_field, "product_name", self._supplierinfo_product_field()]},
            )
        except Exception as exc:
            logger.warning("No se pudo leer supplierinfo para mapeo '%s': %s", invoice_detail, exc)
            return None
        if not records:
            return None
        product_id = self._product_id_from_supplierinfo_record(records[0])
        if product_id:
            logger.info("Producto obtenido desde mapeo supplierinfo: %s -> product_id %s", invoice_detail, product_id)
        return product_id

    def ensure_product_for_supplier(
        self,
        invoice_line: "InvoiceLine",
        supplier_id: Optional[int],
        auto_create: bool = False,
        supplier_name: Optional[str] = None,
    ) -> Optional[int]:
        """Garantiza que exista un product_id compatible con el proveedor."""
        supplier_label = supplier_name or getattr(invoice_line, "supplier_name", None)
        supplier_id_norm = self._normalize_id(supplier_id) if supplier_id is not None else None
        force_hitl = str(os.getenv("FORCE_HITL", "")).strip().lower() in {"1", "true", "yes", "on"}

        def _product_has_supplier(pid: int) -> bool:
            if supplier_id_norm is None:
                return True
            try:
                recs = self._execute_kw(
                    "product.product",
                    "read",
                    [[pid]],
                    {"fields": ["seller_ids"]},
                )
                seller_ids = self._normalize_id_list(recs[0].get("seller_ids") or []) if recs else []
                if not seller_ids:
                    return False
                partner_field = self._supplierinfo_partner_field()
                supplier_infos = self._execute_kw(
                    "product.supplierinfo",
                    "read",
                    [self._normalize_id_list(seller_ids)],
                    {"fields": ["id", partner_field]},
                )
                for info in supplier_infos:
                    partner_ref = info.get(partner_field)
                    partner_ref_id = self._normalize_id(partner_ref)
                    if partner_ref_id == supplier_id_norm:
                        return True
            except Exception:
                return False
            return False

        def _accept_explicit_product(pid: int) -> Optional[int]:
            if supplier_id is not None and not _product_has_supplier(pid):
                self._ensure_supplier_on_product(pid, supplier_id, invoice_line.precio_unitario)
            if not force_hitl:
                return pid
            return None
        # 1) Mapeo directo por SKU si viene en la línea
        sku = getattr(invoice_line, "sku", None) or getattr(invoice_line, "default_code", None)
        if sku:
            pid_by_sku = self.get_product_id_by_sku(str(sku))
            if pid_by_sku is not None:
                accepted = _accept_explicit_product(pid_by_sku)
                if accepted is not None:
                    return accepted
                # Si FORCE_HITL está activo y no hay seller válido, seguir a candidatos.

        # 2) Mapeo directo por supplierinfo.product_name
        mapped_id = self.get_mapped_product_id(getattr(invoice_line, "detalle", None), supplier_id)
        if mapped_id is not None:
            accepted = _accept_explicit_product(mapped_id)
            if accepted is not None:
                return accepted
        # 2b) Coincidencia parcial en supplierinfo si hay proveedor
        mapped_partial = self.get_mapped_product_id(getattr(invoice_line, "detalle", None), supplier_id, partial=True)
        if mapped_partial is not None:
            accepted = _accept_explicit_product(mapped_partial)
            if accepted is not None:
                return accepted

        candidates = self.get_product_candidates(
            getattr(invoice_line, "detalle", None),
            supplier_id=supplier_id,
            supplier_name=supplier_label,
            invoice_line=invoice_line,
        )

        if candidates:
            best = candidates[0]
            best_score = float(best.get("score") or 0.0)
            product_id = self._normalize_id(best.get("id"))
            if product_id is not None and (
                best_score >= MIN_PRODUCT_CONFIDENCE
                or (auto_create and best_score >= MIN_PRODUCT_EMBEDDING_SCORE)
            ):
                if supplier_id is not None:
                    self._ensure_supplier_on_product(product_id, supplier_id, invoice_line.precio_unitario)
                return product_id

        if not auto_create:
            return None
        if supplier_id is None:
            logger.warning("No se puede crear producto sin supplier_id al forzar auto_create.")
            return None

        logger.info("Creando producto nuevo para %s", getattr(invoice_line, "detalle", "producto desconocido"))
        uom_id = self._get_default_uom_id()
        category_id = self._get_default_category_id()
        product_id = self._execute_kw(
            "product.product",
            "create",
            [[
                {
                    "name": getattr(invoice_line, "detalle", "Producto sin nombre"),
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


    def _resolve_product_by_default_code(self, default_code: str, supplier_id: Optional[int] = None) -> Optional[int]:
        """Busca product_id por default_code, opcionalmente filtrando por supplier_id."""
        raw_value = default_code
        default_code = self._sanitize_default_code(default_code)
        if not default_code and not raw_value:
            return None
        base_domain: list | None = [["default_code", "=", default_code]] if default_code else None
        barcode_domain: list | None = [["barcode", "=", default_code]] if default_code else None
        raw_name = str(raw_value).strip() if raw_value is not None else ""
        raw_name = raw_name.strip("`'\"").strip()
        name_domain: list | None = [["name", "ilike", raw_name]] if raw_name and any(ch.isspace() for ch in raw_name) else None

        def _search(domain: list | None) -> Optional[int]:
            if not domain:
                return None
            try:
                ids = self._execute_kw(
                    "product.product",
                    "search",
                    [domain],
                    {"limit": 1},
                )
                if ids:
                    logger.info("SKU resolve: product.product domain=%s -> %s", domain, ids)
                    return self._normalize_id(ids[0])
            except Exception as exc:
                logger.warning("No se pudo resolver product.product con dominio %s: %s", domain, exc)
            return None

        def _search_unique(domain: list | None) -> Optional[int]:
            if not domain:
                return None
            try:
                ids = self._execute_kw(
                    "product.product",
                    "search",
                    [domain],
                    {"limit": 2},
                )
                if len(ids) == 1:
                    logger.info("SKU resolve (unique): product.product domain=%s -> %s", domain, ids)
                    return self._normalize_id(ids[0])
                if len(ids) > 1:
                    logger.warning("Dominio %s devolvió múltiples productos; se omite.", domain)
            except Exception as exc:
                logger.warning("No se pudo resolver product.product con dominio %s: %s", domain, exc)
            return None

        def _search_template(domain: list | None) -> Optional[int]:
            if not domain:
                return None
            try:
                ids = self._execute_kw(
                    "product.template",
                    "search",
                    [domain],
                    {"limit": 1},
                )
                if ids:
                    logger.info("SKU resolve: product.template domain=%s -> %s", domain, ids)
                    return self._normalize_id(ids[0])
            except Exception as exc:
                logger.warning("No se pudo resolver product.template con dominio %s: %s", domain, exc)
            return None

        def _product_from_template(template_id: int) -> Optional[int]:
            try:
                ids = self._execute_kw(
                    "product.product",
                    "search",
                    [[["product_tmpl_id", "=", int(template_id)]]],
                    {"limit": 1},
                )
                if ids:
                    logger.info("SKU resolve: product.product from template=%s -> %s", template_id, ids)
                    return self._normalize_id(ids[0])
            except Exception as exc:
                logger.warning("No se pudo resolver product.product para template %s: %s", template_id, exc)
            return None

        # 1) Intento filtrado por proveedor si hay supplier_id (default_code / barcode).
        if supplier_id is not None:
            if base_domain:
                supplier_domain = ["&", ["seller_ids.partner_id", "=", int(supplier_id)], base_domain]
                found = _search(supplier_domain)
                if found is not None:
                    return found
            if barcode_domain:
                supplier_barcode = ["&", ["seller_ids.partner_id", "=", int(supplier_id)], barcode_domain]
                found = _search(supplier_barcode)
                if found is not None:
                    return found

        # 2) Fallback global (sin filtrar por proveedor).
        found = _search(base_domain)
        if found is not None:
            return found
        found = _search(barcode_domain)
        if found is not None:
            return found

        # 3) Fallback adicional: buscar en product.template (default_code / barcode) y resolver variante.
        if supplier_id is not None:
            if base_domain:
                template_domain = ["&", ["seller_ids.partner_id", "=", int(supplier_id)], base_domain]
                template_id = _search_template(template_domain)
                if template_id is not None:
                    product_id = _product_from_template(template_id)
                    if product_id is not None:
                        return product_id
            if barcode_domain:
                template_barcode = ["&", ["seller_ids.partner_id", "=", int(supplier_id)], barcode_domain]
                template_id = _search_template(template_barcode)
                if template_id is not None:
                    product_id = _product_from_template(template_id)
                    if product_id is not None:
                        return product_id

        template_id = _search_template(base_domain)
        if template_id is not None:
            product_id = _product_from_template(template_id)
            if product_id is not None:
                return product_id
        template_id = _search_template(barcode_domain)
        if template_id is not None:
            product_id = _product_from_template(template_id)
            if product_id is not None:
                return product_id

        # 4) Último fallback: nombre exacto/ilike cuando el input parece nombre (con espacios).
        if supplier_id is not None:
            supplier_name_domain = ["&", ["seller_ids.partner_id", "=", int(supplier_id)], name_domain] if name_domain else None
            found = _search_unique(supplier_name_domain)
            if found is not None:
                return found
        found = _search_unique(name_domain)
        if found is not None:
            return found
        template_id = _search_template(name_domain)
        if template_id is not None:
            product_id = _product_from_template(template_id)
            if product_id is not None:
                return product_id
        return None

    def map_product_decision(self, invoice_product_name: str, odoo_product_id: Optional[int], supplier_id: int, default_code: Optional[str] = None) -> None:
        """Registra la decisión humana asociando el detalle de factura a un product_id en supplierinfo."""
        if not invoice_product_name:
            raise ValueError("invoice_product_name es requerido para mapear el producto.")
        product_id = self._normalize_id(odoo_product_id) if odoo_product_id is not None else None
        supplier_id_norm = self._normalize_id(supplier_id)
        default_code = self._sanitize_default_code(default_code)
        if product_id is None and default_code:
            product_id = self._resolve_product_by_default_code(default_code, supplier_id_norm)
        if product_id is None or supplier_id_norm is None:
            raise ValueError("Se requiere product_id (o default_code resoluble) y supplier_id válidos.")

        fields = self._get_supplierinfo_fields()
        partner_field = self._supplierinfo_partner_field()
        product_field = self._supplierinfo_product_field()

        product_data = self._execute_kw(
            "product.product",
            "read",
            [[product_id]],
            {"fields": ["product_tmpl_id", "name"]},
        )
        template_id = None
        if product_data:
            template_id = self._normalize_id(product_data[0].get("product_tmpl_id"))
        if product_field == "product_tmpl_id":
            target_product_ref = template_id if template_id is not None else product_id
        else:
            target_product_ref = product_id
        domain = [
            [product_field, "=", target_product_ref],
            [partner_field, "=", supplier_id_norm],
        ]
        existing = self._execute_kw(
            "product.supplierinfo",
            "search_read",
            [domain],
            {"fields": ["id", "product_name"]},
        )
        if existing:
            updates: Dict[str, Any] = {}
            if "product_name" in fields and (existing[0].get("product_name") or "") != invoice_product_name:
                updates["product_name"] = invoice_product_name
            if updates:
                self._execute_kw(
                    "product.supplierinfo",
                    "write",
                    [[existing[0]["id"]], updates],
                )
            return

        self._create_supplierinfo_record(
            template_id,
            product_id,
            supplier_id_norm,
            price=0.0,
            product_name=invoice_product_name,
        )


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

    def _find_or_create_supplier(self, supplier_name: str, supplier_rut: Optional[str] = None) -> int:
        supplier_label = (supplier_name or "Proveedor Desconocido").strip()
        normalized_rut = self._normalize_vat_strict(supplier_rut)
        raw_rut = supplier_rut.strip() if isinstance(supplier_rut, str) else None
        conditions: List[Tuple[str, str, Any]] = []
        if supplier_label:
            conditions.append(("name", "ilike", supplier_label))
        if normalized_rut:
            conditions.append(("vat", "ilike", normalized_rut))
        if raw_rut and raw_rut != normalized_rut:
            conditions.append(("vat", "ilike", raw_rut))
        domain = self._build_or_domain(conditions)
        try:
            partners = self._execute_kw(
                "res.partner",
                "search_read",
                [domain],
                {"fields": ["id", "name", "vat", "supplier_rank"], "limit": 5},
            )
        except Exception as exc:
            logger.warning("No se pudo buscar proveedor en Odoo: %s", exc)
            partners = []

        best_partner: Optional[dict] = None
        best_score = 0.0
        for partner in partners or []:
            score = self._string_similarity(supplier_label, partner.get("name"))
            if score > best_score:
                best_score = score
                best_partner = partner

        if best_partner:
            logger.info(
                "Proveedor detectado por similitud: %s (ID %s, score %.2f)",
                best_partner.get("name"),
                best_partner.get("id"),
                best_score,
            )
            selected = self._normalize_id(best_partner.get("id"))
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
            return int(selected)
        logger.info(
            "No se encontró proveedor para '%s'. Se creará automáticamente uno nuevo.",
            supplier_label,
        )
        try:
            values = {"name": supplier_label, "company_type": "company", "supplier_rank": 1}
            if normalized_rut:
                values["vat"] = normalized_rut
            return self._execute_kw(
                "res.partner",
                "create",
                [[values]],
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
        def _create_order_via_xmlrpc(values: Dict[str, Any]) -> int:
            return self._execute_kw(
                "purchase.order",
                "create",
                [values],
            )

        def _create_lines_via_xmlrpc(order_id: int, lines_payload: List[dict]) -> None:
            for payload in lines_payload:
                payload["order_id"] = order_id
                self._execute_kw("purchase.order.line", "create", [[payload]])

        partner_id = self._find_or_create_supplier(invoice.supplier_name or "Proveedor Desconocido", getattr(invoice, "supplier_rut", None))
        today = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        line_values: List[dict] = []
        for line in invoice.lines:
            product_id = self.ensure_product_for_supplier(line, partner_id)
            product_id = self._normalize_id(product_id)
            if product_id is None:
                raise RuntimeError(f"No se pudo determinar product_id para la línea '{line.detalle}'.")
            product_data = self._get_product_details(product_id)
            taxes = self._get_line_taxes(product_data.get("supplier_taxes_id", []))
            uom = self._normalize_id(product_data.get("uom_po_id") or product_data.get("uom_id") or self._get_default_uom_id())
            if not taxes:
                logger.warning(
                    "La línea '%s' no tiene impuestos de compra; usa fallback vacío. Configura DEFAULT_PURCHASE_TAX_IDS si corresponde.",
                    line.detalle,
                )
            line_values.append(
                {
                    "name": line.detalle,
                    "product_id": product_id,
                    "product_qty": line.cantidad,
                    "price_unit": line.precio_unitario,
                    "product_uom": uom,
                    "taxes_id": taxes or [],
                    "date_planned": today,
                }
            )
        # Sanitiza payload de líneas: product_id/product_uom como int y taxes_id como lista de ints.
        sanitized_lines: List[dict] = []
        for lv in line_values:
            sanitized_lines.append(
                {
                    **lv,
                    "product_id": self._normalize_id(lv.get("product_id")),
                    "product_uom": self._normalize_id(lv.get("product_uom")),
                    "taxes_id": self._normalize_id_list(lv.get("taxes_id") or []),
                }
            )
        line_values = sanitized_lines
        logger.debug("Payload OdooSupply line_values=%s", line_values)

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
            logger.error("Error al crear la orden en Odoo usando OdooSupply: %s", exc, exc_info=True)
            # Fallback a XML-RPC directo para evitar errores de cliente (p.ej. unhashable list)
            try:
                order_vals = {
                    "partner_id": partner_id,
                    "date_order": today,
                    "origin": invoice.supplier_name,
                    "order_line": [],
                }
                order_id = _create_order_via_xmlrpc(order_vals)
                _create_lines_via_xmlrpc(order_id, line_values)
                logger.info("Orden creada vía XML-RPC fallback (ID %s) tras fallo en OdooSupply.", order_id)
                try:
                    order = self.confirm_purchase_order(order_id)
                    logger.info("Orden confirmada vía fallback (ID %s, estado %s).", order_id, order.get("state"))
                except Exception as confirm_exc:
                    logger.warning("No se pudo confirmar la orden creada por fallback (ID %s): %s", order_id, confirm_exc)
            except Exception as fallback_exc:
                logger.error("Fallback XML-RPC también falló: %s", fallback_exc, exc_info=True)
                raise RuntimeError(
                    f"Odoo rechazó la creación de la orden. Detalle: {exc}"
                ) from exc

        order = self._execute_kw(
            "purchase.order",
            "read",
            [[order_id]],
            {"fields": ["id", "name", "state", "amount_untaxed", "amount_tax", "amount_total", "order_line", "picking_ids", "partner_id"]},
        )[0]
        return order
    
    def confirm_order_receipt(self, order: dict, qty_by_product: Dict[int, float] | None = None) -> None:
        """Confirma la recepcion de pickings asociados a la orden, ruteando segun prefijo de SKU."""
        picking_ids = self._normalize_id_list(order.get("picking_ids", []))
        if not picking_ids:
            return
        dest_mp_me = self._get_location_id_by_name("JS/Stock/Materia Prima y Envases")
        dest_pt = self._get_location_id_by_name("JS/Stock")
        if dest_mp_me is None:
            raise RuntimeError(
                "No se encontro la ubicacion 'JS/Stock/Materia Prima y Envases' en Odoo."
            )
        if dest_pt is None:
            raise RuntimeError("No se encontro la ubicacion 'JS/Stock' en Odoo.")
        qty_by_product = qty_by_product or {}

        has_move_qty_done = False
        try:
            move_fields = self._execute_kw(
                "stock.move", "fields_get", [], {"attributes": ["type"]}
            )
            has_move_qty_done = "quantity_done" in move_fields
        except Exception:
            has_move_qty_done = False

        pickings = self._execute_kw(
            "stock.picking",
            "read",
            [picking_ids],
            {
                "fields": [
                    "id",
                    "state",
                    "move_ids_without_package",
                    "move_ids",
                    "location_id",
                    "location_dest_id",
                    "picking_type_id",
                    "partner_id",
                    "origin",
                    "company_id",
                    "move_type",
                    "purchase_id",
                ]
            },
        )
        active_pickings = [p for p in pickings if p.get("state") != "cancel"]
        if not active_pickings:
            return

        picking_moves: Dict[int, List[dict]] = {}
        product_ids: set[int] = set()

        for picking in active_pickings:
            try:
                self._execute_kw("stock.picking", "action_confirm", [[picking.get("id")]])
            except Exception:
                pass

            move_ids = self._normalize_id_list(
                picking.get("move_ids_without_package") or picking.get("move_ids") or []
            )
            if not move_ids:
                continue
            moves = self._execute_kw(
                "stock.move",
                "read",
                [move_ids],
                {
                    "fields": [
                        "id",
                        "product_id",
                        "location_id",
                        "location_dest_id",
                        "product_uom_qty",
                        "move_line_ids",
                    ]
                },
            )
            picking_id = self._normalize_id(picking.get("id"))
            if picking_id is not None:
                picking_moves[picking_id] = moves
            for move in moves or []:
                prod_id = self._normalize_id(move.get("product_id"))
                if prod_id is not None:
                    product_ids.add(prod_id)

        product_skus: Dict[int, Optional[str]] = {}
        if product_ids:
            products = self._execute_kw(
                "product.product",
                "read",
                [list(product_ids)],
                {"fields": ["id", "default_code"]},
            )
            for product in products or []:
                prod_id = self._normalize_id(product.get("id"))
                if prod_id is not None:
                    product_skus[prod_id] = product.get("default_code")

        pickings_to_validate: Dict[int, List[dict]] = {}

        for picking in active_pickings:
            picking_id = self._normalize_id(picking.get("id"))
            if picking_id is None:
                continue
            moves = picking_moves.get(picking_id, [])
            if not moves:
                continue

            moves_by_target: Dict[int, List[dict]] = {}
            for mv in moves:
                prod_id = self._normalize_id(mv.get("product_id"))
                sku_raw = product_skus.get(prod_id)
                sku_norm = self._sanitize_default_code(sku_raw) if sku_raw else None
                if not sku_norm:
                    logger.warning(
                        "Producto %s sin SKU; se usara destino JS/Stock.", prod_id
                    )
                    target_loc = dest_pt
                elif sku_norm.startswith(("MP", "ME")):
                    target_loc = dest_mp_me
                else:
                    target_loc = dest_pt
                moves_by_target.setdefault(target_loc, []).append(mv)

            if not moves_by_target:
                continue

            original_dest = self._normalize_id(picking.get("location_dest_id"))
            if original_dest in moves_by_target:
                base_loc = original_dest
            elif dest_pt in moves_by_target:
                base_loc = dest_pt
            else:
                base_loc = next(iter(moves_by_target.keys()))

            target_pickings: Dict[int, int] = {base_loc: picking_id}

            for target_loc, move_group in moves_by_target.items():
                if target_loc == base_loc:
                    continue
                payload = {
                    "picking_type_id": self._normalize_id(picking.get("picking_type_id")),
                    "location_id": self._normalize_id(picking.get("location_id")),
                    "location_dest_id": target_loc,
                    "origin": picking.get("origin"),
                    "partner_id": self._normalize_id(picking.get("partner_id")),
                    "company_id": self._normalize_id(picking.get("company_id")),
                    "move_type": picking.get("move_type"),
                    "purchase_id": self._normalize_id(picking.get("purchase_id")),
                }
                payload = {k: v for k, v in payload.items() if v not in (None, False, "")}
                new_picking_id = self._execute_kw(
                    "stock.picking",
                    "create",
                    [[payload]],
                )
                target_pickings[target_loc] = self._normalize_id(new_picking_id)

            if base_loc != original_dest:
                try:
                    self._execute_kw(
                        "stock.picking",
                        "write",
                        [[picking_id], {"location_dest_id": base_loc}],
                    )
                except Exception as exc:
                    logger.warning(
                        "No se pudo actualizar ubicacion destino del picking %s: %s",
                        picking_id,
                        exc,
                    )

            for target_loc, move_group in moves_by_target.items():
                target_picking_id = target_pickings.get(target_loc)
                if target_picking_id is None:
                    continue
                pickings_to_validate.setdefault(target_picking_id, []).extend(move_group)

                for mv in move_group:
                    mv_id = mv.get("id")
                    prod_id = self._normalize_id(mv.get("product_id"))
                    if mv_id:
                        try:
                            self._execute_kw(
                                "stock.move",
                                "write",
                                [[mv_id], {"location_dest_id": target_loc}],
                            )
                        except Exception as exc:
                            logger.warning(
                                "No se pudo actualizar destino de recepción para move %s: %s",
                                mv_id,
                                exc,
                            )
                        if target_picking_id != picking_id:
                            try:
                                self._execute_kw(
                                    "stock.move",
                                    "write",
                                    [[mv_id], {"picking_id": target_picking_id}],
                                )
                            except Exception as exc:
                                logger.warning(
                                    "No se pudo mover el move %s al picking %s: %s",
                                    mv_id,
                                    target_picking_id,
                                    exc,
                                )

                    line_ids_mv = self._normalize_id_list(mv.get("move_line_ids", []))
                    if line_ids_mv:
                        try:
                            self._execute_kw(
                                "stock.move.line",
                                "write",
                                [line_ids_mv, {"location_dest_id": target_loc}],
                            )
                        except Exception as exc:
                            logger.warning(
                                "No se pudo actualizar destino en move lines %s: %s",
                                line_ids_mv,
                                exc,
                            )

                    qty_planned = mv.get("product_uom_qty") or 0.0
                    qty_invoice = qty_by_product.get(prod_id) if prod_id is not None else None
                    qty_to_set = qty_invoice if qty_invoice is not None else qty_planned

                    if has_move_qty_done and mv_id and qty_to_set:
                        try:
                            self._execute_kw(
                                "stock.move",
                                "write",
                                [[mv_id], {"quantity_done": qty_to_set}],
                            )
                        except Exception as exc:
                            logger.warning(
                                "No se pudo setear quantity_done para move %s: %s",
                                mv_id,
                                exc,
                            )

                    if line_ids_mv and qty_to_set:
                        try:
                            self._execute_kw(
                                "stock.move.line",
                                "write",
                                [line_ids_mv, {"qty_done": qty_to_set}],
                            )
                        except Exception as exc:
                            logger.warning(
                                "No se pudo setear qty_done para move lines %s: %s",
                                line_ids_mv,
                                exc,
                            )
                    elif (not line_ids_mv) and qty_to_set and mv_id:
                        try:
                            src_loc = self._normalize_id(mv.get("location_id"))
                            if src_loc is None:
                                raise RuntimeError(
                                    f"No se pudo resolver la ubicacion origen para move {mv_id}."
                                )
                            self._execute_kw(
                                "stock.move.line",
                                "create",
                                [[
                                    {
                                        "move_id": mv_id,
                                        "product_id": prod_id,
                                        "qty_done": qty_to_set,
                                        "location_id": src_loc,
                                        "location_dest_id": target_loc,
                                    }
                                ]],
                            )
                        except Exception as exc:
                            logger.warning(
                                "No se pudo crear move line para move %s: %s", mv_id, exc
                            )

        for picking_id, moves in pickings_to_validate.items():
            if not moves:
                continue
            try:
                self._execute_kw("stock.picking", "action_confirm", [[picking_id]])
            except Exception:
                pass
            try:
                self._execute_kw("stock.picking", "button_validate", [[picking_id]])
            except Exception as exc:
                logger.warning(
                    "No se pudo validar picking %s: %s", picking_id, exc
                )
    
    def create_invoice_for_order(self, order_id: int) -> None:
        """Genera la factura desde la orden de compra."""
        self._execute_kw("purchase.order", "action_create_invoice", [[order_id]])


odoo_manager = OdooConnectionManager()
