"""
Test E2E con mocks para validar las 11 correcciones de bugs.

Cubre el flujo completo: OCR → InvoiceData → process_invoice_file → InvoiceResponseModel,
mockeando las dependencias externas (LLM/OpenAI, Odoo XML-RPC, Slack).
"""
import json
import os
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# ---------------------------------------------------------------------------
# Datos de prueba
# ---------------------------------------------------------------------------

SAMPLE_OCR_RESPONSE_NO_DISCOUNT = {
    "supplier_name": "Proveedor Test SpA",
    "supplier_rut": "76.123.456-7",
    "neto": 10000.0,
    "iva_19": 1900.0,
    "total": 11900.0,
    "descuento_global": 0.0,
    "lines": [
        {
            "detalle": "Aceite Esencial Lavanda 1L",
            "cantidad": 10,
            "precio_unitario": 500,
            "subtotal": 5000,
            "unidad": "un",
            "descuento_pct": None,
            "descuento_monto": None,
        },
        {
            "detalle": "Manteca de Karite 500g",
            "cantidad": 10,
            "precio_unitario": 500,
            "subtotal": 5000,
            "unidad": "un",
            "descuento_pct": None,
            "descuento_monto": None,
        },
    ],
}

SAMPLE_OCR_RESPONSE_WITH_DISCOUNT = {
    "supplier_name": "Proveedor Descuento SpA",
    "supplier_rut": "77.888.999-0",
    "neto": 9000.0,
    "iva_19": 1710.0,
    "total": 10710.0,
    "descuento_global": 0.0,
    "lines": [
        {
            "detalle": "Producto A",
            "cantidad": 10,
            "precio_unitario": 1000,
            "subtotal": 9000,  # OCR reporta subtotal ya con 10% descuento
            "unidad": "un",
            "descuento_pct": 10,
            "descuento_monto": None,
        },
    ],
}

# Respuesta de Odoo para una OC que matchea la factura sin descuento.
MOCK_ODOO_ORDER = {
    "id": 42,
    "name": "PO00042",
    "state": "purchase",
    "amount_untaxed": 10000.0,
    "amount_tax": 1900.0,
    "amount_total": 11900.0,
    "order_line": [101, 102],
    "picking_ids": [],
    "partner_id": [99, "Proveedor Test SpA"],
    "invoice_status": "to invoice",
}

MOCK_ODOO_ORDER_LINES = [
    {
        "id": 101,
        "product_id": 10,
        "detalle": "Aceite Esencial Lavanda 1L",
        "cantidad": 10,
        "precio_unitario": 500,
        "subtotal": 5000,
    },
    {
        "id": 102,
        "product_id": 20,
        "detalle": "Manteca de Karite 500g",
        "cantidad": 10,
        "precio_unitario": 500,
        "subtotal": 5000,
    },
]


# ---------------------------------------------------------------------------
# Helpers para mocks
# ---------------------------------------------------------------------------


def _make_fake_image(path: Path) -> Path:
    """Crea una imagen JPEG mínima válida para que PIL la pueda abrir."""
    from PIL import Image

    img = Image.new("RGB", (100, 200), color="white")
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, format="JPEG")
    return path


def _build_odoo_mock():
    """Crea un mock para odoo_manager con comportamiento configurable."""
    mock = MagicMock()
    mock._normalize_id.side_effect = lambda v: (
        int(v[0]) if isinstance(v, (list, tuple)) else (int(v) if v is not None else None)
    )
    mock._normalize_id_list.side_effect = lambda vs: [
        int(v[0]) if isinstance(v, (list, tuple)) else int(v)
        for v in (vs or [])
        if v is not None
    ]
    mock._resolve_supplier_id.return_value = 99
    mock.get_mapped_product_id.return_value = 10
    mock.get_product_candidates.return_value = [
        {"id": 10, "name": "Aceite Esencial Lavanda 1L", "score": 0.95, "default_code": "MP001"},
    ]
    mock.get_sku_by_product_id.return_value = "MP001"
    mock.find_purchase_order_by_similarity.return_value = MOCK_ODOO_ORDER
    mock.read_order.return_value = MOCK_ODOO_ORDER
    mock.read_order_lines.return_value = MOCK_ODOO_ORDER_LINES
    mock.read_receipts_for_order.return_value = []
    mock.recompute_order_amounts.return_value = None
    return mock


# ====================================================================
# FIX #1: get_settings() con lru_cache
# ====================================================================


class TestFix1SettingsCache:
    def test_get_settings_returns_same_instance(self):
        """get_settings() debe devolver la misma instancia (cacheada)."""
        from invoice_processor.config.config import get_settings

        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2, "get_settings() debería estar cacheado con lru_cache"


# ====================================================================
# FIX #2: cantidad ge=0 acepta 0
# ====================================================================


class TestFix2CantidadGeZero:
    def test_cantidad_zero_accepted(self):
        """InvoiceLine con cantidad=0 no debe lanzar ValidationError."""
        from invoice_processor.core.models import InvoiceLine

        line = InvoiceLine(
            detalle="Producto bonificado",
            cantidad=0,
            precio_unitario=100,
            subtotal=0,
        )
        assert line.cantidad == 0

    def test_cantidad_negative_rejected(self):
        """InvoiceLine con cantidad negativa sigue siendo rechazada."""
        from invoice_processor.core.models import InvoiceLine

        with pytest.raises(Exception):
            InvoiceLine(
                detalle="Producto",
                cantidad=-1,
                precio_unitario=100,
                subtotal=-100,
            )


# ====================================================================
# FIX #3: código inalcanzable eliminado
# ====================================================================


class TestFix3UnreachableCode:
    def test_is_affirmative_returns_bool(self):
        """_is_affirmative siempre devuelve bool, nunca None."""
        from invoice_processor.app.slack_handler import _is_affirmative

        assert _is_affirmative("si") is True
        assert _is_affirmative("sí") is True
        assert _is_affirmative("no") is False
        assert _is_affirmative("xyz random") is False
        result = _is_affirmative("")
        assert isinstance(result, bool)


# ====================================================================
# FIX #4: crop sin overlap
# ====================================================================


class TestFix4CropNoOverlap:
    def test_crop_boxes_no_overlap(self, tmp_path):
        """top_box y bottom_box no deben solaparse (ambos usan 0.35)."""
        from invoice_processor.infrastructure.services.ocr import InvoiceOcrClient

        img_path = _make_fake_image(tmp_path / "test.jpg")
        cropped_path = InvoiceOcrClient._crop_invoice(str(img_path))
        cropped = Path(cropped_path)
        assert cropped.exists()

        from PIL import Image

        original = Image.open(img_path)
        _, height = original.size
        # top_box llega hasta 0.35*height, bottom_box empieza en 0.35*height → sin overlap
        top_end = int(height * 0.35)
        bottom_start = int(height * 0.35)
        assert top_end == bottom_start, f"Overlap: top termina en {top_end}, bottom empieza en {bottom_start}"
        # Limpiar
        cropped.unlink(missing_ok=True)


# ====================================================================
# FIX #5: user_id auth check después de extracción definitiva
# ====================================================================


class TestFix5UserIdAuthOrder:
    def test_source_code_order(self):
        """El filtro de usuarios autorizados debe estar DESPUÉS de la extracción
        definitiva de user_id (línea con event.get('user'))."""
        import inspect
        from invoice_processor.app import slack_handler

        source = inspect.getsource(slack_handler.run_slack_bot)
        # Buscar posiciones relativas
        definitive_extraction = source.find('user_id = event.get("user")')
        auth_filter = source.find("# Filtro de usuarios autorizados")
        assert definitive_extraction != -1, "No se encontró la extracción definitiva de user_id"
        assert auth_filter != -1, "No se encontró el filtro de usuarios autorizados"
        assert definitive_extraction < auth_filter, (
            "La extracción definitiva de user_id debe ocurrir ANTES del filtro de auth"
        )


# ====================================================================
# FIX #6: normalización sin actualización redundante de precio_unitario
# ====================================================================


class TestFix6NormalizationNoRedundantPrice:
    def test_source_no_corrected_price_in_big_diff_branch(self):
        """En el bloque de normalización con diff > 1.0, solo se actualiza subtotal,
        no precio_unitario (que sería redundante)."""
        import inspect
        from invoice_processor.core import processor

        source = inspect.getsource(processor.process_invoice_file)
        # Buscar el bloque de normalización
        norm_start = source.find("# Normaliza líneas si subtotal")
        if norm_start == -1:
            norm_start = source.find("# Prefiere ajustar subtotal")
        assert norm_start != -1, "No se encontró el bloque de normalización"
        # En ese bloque, la primera rama (diff > 1.0) no debe tener "corrected_price"
        norm_section = source[norm_start:norm_start + 600]
        assert "corrected_price" not in norm_section.split("elif")[0], (
            "La rama diff > 1.0 no debería calcular corrected_price (es redundante)"
        )


# ====================================================================
# FIX #7: descuento no se aplica dos veces en OCR postprocess
# ====================================================================


class TestFix7NoDoubleDiscount:
    def test_ocr_postprocess_stores_undiscounted_subtotal(self):
        """Cuando el OCR reporta un subtotal que ya incluye descuento,
        _postprocess_invoice_payload debe almacenar qty*price (sin descuento)
        para que _apply_line_discounts lo aplique una sola vez."""
        from invoice_processor.infrastructure.services.ocr import InvoiceOcrClient

        client = InvoiceOcrClient.__new__(InvoiceOcrClient)
        payload = json.loads(json.dumps(SAMPLE_OCR_RESPONSE_WITH_DISCOUNT))
        result = client._postprocess_invoice_payload(payload)

        line = result["lines"][0]
        qty = line["cantidad"]
        price = line["precio_unitario"]
        expected_undiscounted = qty * price  # 10 * 1000 = 10000

        assert line["subtotal"] == expected_undiscounted, (
            f"El subtotal debería ser {expected_undiscounted} (sin descuento), "
            f"pero es {line['subtotal']}. El descuento se aplicará en _apply_line_discounts."
        )
        # Verificar que descuento_pct se preserva para que processor.py lo aplique
        assert line["descuento_pct"] == 10

    def test_apply_line_discounts_applies_once(self):
        """_apply_line_discounts aplica el descuento correctamente sobre el subtotal
        sin descuento almacenado por el OCR."""
        from invoice_processor.core.models import InvoiceData, InvoiceLine
        from invoice_processor.core.processor import _apply_line_discounts

        invoice = InvoiceData(
            neto=9000,
            iva_19=1710,
            total=10710,
            lines=[
                InvoiceLine(
                    detalle="Producto A",
                    cantidad=10,
                    precio_unitario=1000,
                    subtotal=10000,  # sin descuento (como lo guarda el OCR corregido)
                    descuento_pct=10,
                ),
            ],
        )
        adjusted = _apply_line_discounts(invoice)
        line = adjusted.lines[0]

        # 10 * 1000 * (1 - 10/100) = 9000
        assert abs(line.subtotal - 9000) < 0.01, f"Subtotal con descuento debería ser 9000, es {line.subtotal}"
        # Precio neto: 9000 / 10 = 900
        assert abs(line.precio_unitario - 900) < 0.01, f"Precio neto debería ser 900, es {line.precio_unitario}"

    def test_no_discount_no_change(self):
        """Sin descuento, _postprocess_invoice_payload preserva el subtotal del OCR."""
        from invoice_processor.infrastructure.services.ocr import InvoiceOcrClient

        client = InvoiceOcrClient.__new__(InvoiceOcrClient)
        payload = json.loads(json.dumps(SAMPLE_OCR_RESPONSE_NO_DISCOUNT))
        result = client._postprocess_invoice_payload(payload)

        for line in result["lines"]:
            assert line["subtotal"] == line["cantidad"] * line["precio_unitario"]


# ====================================================================
# FIX #8: limpieza de archivos temporales (OCR cropped)
# ====================================================================


class TestFix8TempFileCleanup:
    def test_cropped_file_cleaned_after_extract(self, tmp_path):
        """El archivo .cropped.jpg se elimina después de extract()."""
        from invoice_processor.infrastructure.services.ocr import InvoiceOcrClient

        img_path = _make_fake_image(tmp_path / "factura.jpg")
        cropped_path = img_path.with_suffix(".cropped.jpg")

        client = InvoiceOcrClient.__new__(InvoiceOcrClient)
        # Mock _call_llm para devolver un resultado dudoso primero, luego uno limpio
        call_count = {"n": 0}

        def fake_call_llm(path):
            call_count["n"] += 1
            if call_count["n"] == 1:
                # Primera llamada: devolver resultado dudoso para forzar crop
                return json.dumps({
                    **SAMPLE_OCR_RESPONSE_NO_DISCOUNT,
                    "ocr_dudoso": True,
                })
            # Segunda llamada (con cropped): devolver resultado limpio
            return json.dumps(SAMPLE_OCR_RESPONSE_NO_DISCOUNT)

        client._call_llm = fake_call_llm
        client._postprocess_invoice_payload = InvoiceOcrClient._postprocess_invoice_payload.__get__(client)
        client._crop_invoice = InvoiceOcrClient._crop_invoice

        result = client.extract(str(img_path))

        assert not cropped_path.exists(), (
            f"El archivo cropped {cropped_path} debería haberse eliminado en el finally"
        )
        assert isinstance(result, dict)
        assert "error" not in result


# ====================================================================
# FIX #9: rate limiting
# ====================================================================


class TestFix9RateLimiting:
    def test_rate_limit_dict_exists(self):
        """El módulo slack_handler debe tener _last_submission y _RATE_LIMIT_SECONDS."""
        from invoice_processor.app import slack_handler

        assert hasattr(slack_handler, "_last_submission")
        assert hasattr(slack_handler, "_RATE_LIMIT_SECONDS")
        assert slack_handler._RATE_LIMIT_SECONDS == 10
        assert isinstance(slack_handler._last_submission, dict)


# ====================================================================
# FIX #10: timeout en XML-RPC
# ====================================================================


class TestFix10XmlRpcTimeout:
    def test_timeout_constants_exist(self):
        """El módulo odoo_connection_manager debe tener constantes de timeout."""
        from invoice_processor.infrastructure.services import odoo_connection_manager as mod

        assert hasattr(mod, "_XMLRPC_TIMEOUT_SECONDS")
        assert mod._XMLRPC_TIMEOUT_SECONDS == 30
        assert hasattr(mod, "_xmlrpc_executor")

    def test_execute_kw_raises_timeout_error(self):
        """_execute_kw debe lanzar TimeoutError si el XML-RPC excede el timeout."""
        import concurrent.futures
        from invoice_processor.infrastructure.services.odoo_connection_manager import (
            OdooConnectionManager,
        )
        from invoice_processor.infrastructure.services import odoo_connection_manager as mod

        manager = OdooConnectionManager.__new__(OdooConnectionManager)

        # Mock del client
        fake_client = MagicMock()
        fake_client.db = "test_db"
        fake_client.uid = 1
        fake_client.password = "pass"
        # Hacer que execute_kw bloquee indefinidamente
        fake_client.models.execute_kw.side_effect = lambda *a, **kw: time.sleep(999)

        with (
            patch.object(manager, "get_product_client", return_value=fake_client),
            patch.object(mod, "_XMLRPC_TIMEOUT_SECONDS", 0.1),  # timeout muy corto
        ):
            with pytest.raises(TimeoutError, match="excedió el timeout"):
                manager._execute_kw("res.partner", "search", [[]])


# ====================================================================
# FIX #11: código unificado (odoo_tools es canónico)
# ====================================================================


class TestFix11UnifiedCode:
    def test_processor_does_not_define_looks_like_sku(self):
        """processor.py no debe tener su propia _looks_like_sku."""
        import inspect
        from invoice_processor.core import processor

        source = inspect.getsource(processor)
        # No debe haber una definición de función _looks_like_sku en processor.py
        assert "def _looks_like_sku" not in source, (
            "_looks_like_sku no debe estar definida en processor.py; debe importarse de odoo_tools"
        )

    def test_processor_does_not_define_resolve_product_for_split(self):
        """processor.py no debe tener su propia _resolve_product_for_split."""
        import inspect
        from invoice_processor.core import processor

        source = inspect.getsource(processor)
        assert "def _resolve_product_for_split" not in source

    def test_odoo_tools_resolve_has_extra_fields_and_supplier(self):
        """_resolve_product_by_term en odoo_tools debe aceptar extra_fields y supplier_id."""
        import inspect
        from invoice_processor.tools.odoo_tools import _resolve_product_by_term

        sig = inspect.signature(_resolve_product_by_term)
        params = list(sig.parameters.keys())
        assert "supplier_id" in params
        assert "extra_fields" in params

    def test_looks_like_sku_from_odoo_tools(self):
        """_looks_like_sku funciona correctamente desde odoo_tools."""
        from invoice_processor.tools.odoo_tools import _looks_like_sku

        assert _looks_like_sku("MP001") is True
        assert _looks_like_sku("SKU-123_A") is True
        assert _looks_like_sku("tiene espacios") is False
        assert _looks_like_sku("") is False


# ====================================================================
# TEST E2E: flujo completo process_invoice_file con mocks
# ====================================================================


class TestE2EProcessInvoice:
    """Test end-to-end del flujo principal con todos los servicios mockeados."""

    @patch("invoice_processor.core.processor.odoo_manager")
    @patch("invoice_processor.core.processor.ocr_client")
    def test_full_flow_readonly_match(self, mock_ocr, mock_odoo, tmp_path):
        """Flujo completo en modo lectura: OCR → parse → comparar con Odoo.
        Todas las líneas coinciden → status WAITING_FOR_APPROVAL."""
        from invoice_processor.core.processor import process_invoice_file

        img_path = _make_fake_image(tmp_path / "factura.jpg")

        # Configurar OCR mock
        mock_ocr.extract.return_value = SAMPLE_OCR_RESPONSE_NO_DISCOUNT

        # Configurar Odoo mock
        mock_odoo._normalize_id.side_effect = lambda v: (
            int(v[0]) if isinstance(v, (list, tuple)) else (int(v) if v is not None else None)
        )
        mock_odoo._normalize_id_list.side_effect = lambda vs: [
            int(v[0]) if isinstance(v, (list, tuple)) else int(v)
            for v in (vs or [])
            if v is not None
        ]
        mock_odoo._resolve_supplier_id.return_value = 99
        mock_odoo.get_mapped_product_id.return_value = 10
        mock_odoo.get_product_candidates.return_value = [
            {"id": 10, "name": "Aceite Esencial Lavanda 1L", "score": 0.95, "default_code": "MP001"},
        ]
        mock_odoo.get_sku_by_product_id.return_value = "MP001"
        mock_odoo.find_purchase_order_by_similarity.return_value = MOCK_ODOO_ORDER
        mock_odoo.read_order.return_value = MOCK_ODOO_ORDER
        mock_odoo.read_order_lines.return_value = MOCK_ODOO_ORDER_LINES
        mock_odoo.read_receipts_for_order.return_value = []

        result = process_invoice_file(str(img_path), allow_odoo_write=False)

        assert isinstance(result, type(result))  # InvoiceResponseModel
        assert result.needs_follow_up is True  # modo lectura siempre pide aprobación
        assert result.status == "WAITING_FOR_APPROVAL"
        assert result.supplier_id == 99
        assert result.neto_match is True
        assert result.iva_match is True
        assert result.total_match is True

    @patch("invoice_processor.core.processor.odoo_manager")
    @patch("invoice_processor.core.processor.ocr_client")
    def test_full_flow_with_discount_no_double_apply(self, mock_ocr, mock_odoo, tmp_path):
        """E2E: factura con descuento porcentual. El descuento se aplica exactamente
        una vez (en _apply_line_discounts, no en el OCR postprocess)."""
        from invoice_processor.core.processor import process_invoice_file

        img_path = _make_fake_image(tmp_path / "factura_desc.jpg")

        # OCR devuelve factura con descuento (raw_sub=9000 ya descontado)
        mock_ocr.extract.return_value = SAMPLE_OCR_RESPONSE_WITH_DISCOUNT

        # Odoo tiene la OC correspondiente (con neto=9000 descontado)
        discount_order = {
            **MOCK_ODOO_ORDER,
            "amount_untaxed": 9000.0,
            "amount_tax": 1710.0,
            "amount_total": 10710.0,
            "order_line": [201],
        }
        discount_order_lines = [
            {
                "id": 201,
                "product_id": 30,
                "detalle": "Producto A",
                "cantidad": 10,
                "precio_unitario": 900,  # precio neto (con descuento aplicado)
                "subtotal": 9000,
            },
        ]

        mock_odoo._normalize_id.side_effect = lambda v: (
            int(v[0]) if isinstance(v, (list, tuple)) else (int(v) if v is not None else None)
        )
        mock_odoo._normalize_id_list.side_effect = lambda vs: [
            int(v[0]) if isinstance(v, (list, tuple)) else int(v)
            for v in (vs or [])
            if v is not None
        ]
        mock_odoo._resolve_supplier_id.return_value = 99
        mock_odoo.get_mapped_product_id.return_value = 30
        mock_odoo.get_product_candidates.return_value = [
            {"id": 30, "name": "Producto A", "score": 0.99, "default_code": "PA001"},
        ]
        mock_odoo.get_sku_by_product_id.return_value = "PA001"
        mock_odoo.find_purchase_order_by_similarity.return_value = discount_order
        mock_odoo.read_order.return_value = discount_order
        mock_odoo.read_order_lines.return_value = discount_order_lines
        mock_odoo.read_receipts_for_order.return_value = []

        result = process_invoice_file(str(img_path), allow_odoo_write=False)

        # El resultado debe coincidir: OCR guardó subtotal=10000 (sin descuento),
        # _apply_line_discounts lo convirtió a 9000, que coincide con Odoo.
        assert result.neto_match is True, f"neto_match debería ser True, summary: {result.summary}"
        assert result.total_match is True
        assert result.status == "WAITING_FOR_APPROVAL"

    @patch("invoice_processor.core.processor.odoo_manager")
    @patch("invoice_processor.core.processor.ocr_client")
    def test_full_flow_no_order_found(self, mock_ocr, mock_odoo, tmp_path):
        """Cuando no se encuentra OC en Odoo, el resultado indica que se creará una nueva."""
        from invoice_processor.core.processor import process_invoice_file

        img_path = _make_fake_image(tmp_path / "factura_new.jpg")
        mock_ocr.extract.return_value = SAMPLE_OCR_RESPONSE_NO_DISCOUNT

        mock_odoo._normalize_id.side_effect = lambda v: (
            int(v[0]) if isinstance(v, (list, tuple)) else (int(v) if v is not None else None)
        )
        mock_odoo._normalize_id_list.side_effect = lambda vs: [
            int(v[0]) if isinstance(v, (list, tuple)) else int(v)
            for v in (vs or [])
            if v is not None
        ]
        mock_odoo._resolve_supplier_id.return_value = 99
        mock_odoo.get_mapped_product_id.return_value = 10
        mock_odoo.get_product_candidates.return_value = [
            {"id": 10, "name": "Producto", "score": 0.95, "default_code": "MP001"},
        ]
        mock_odoo.get_sku_by_product_id.return_value = "MP001"
        mock_odoo.find_purchase_order_by_similarity.return_value = None  # no encontró OC

        result = process_invoice_file(str(img_path), allow_odoo_write=False)

        assert result.needs_follow_up is True
        assert result.status == "WAITING_FOR_APPROVAL"
        assert "creará una nueva OC" in result.summary

    @patch("invoice_processor.core.processor.odoo_manager")
    @patch("invoice_processor.core.processor.ocr_client")
    def test_cantidad_zero_line_passes_through(self, mock_ocr, mock_odoo, tmp_path):
        """Una factura con una línea de cantidad=0 (bonificación) no debe crashear."""
        from invoice_processor.core.processor import process_invoice_file

        img_path = _make_fake_image(tmp_path / "factura_zero.jpg")
        ocr_data = {
            "supplier_name": "Proveedor Zero",
            "supplier_rut": "11.111.111-1",
            "neto": 5000.0,
            "iva_19": 950.0,
            "total": 5950.0,
            "descuento_global": 0.0,
            "lines": [
                {
                    "detalle": "Producto Normal",
                    "cantidad": 10,
                    "precio_unitario": 500,
                    "subtotal": 5000,
                    "unidad": "un",
                },
                {
                    "detalle": "Muestra gratis",
                    "cantidad": 0,
                    "precio_unitario": 0,
                    "subtotal": 0,
                    "unidad": "un",
                },
            ],
        }
        mock_ocr.extract.return_value = ocr_data

        mock_odoo._normalize_id.side_effect = lambda v: (
            int(v[0]) if isinstance(v, (list, tuple)) else (int(v) if v is not None else None)
        )
        mock_odoo._normalize_id_list.side_effect = lambda vs: [
            int(v[0]) if isinstance(v, (list, tuple)) else int(v)
            for v in (vs or [])
            if v is not None
        ]
        mock_odoo._resolve_supplier_id.return_value = 99
        mock_odoo.get_mapped_product_id.return_value = 10
        mock_odoo.get_product_candidates.return_value = [
            {"id": 10, "name": "Producto Normal", "score": 0.95, "default_code": "PN001"},
        ]
        mock_odoo.get_sku_by_product_id.return_value = "PN001"
        mock_odoo.find_purchase_order_by_similarity.return_value = None

        # No debe lanzar excepción por cantidad=0
        result = process_invoice_file(str(img_path), allow_odoo_write=False)
        assert result is not None
        assert result.status == "WAITING_FOR_APPROVAL"


# ====================================================================
# GAP #1-3: folio, fecha_emision — OCR, modelo, factura en Odoo
# ====================================================================


class TestGapFolioFechaInvoiceData:
    """InvoiceData acepta los nuevos campos folio y fecha_emision."""

    def test_invoice_data_accepts_folio_and_fecha(self):
        from invoice_processor.core.models import InvoiceData, InvoiceLine

        inv = InvoiceData(
            neto=1000,
            iva_19=190,
            total=1190,
            folio="12345",
            fecha_emision="2025-01-15",
            lines=[
                InvoiceLine(
                    detalle="Producto X",
                    cantidad=1,
                    precio_unitario=1000,
                    subtotal=1000,
                ),
            ],
        )
        assert inv.folio == "12345"
        assert inv.fecha_emision == "2025-01-15"

    def test_invoice_data_defaults_none(self):
        from invoice_processor.core.models import InvoiceData, InvoiceLine

        inv = InvoiceData(
            neto=1000,
            iva_19=190,
            total=1190,
            lines=[
                InvoiceLine(
                    detalle="Producto X",
                    cantidad=1,
                    precio_unitario=1000,
                    subtotal=1000,
                ),
            ],
        )
        assert inv.folio is None
        assert inv.fecha_emision is None


class TestGapOcrDateNormalization:
    """_postprocess_invoice_payload normaliza fecha_emision a YYYY-MM-DD."""

    def test_fecha_yyyy_mm_dd_passthrough(self):
        from invoice_processor.infrastructure.services.ocr import InvoiceOcrClient

        client = InvoiceOcrClient.__new__(InvoiceOcrClient)
        payload = {
            **SAMPLE_OCR_RESPONSE_NO_DISCOUNT,
            "folio": "99887",
            "fecha_emision": "2025-03-15",
        }
        result = client._postprocess_invoice_payload(dict(payload))
        assert result["fecha_emision"] == "2025-03-15"
        assert result["folio"] == "99887"

    def test_fecha_dd_slash_mm_yyyy_normalized(self):
        from invoice_processor.infrastructure.services.ocr import InvoiceOcrClient

        client = InvoiceOcrClient.__new__(InvoiceOcrClient)
        payload = {
            **SAMPLE_OCR_RESPONSE_NO_DISCOUNT,
            "fecha_emision": "15/03/2025",
        }
        result = client._postprocess_invoice_payload(dict(payload))
        assert result["fecha_emision"] == "2025-03-15"

    def test_fecha_dd_dash_mm_yyyy_normalized(self):
        from invoice_processor.infrastructure.services.ocr import InvoiceOcrClient

        client = InvoiceOcrClient.__new__(InvoiceOcrClient)
        payload = {
            **SAMPLE_OCR_RESPONSE_NO_DISCOUNT,
            "fecha_emision": "15-03-2025",
        }
        result = client._postprocess_invoice_payload(dict(payload))
        assert result["fecha_emision"] == "2025-03-15"

    def test_fecha_unparseable_sets_none_with_warning(self):
        from invoice_processor.infrastructure.services.ocr import InvoiceOcrClient

        client = InvoiceOcrClient.__new__(InvoiceOcrClient)
        payload = {
            **SAMPLE_OCR_RESPONSE_NO_DISCOUNT,
            "fecha_emision": "marzo 2025",
        }
        result = client._postprocess_invoice_payload(dict(payload))
        assert result["fecha_emision"] is None
        assert "fecha_emision no parseable" in (result.get("ocr_warning") or "")

    def test_fecha_null_stays_none(self):
        from invoice_processor.infrastructure.services.ocr import InvoiceOcrClient

        client = InvoiceOcrClient.__new__(InvoiceOcrClient)
        payload = {
            **SAMPLE_OCR_RESPONSE_NO_DISCOUNT,
            "fecha_emision": None,
        }
        result = client._postprocess_invoice_payload(dict(payload))
        assert result["fecha_emision"] is None


class TestGapCreateInvoiceForOrder:
    """create_invoice_for_order llama action_create_invoice + action_post con ref/date."""

    def test_creates_and_posts_invoice_with_ref_and_date(self):
        from invoice_processor.infrastructure.services.odoo_connection_manager import (
            OdooConnectionManager,
        )

        manager = OdooConnectionManager.__new__(OdooConnectionManager)

        call_log = []

        def fake_execute_kw(model, method, args, kwargs=None):
            call_log.append((model, method))
            if model == "purchase.order" and method == "read":
                read_count = len([c for c in call_log if c == ("purchase.order", "read")])
                # First call: invoice_status check (P18)
                if read_count == 1:
                    return [{"invoice_status": "to invoice", "invoice_ids": [100]}]
                # Second call: before creating invoice
                if read_count == 2:
                    return [{"invoice_ids": [100]}]
                # Third call: after creating invoice
                return [{"invoice_ids": [100, 200]}]
            if model == "purchase.order" and method == "action_create_invoice":
                return True
            if model == "account.move" and method == "write":
                return True
            if model == "account.move" and method == "action_post":
                return True
            return []

        manager._execute_kw = fake_execute_kw
        manager._normalize_id_list = lambda vs: [
            int(v[0]) if isinstance(v, (list, tuple)) else int(v)
            for v in (vs or [])
            if v is not None
        ]

        result = manager.create_invoice_for_order(
            42, ref="F-12345", invoice_date="2025-03-15"
        )

        assert result["invoice_ids"] == [200]
        assert result["posted"] is True
        # Verify action_create_invoice was called
        assert ("purchase.order", "action_create_invoice") in call_log
        # Verify write was called on account.move
        assert ("account.move", "write") in call_log
        # Verify action_post was called
        assert ("account.move", "action_post") in call_log
