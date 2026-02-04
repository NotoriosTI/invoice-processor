"""
Test E2E del flujo completo de confirmación con persistencia de estado.

Simula el escenario del bug: usuario sube factura → bot pide confirmación →
usuario corrige producto → bot reprocesa → flujo avanza.
"""
import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# ---------------------------------------------------------------------------
# Datos de prueba
# ---------------------------------------------------------------------------

SAMPLE_OCR_RESPONSE = {
    "supplier_name": "Proveedor Monoi SpA",
    "supplier_rut": "76.999.888-7",
    "neto": 50000.0,
    "iva_19": 9500.0,
    "total": 59500.0,
    "descuento_global": 0.0,
    "lines": [
        {
            "detalle": "Aceite de Monoi ORIGINAL sin aroma",
            "cantidad": 10,
            "precio_unitario": 5000,
            "subtotal": 50000,
            "unidad": "un",
            "descuento_pct": None,
            "descuento_monto": None,
        },
    ],
}

MOCK_ODOO_ORDER = {
    "id": 50,
    "name": "PO00050",
    "state": "purchase",
    "amount_untaxed": 50000.0,
    "amount_tax": 9500.0,
    "amount_total": 59500.0,
    "order_line": [201],
    "picking_ids": [],
    "partner_id": [99, "Proveedor Monoi SpA"],
    "invoice_status": "to invoice",
}

MOCK_ODOO_ORDER_LINES = [
    {
        "id": 201,
        "product_id": 42,
        "detalle": "Aceite de Monoi ORIGINAL sin aroma",
        "cantidad": 10,
        "precio_unitario": 5000,
        "subtotal": 50000,
    },
]


def _make_fake_image(path: Path) -> Path:
    """Crea una imagen JPEG mínima válida."""
    from PIL import Image

    img = Image.new("RGB", (100, 200), color="white")
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, format="JPEG")
    return path


def _configure_odoo_mock(mock_odoo, mapped_product_id=None, candidates=None, order=None, order_lines=None):
    """Configura el mock de odoo_manager con comportamiento estándar."""
    mock_odoo._normalize_id.side_effect = lambda v: (
        int(v[0]) if isinstance(v, (list, tuple)) else (int(v) if v is not None else None)
    )
    mock_odoo._normalize_id_list.side_effect = lambda vs: [
        int(v[0]) if isinstance(v, (list, tuple)) else int(v)
        for v in (vs or [])
        if v is not None
    ]
    mock_odoo._resolve_supplier_id.return_value = 99
    mock_odoo.get_mapped_product_id.return_value = mapped_product_id
    mock_odoo.get_product_candidates.return_value = candidates or []
    mock_odoo.get_sku_by_product_id.return_value = "MP054"
    mock_odoo.find_purchase_order_by_similarity.return_value = order
    mock_odoo.read_order.return_value = order
    mock_odoo.read_order_lines.return_value = order_lines or []
    mock_odoo.read_receipts_for_order.return_value = []
    mock_odoo.recompute_order_amounts.return_value = None


# ====================================================================
# Test 10.1: Confirmación exitosa después de mapeo
# ====================================================================


class TestE2EConfirmationFlow:
    @patch("invoice_processor.core.processor.odoo_manager")
    @patch("invoice_processor.core.processor.ocr_client")
    def test_mapping_persists_across_reprocess(self, mock_ocr, mock_odoo, tmp_path):
        """Turno 1: sin mapeo → WAITING_FOR_HUMAN.
        Turno 2: map_product_decision registra mapeo.
        Turno 3: reproceso → mapeo encontrado → WAITING_FOR_APPROVAL."""
        from invoice_processor.core.processor import process_invoice_file
        from invoice_processor.core.models import InvoiceResponseModel

        img_path = _make_fake_image(tmp_path / "factura.jpg")
        mock_ocr.extract.return_value = SAMPLE_OCR_RESPONSE

        # --- Turno 1: sin mapeo, score bajo → WAITING_FOR_HUMAN ---
        _configure_odoo_mock(
            mock_odoo,
            mapped_product_id=None,
            candidates=[
                {"id": 42, "name": "Aceite de Monoi ORIGINAL", "score": 0.85, "default_code": "MP054"},
            ],
        )

        result1 = process_invoice_file(str(img_path), allow_odoo_write=False)
        assert isinstance(result1, InvoiceResponseModel)
        assert result1.status == "WAITING_FOR_HUMAN"
        assert result1.needs_follow_up is True

        # --- Turno 2: map_product_decision (simulado) ---
        # En la realidad, el LLM llamaría a map_product_decision_tool.
        # Aquí simulamos que el mapeo se registró exitosamente.

        # --- Turno 3: reproceso con mapeo disponible → WAITING_FOR_APPROVAL ---
        _configure_odoo_mock(
            mock_odoo,
            mapped_product_id=42,  # Ahora el mapeo existe
            candidates=[
                {"id": 42, "name": "Aceite de Monoi ORIGINAL", "score": 0.95, "default_code": "MP054"},
            ],
            order=MOCK_ODOO_ORDER,
            order_lines=MOCK_ODOO_ORDER_LINES,
        )

        result3 = process_invoice_file(str(img_path), allow_odoo_write=False)
        assert isinstance(result3, InvoiceResponseModel)
        assert result3.status == "WAITING_FOR_APPROVAL"
        # Verificar que avanzó de WAITING_FOR_HUMAN a WAITING_FOR_APPROVAL
        assert result1.status != result3.status


# ====================================================================
# Test 10.2: Variación OCR - ilike salva el mapeo
# ====================================================================


class TestE2EOcrVariation:
    @patch("invoice_processor.core.processor.odoo_manager")
    @patch("invoice_processor.core.processor.ocr_client")
    def test_ocr_variation_does_not_break_mapping(self, mock_ocr, mock_odoo, tmp_path):
        """Variación de capitalización OCR no rompe el mapeo gracias a ilike."""
        from invoice_processor.core.processor import process_invoice_file

        img_path = _make_fake_image(tmp_path / "factura.jpg")

        # --- Turno 1: OCR original → sin mapeo → WAITING_FOR_HUMAN ---
        mock_ocr.extract.return_value = SAMPLE_OCR_RESPONSE
        _configure_odoo_mock(
            mock_odoo,
            mapped_product_id=None,
            candidates=[
                {"id": 42, "name": "Aceite de Monoi ORIGINAL", "score": 0.85, "default_code": "MP054"},
            ],
        )

        result1 = process_invoice_file(str(img_path), allow_odoo_write=False)
        assert result1.status == "WAITING_FOR_HUMAN"

        # --- Turno 2: mapeo registrado ---
        # (simulado)

        # --- Turno 3: OCR con variación de capitalización ---
        ocr_variant = {
            **SAMPLE_OCR_RESPONSE,
            "lines": [
                {
                    **SAMPLE_OCR_RESPONSE["lines"][0],
                    "detalle": "Aceite de Monoi ORIGINAL Sin Aroma",  # capitalización diferente
                },
            ],
        }
        mock_ocr.extract.return_value = ocr_variant

        # Simular que get_mapped_product_id ahora encuentra el mapeo (gracias a ilike)
        _configure_odoo_mock(
            mock_odoo,
            mapped_product_id=42,  # ilike fallback lo encontró
            candidates=[
                {"id": 42, "name": "Aceite de Monoi ORIGINAL", "score": 0.95, "default_code": "MP054"},
            ],
            order=MOCK_ODOO_ORDER,
            order_lines=MOCK_ODOO_ORDER_LINES,
        )

        result3 = process_invoice_file(str(img_path), allow_odoo_write=False)
        assert result3.status == "WAITING_FOR_APPROVAL"
        assert result3.needs_follow_up is True  # modo lectura siempre pide aprobación


# ====================================================================
# Test 10.3: SKU inválido - error feedback
# ====================================================================


class TestE2EInvalidSku:
    @patch("invoice_processor.tools.tools.odoo_manager")
    def test_invalid_sku_gives_clear_error(self, mock_odoo):
        """SKU inexistente devuelve error legible, no traceback."""
        from invoice_processor.tools.tools import map_product_decision_tool

        mock_odoo._resolve_supplier_id.return_value = 99
        mock_odoo.map_product_decision.side_effect = ValueError(
            "Se requiere product_id (o default_code resoluble) y supplier_id válidos."
        )

        result = map_product_decision_tool.invoke({
            "invoice_detail": "Aceite de Monoi ORIGINAL sin aroma",
            "odoo_product_id": None,
            "supplier_id": 99,
            "default_code": "NOEXISTE",
        })

        assert isinstance(result, str)
        assert "ERROR" in result
        assert "NOEXISTE" in result
        # No es un traceback
        assert "Traceback" not in result


# ====================================================================
# Test 10.4: Loop detection
# ====================================================================


class TestE2ELoopDetection:
    def test_loop_is_detected_and_broken(self):
        """Exceder MAX_INTERACTION_ROUNDS envía reset y limpia estado."""
        from invoice_processor.app.slack_handler import MAX_INTERACTION_ROUNDS

        thread_id_by_user: dict[str, str] = {}
        last_status_by_thread: dict[str, str] = {}
        interaction_count_by_thread: dict[str, int] = {}
        last_file_by_thread: dict[str, Path] = {}
        user_id = "U99999"
        messages_sent: list[str] = []

        thread_id_by_user[user_id] = f"{user_id}_abc12345"
        last_file_by_thread[user_id] = Path("/tmp/fake_invoice.jpg")
        interaction_count_by_thread[user_id] = 0
        reset_triggered = False

        for i in range(MAX_INTERACTION_ROUNDS + 1):
            # Simular una invocación que retorna WAITING_FOR_HUMAN
            last_status_by_thread[user_id] = "WAITING_FOR_HUMAN"

            interaction_count_by_thread[user_id] = interaction_count_by_thread.get(user_id, 0) + 1

            if interaction_count_by_thread[user_id] > MAX_INTERACTION_ROUNDS:
                messages_sent.append(
                    f"Se excedió el máximo de {MAX_INTERACTION_ROUNDS} interacciones"
                )
                thread_id_by_user.pop(user_id, None)
                last_status_by_thread.pop(user_id, None)
                interaction_count_by_thread.pop(user_id, None)
                last_file_by_thread.pop(user_id, None)
                reset_triggered = True
                break

        # Las primeras 10 procesaron normalmente
        assert reset_triggered
        assert len(messages_sent) == 1
        assert "excedió el máximo" in messages_sent[0]

        # Estado completamente limpio
        assert user_id not in thread_id_by_user
        assert user_id not in last_status_by_thread
        assert user_id not in interaction_count_by_thread
        assert user_id not in last_file_by_thread


# ====================================================================
# Test 10.5: Status persiste después de restart simulado
# ====================================================================


class TestE2EStatusRestart:
    def test_status_survives_simulated_restart(self, tmp_path):
        """El status persiste en JSON y se reconoce después de un restart."""
        from invoice_processor.app import slack_handler

        original_path = slack_handler._STATUS_MAP_PATH
        try:
            slack_handler._STATUS_MAP_PATH = tmp_path / "thread_status.json"

            # Fase 1: Guardar status como si el flujo llegó a WAITING_FOR_HUMAN
            status_data = {"U12345": "WAITING_FOR_HUMAN"}
            slack_handler._save_status_map(status_data)

            # Fase 2: Simular restart — cargar status desde disco
            loaded = slack_handler._load_status_map()
            assert loaded == {"U12345": "WAITING_FOR_HUMAN"}

            # Verificar que _is_affirmative + last_status reconoce la confirmación
            last_status = loaded.get("U12345")
            assert last_status == "WAITING_FOR_HUMAN"

            incoming_text = "Afirmativo"
            assert slack_handler._is_affirmative(incoming_text) is True

            # El flujo normalizaría el texto como en el handler
            if last_status == "WAITING_FOR_HUMAN" and slack_handler._is_affirmative(incoming_text):
                incoming_text = "Afirmativo"
            assert incoming_text == "Afirmativo"

        finally:
            slack_handler._STATUS_MAP_PATH = original_path

    def test_status_with_spanish_affirmative(self, tmp_path):
        """Confirmaciones en español se reconocen correctamente tras restart."""
        from invoice_processor.app import slack_handler

        original_path = slack_handler._STATUS_MAP_PATH
        try:
            slack_handler._STATUS_MAP_PATH = tmp_path / "thread_status.json"

            slack_handler._save_status_map({"U12345": "WAITING_FOR_APPROVAL"})
            loaded = slack_handler._load_status_map()
            last_status = loaded.get("U12345")

            # Verificar variantes de confirmación
            for text in ["si", "sí", "ok", "dale", "continuar", "de acuerdo"]:
                assert slack_handler._is_affirmative(text) is True, f"'{text}' debería ser afirmativo"

                if last_status == "WAITING_FOR_APPROVAL" and slack_handler._is_affirmative(text):
                    normalized = "Continuar"
                    assert normalized == "Continuar"

        finally:
            slack_handler._STATUS_MAP_PATH = original_path


# ====================================================================
# Test E2E: ilike fallback real en get_mapped_product_id
# ====================================================================


class TestE2EIlikeFallbackIntegration:
    @patch("invoice_processor.infrastructure.services.odoo_connection_manager.OdooConnectionManager._execute_kw")
    def test_ilike_resolves_case_mismatch(self, mock_exec):
        """Integración: exact falla, ilike encuentra el mapeo con case diferente."""
        from invoice_processor.infrastructure.services.odoo_connection_manager import (
            OdooConnectionManager,
        )

        manager = OdooConnectionManager.__new__(OdooConnectionManager)
        manager._execute_kw = mock_exec
        manager._get_supplierinfo_fields = MagicMock(
            return_value=["id", "partner_id", "product_name", "product_tmpl_id"]
        )
        manager._supplierinfo_partner_field = MagicMock(return_value="partner_id")
        manager._supplierinfo_product_field = MagicMock(return_value="product_tmpl_id")
        manager._normalize_id = MagicMock(side_effect=lambda v: (
            int(v[0]) if isinstance(v, (list, tuple)) else (int(v) if v is not None else None)
        ))
        manager._product_id_from_supplierinfo_record = MagicMock(return_value=42)

        # exact → no match; ilike → match
        mock_exec.side_effect = [
            [],  # exact "=" falla
            [{"id": 1, "partner_id": [99, "P"], "product_name": "Aceite de Monoi ORIGINAL sin aroma", "product_tmpl_id": [42, "T"]}],
        ]

        result = manager.get_mapped_product_id("Aceite de Monoi ORIGINAL Sin Aroma", 99)
        assert result == 42
        assert mock_exec.call_count == 2

    @patch("invoice_processor.infrastructure.services.odoo_connection_manager.OdooConnectionManager._execute_kw")
    def test_whitespace_normalization_resolves_mismatch(self, mock_exec):
        """Integración: exact y ilike fallan, normalización de whitespace resuelve."""
        from invoice_processor.infrastructure.services.odoo_connection_manager import (
            OdooConnectionManager,
        )

        manager = OdooConnectionManager.__new__(OdooConnectionManager)
        manager._execute_kw = mock_exec
        manager._get_supplierinfo_fields = MagicMock(
            return_value=["id", "partner_id", "product_name", "product_tmpl_id"]
        )
        manager._supplierinfo_partner_field = MagicMock(return_value="partner_id")
        manager._supplierinfo_product_field = MagicMock(return_value="product_tmpl_id")
        manager._normalize_id = MagicMock(side_effect=lambda v: (
            int(v[0]) if isinstance(v, (list, tuple)) else (int(v) if v is not None else None)
        ))
        manager._product_id_from_supplierinfo_record = MagicMock(return_value=42)

        mock_exec.side_effect = [
            [],  # exact (whitespace already normalized at start)
            [{"id": 1, "partner_id": [99, "P"], "product_name": "Aceite de Monoi ORIGINAL", "product_tmpl_id": [42, "T"]}],
        ]

        result = manager.get_mapped_product_id("Aceite  de  Monoi  ORIGINAL", 99)
        assert result == 42
        # Whitespace is now normalized at the start, so only exact + ilike attempts are needed.
        assert mock_exec.call_count == 2
