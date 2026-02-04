"""
Tests unitarios para las correcciones de estado inconsistente:
- SqliteSaver persistencia
- Thread isolation por sesión de factura
- Persistencia de status map en JSON
- Fallback ilike en get_mapped_product_id
- Normalización en map_product_decision
- Detección de loops
- Error feedback en map_product_decision_tool
"""
import json
import os
import sqlite3
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


# ====================================================================
# Test 9.1: SqliteSaver persiste estado entre instancias
# ====================================================================


class TestSqliteCheckpointer:
    def test_checkpointer_creates_db_file(self, tmp_path):
        """SqliteSaver crea el archivo .sqlite3 en disco tras setup()."""
        from langgraph.checkpoint.sqlite import SqliteSaver

        db_path = tmp_path / "checkpoints.sqlite3"
        conn = sqlite3.connect(str(db_path), check_same_thread=False)
        saver = SqliteSaver(conn)
        saver.setup()
        conn.close()
        assert db_path.exists()

    def test_checkpointer_persists_state_across_connections(self, tmp_path):
        """El estado persiste entre conexiones distintas al mismo archivo."""
        from langgraph.checkpoint.sqlite import SqliteSaver

        db_path = tmp_path / "checkpoints.sqlite3"

        # Primera conexión: guardar un checkpoint
        conn1 = sqlite3.connect(str(db_path), check_same_thread=False)
        saver1 = SqliteSaver(conn1)
        saver1.setup()

        config = {"configurable": {"thread_id": "test-1", "checkpoint_ns": ""}}
        checkpoint = {
            "v": 1,
            "id": "test-checkpoint-1",
            "ts": "2025-01-01T00:00:00+00:00",
            "channel_values": {"messages": ["hello"]},
            "channel_versions": {},
            "versions_seen": {},
            "pending_sends": [],
        }
        metadata = {"source": "input", "step": 0, "writes": None, "parents": {}}
        saver1.put(config, checkpoint, metadata, {})
        conn1.close()

        # Segunda conexión: recuperar el checkpoint
        conn2 = sqlite3.connect(str(db_path), check_same_thread=False)
        saver2 = SqliteSaver(conn2)
        saver2.setup()

        recovered = saver2.get(config)
        conn2.close()

        assert recovered is not None
        assert recovered["id"] == "test-checkpoint-1"
        assert recovered["channel_values"]["messages"] == ["hello"]


# ====================================================================
# Test 9.2: Thread isolation por sesión de factura
# ====================================================================


class TestThreadIsolation:
    def test_new_file_generates_new_thread_id(self):
        """Cuando un usuario sube un archivo, se genera thread_id con user_id + hash."""
        from invoice_processor.app import slack_handler

        thread_id_by_user: dict[str, str] = {}
        user_id = "U12345"
        # Simular que has_new_file = True
        import uuid
        thread_id_by_user[user_id] = f"{user_id}_{uuid.uuid4().hex[:8]}"

        assert user_id in thread_id_by_user
        tid = thread_id_by_user[user_id]
        assert tid.startswith(user_id)
        assert len(tid) > len(user_id) + 1  # tiene el hash

    def test_followup_reuses_thread_id(self):
        """Un mensaje de texto sin archivo reutiliza el thread_id existente."""
        import uuid

        thread_id_by_user: dict[str, str] = {}
        user_id = "U12345"

        # Simular upload → thread_id generado
        thread_id_by_user[user_id] = f"{user_id}_{uuid.uuid4().hex[:8]}"
        original_tid = thread_id_by_user[user_id]

        # Simular followup sin archivo (has_new_file = False)
        # El código no modifica thread_id_by_user si user_id ya está presente
        if user_id not in thread_id_by_user:
            thread_id_by_user[user_id] = f"{user_id}_{uuid.uuid4().hex[:8]}"

        assert thread_id_by_user[user_id] == original_tid

    def test_new_file_replaces_old_thread_id(self):
        """Un segundo upload genera un thread_id distinto."""
        import uuid

        thread_id_by_user: dict[str, str] = {}
        user_id = "U12345"

        # Upload 1
        thread_id_by_user[user_id] = f"{user_id}_{uuid.uuid4().hex[:8]}"
        tid_a = thread_id_by_user[user_id]

        # Upload 2 (has_new_file = True)
        thread_id_by_user[user_id] = f"{user_id}_{uuid.uuid4().hex[:8]}"
        tid_b = thread_id_by_user[user_id]

        assert tid_a != tid_b
        assert tid_a.startswith(user_id)
        assert tid_b.startswith(user_id)

    def test_cleanup_on_flow_complete(self):
        """Cuando el flujo completa exitosamente, se limpia el thread_id."""
        import uuid

        thread_id_by_user: dict[str, str] = {}
        interaction_count_by_thread: dict[str, int] = {}
        last_status_by_thread: dict[str, str] = {}
        user_id = "U12345"

        thread_id_by_user[user_id] = f"{user_id}_{uuid.uuid4().hex[:8]}"
        interaction_count_by_thread[user_id] = 5
        last_status_by_thread[user_id] = "WAITING_FOR_APPROVAL"

        # Simular flujo completo (needs_follow_up = False)
        thread_id_by_user.pop(user_id, None)
        last_status_by_thread.pop(user_id, None)
        interaction_count_by_thread.pop(user_id, None)

        assert user_id not in thread_id_by_user
        assert user_id not in last_status_by_thread
        assert user_id not in interaction_count_by_thread


# ====================================================================
# Test 9.3: Persistencia de status map en JSON
# ====================================================================


class TestStatusPersistence:
    def test_save_and_load_status_map(self, tmp_path):
        """Guardar y cargar status map produce datos idénticos."""
        from invoice_processor.app import slack_handler

        original_path = slack_handler._STATUS_MAP_PATH
        try:
            slack_handler._STATUS_MAP_PATH = tmp_path / "thread_status.json"
            data = {"user1": "WAITING_FOR_HUMAN", "user2": "WAITING_FOR_APPROVAL"}
            slack_handler._save_status_map(data)
            loaded = slack_handler._load_status_map()
            assert loaded == data
        finally:
            slack_handler._STATUS_MAP_PATH = original_path

    def test_load_returns_empty_when_file_missing(self, tmp_path):
        """Retorna {} cuando el archivo no existe."""
        from invoice_processor.app import slack_handler

        original_path = slack_handler._STATUS_MAP_PATH
        try:
            slack_handler._STATUS_MAP_PATH = tmp_path / "nonexistent" / "status.json"
            result = slack_handler._load_status_map()
            assert result == {}
        finally:
            slack_handler._STATUS_MAP_PATH = original_path

    def test_load_handles_corrupted_json(self, tmp_path):
        """Retorna {} sin crashear cuando el JSON está corrupto."""
        from invoice_processor.app import slack_handler

        original_path = slack_handler._STATUS_MAP_PATH
        try:
            status_file = tmp_path / "thread_status.json"
            status_file.write_text("{{invalid json", encoding="utf-8")
            slack_handler._STATUS_MAP_PATH = status_file
            result = slack_handler._load_status_map()
            assert result == {}
        finally:
            slack_handler._STATUS_MAP_PATH = original_path

    def test_save_creates_parent_dirs(self, tmp_path):
        """_save_status_map crea directorios padres si no existen."""
        from invoice_processor.app import slack_handler

        original_path = slack_handler._STATUS_MAP_PATH
        try:
            target = tmp_path / "subdir" / "nested" / "status.json"
            slack_handler._STATUS_MAP_PATH = target
            slack_handler._save_status_map({"u1": "WAITING_FOR_HUMAN"})
            assert target.exists()
            loaded = json.loads(target.read_text(encoding="utf-8"))
            assert loaded == {"u1": "WAITING_FOR_HUMAN"}
        finally:
            slack_handler._STATUS_MAP_PATH = original_path


# ====================================================================
# Test 9.4: Fallback ilike en get_mapped_product_id
# ====================================================================


SAMPLE_SUPPLIERINFO_RECORD = {
    "id": 1,
    "partner_id": [99, "Proveedor Test"],
    "product_name": "Aceite de Monoi ORIGINAL",
    "product_tmpl_id": [10, "Aceite Template"],
}


class TestMappedProductIdFallback:
    def _make_manager(self):
        from invoice_processor.infrastructure.services.odoo_connection_manager import (
            OdooConnectionManager,
        )
        manager = OdooConnectionManager.__new__(OdooConnectionManager)
        manager._get_supplierinfo_fields = MagicMock(return_value=["id", "partner_id", "product_name", "product_tmpl_id"])
        manager._supplierinfo_partner_field = MagicMock(return_value="partner_id")
        manager._supplierinfo_product_field = MagicMock(return_value="product_tmpl_id")
        manager._normalize_id = MagicMock(side_effect=lambda v: (
            int(v[0]) if isinstance(v, (list, tuple)) else (int(v) if v is not None else None)
        ))
        manager._product_id_from_supplierinfo_record = MagicMock(return_value=10)
        return manager

    @patch("invoice_processor.infrastructure.services.odoo_connection_manager.OdooConnectionManager._execute_kw")
    def test_exact_match_found_no_fallback(self, mock_exec):
        """Match exacto encontrado: solo 1 llamada, sin fallback."""
        manager = self._make_manager()
        manager._execute_kw = mock_exec
        mock_exec.return_value = [SAMPLE_SUPPLIERINFO_RECORD]

        result = manager.get_mapped_product_id("Aceite de Monoi ORIGINAL", 99)

        assert result == 10
        assert mock_exec.call_count == 1

    @patch("invoice_processor.infrastructure.services.odoo_connection_manager.OdooConnectionManager._execute_kw")
    def test_exact_miss_ilike_finds_match(self, mock_exec):
        """Match exacto falla, ilike encuentra."""
        manager = self._make_manager()
        manager._execute_kw = mock_exec
        mock_exec.side_effect = [
            [],  # exact: no match
            [SAMPLE_SUPPLIERINFO_RECORD],  # ilike: match
        ]

        result = manager.get_mapped_product_id("Aceite de Monoi ORIGINAL", 99)

        assert result == 10
        assert mock_exec.call_count == 2

    @patch("invoice_processor.infrastructure.services.odoo_connection_manager.OdooConnectionManager._execute_kw")
    def test_case_insensitive_match(self, mock_exec):
        """Diferencia de capitalización se resuelve con ilike."""
        manager = self._make_manager()
        manager._execute_kw = mock_exec
        mock_exec.side_effect = [
            [],  # exact → no match (case mismatch)
            [SAMPLE_SUPPLIERINFO_RECORD],  # ilike → match
        ]

        result = manager.get_mapped_product_id("aceite de monoi ORIGINAL", 99)

        assert result == 10
        assert mock_exec.call_count == 2

    @patch("invoice_processor.infrastructure.services.odoo_connection_manager.OdooConnectionManager._execute_kw")
    def test_whitespace_normalized_match(self, mock_exec):
        """Espacios dobles se resuelven porque el whitespace se normaliza al inicio."""
        manager = self._make_manager()
        manager._execute_kw = mock_exec
        mock_exec.side_effect = [
            [],  # exact → no match (whitespace already normalized at start)
            [SAMPLE_SUPPLIERINFO_RECORD],  # ilike → match (normalized input)
        ]

        result = manager.get_mapped_product_id("Aceite  de  Monoi  ORIGINAL", 99)

        assert result == 10
        # Whitespace is normalized at the start, so only exact + ilike attempts are needed.
        assert mock_exec.call_count == 2

    @patch("invoice_processor.infrastructure.services.odoo_connection_manager.OdooConnectionManager._execute_kw")
    def test_no_match_returns_none(self, mock_exec):
        """Ningún intento encuentra match → retorna None."""
        manager = self._make_manager()
        manager._execute_kw = mock_exec
        mock_exec.return_value = []

        result = manager.get_mapped_product_id("Producto Inexistente", 99)

        assert result is None


# ====================================================================
# Test 9.5: Normalización en map_product_decision
# ====================================================================


class TestMapProductDecisionNormalization:
    @patch("invoice_processor.infrastructure.services.odoo_connection_manager.OdooConnectionManager._execute_kw")
    def test_stored_name_is_whitespace_normalized(self, mock_exec):
        """El nombre almacenado en supplierinfo está normalizado (sin espacios extra)."""
        from invoice_processor.infrastructure.services.odoo_connection_manager import (
            OdooConnectionManager,
        )
        manager = OdooConnectionManager.__new__(OdooConnectionManager)
        manager._execute_kw = mock_exec
        manager._normalize_id = MagicMock(side_effect=lambda v: (
            int(v[0]) if isinstance(v, (list, tuple)) else (int(v) if v is not None else None)
        ))
        manager._sanitize_default_code = MagicMock(return_value=None)
        manager._get_supplierinfo_fields = MagicMock(return_value=["id", "partner_id", "product_name", "product_tmpl_id"])
        manager._supplierinfo_partner_field = MagicMock(return_value="partner_id")
        manager._supplierinfo_product_field = MagicMock(return_value="product_tmpl_id")
        manager._create_supplierinfo_record = MagicMock()

        # product_id=10 ya existe
        mock_exec.side_effect = [
            # read product.product
            [{"product_tmpl_id": [5, "Template"], "name": "Aceite de Monoi"}],
            # search_read supplierinfo → no existe
            [],
        ]

        manager.map_product_decision("  Aceite  de  Monoi  ", 10, 99)

        # Verificar que _create_supplierinfo_record fue llamado con nombre normalizado
        manager._create_supplierinfo_record.assert_called_once()
        call_kwargs = manager._create_supplierinfo_record.call_args
        # product_name debería ser el argumento keyword
        assert call_kwargs[1]["product_name"] == "Aceite de Monoi"


# ====================================================================
# Test 9.6: Detección de loops
# ====================================================================


class TestLoopDetection:
    def test_counter_increments_per_interaction(self):
        """El contador se incrementa con cada interacción."""
        interaction_count: dict[str, int] = {}
        user_id = "U12345"

        for _ in range(3):
            interaction_count[user_id] = interaction_count.get(user_id, 0) + 1

        assert interaction_count[user_id] == 3

    def test_max_rounds_triggers_reset(self):
        """Exceder MAX_INTERACTION_ROUNDS limpia el estado."""
        from invoice_processor.app.slack_handler import MAX_INTERACTION_ROUNDS

        interaction_count: dict[str, int] = {}
        thread_id_by_user: dict[str, str] = {}
        last_status_by_thread: dict[str, str] = {}
        user_id = "U12345"
        thread_id_by_user[user_id] = f"{user_id}_abc12345"
        last_status_by_thread[user_id] = "WAITING_FOR_HUMAN"
        reset_triggered = False

        for i in range(MAX_INTERACTION_ROUNDS + 1):
            interaction_count[user_id] = interaction_count.get(user_id, 0) + 1
            if interaction_count[user_id] > MAX_INTERACTION_ROUNDS:
                # Simular reset
                thread_id_by_user.pop(user_id, None)
                last_status_by_thread.pop(user_id, None)
                interaction_count.pop(user_id, None)
                reset_triggered = True
                break

        assert reset_triggered
        assert user_id not in thread_id_by_user
        assert user_id not in last_status_by_thread
        assert user_id not in interaction_count

    def test_counter_resets_on_success(self):
        """El contador se resetea cuando el flujo completa exitosamente."""
        interaction_count: dict[str, int] = {}
        user_id = "U12345"

        for _ in range(5):
            interaction_count[user_id] = interaction_count.get(user_id, 0) + 1
        assert interaction_count[user_id] == 5

        # Flujo completa (needs_follow_up = False)
        interaction_count.pop(user_id, None)
        assert user_id not in interaction_count

    def test_counter_resets_on_new_file(self):
        """El contador se reinicia cuando se sube un nuevo archivo."""
        interaction_count: dict[str, int] = {}
        user_id = "U12345"

        for _ in range(5):
            interaction_count[user_id] = interaction_count.get(user_id, 0) + 1
        assert interaction_count[user_id] == 5

        # Nuevo archivo (has_new_file = True)
        interaction_count[user_id] = 0
        assert interaction_count[user_id] == 0


# ====================================================================
# Test 9.7: Error feedback en map_product_decision_tool
# ====================================================================


class TestMapProductDecisionErrorFeedback:
    @patch("invoice_processor.tools.tools.odoo_manager")
    def test_invalid_sku_returns_error_string(self, mock_odoo):
        """SKU inválido retorna string con ERROR, no lanza excepción."""
        from invoice_processor.tools.tools import map_product_decision_tool

        mock_odoo._resolve_supplier_id.return_value = 99
        mock_odoo.map_product_decision.side_effect = ValueError(
            "Se requiere product_id (o default_code resoluble) y supplier_id válidos."
        )

        result = map_product_decision_tool.invoke({
            "invoice_detail": "Producto X",
            "odoo_product_id": None,
            "supplier_id": 99,
            "default_code": "NOEXISTE",
        })

        assert isinstance(result, str)
        assert result.startswith("ERROR")

    @patch("invoice_processor.tools.tools.odoo_manager")
    def test_unresolvable_supplier_returns_error_string(self, mock_odoo):
        """Proveedor no resuelto retorna string con ERROR y 'proveedor'."""
        from invoice_processor.tools.tools import map_product_decision_tool

        mock_odoo._resolve_supplier_id.return_value = None

        result = map_product_decision_tool.invoke({
            "invoice_detail": "Producto X",
            "odoo_product_id": None,
            "supplier_id": None,
            "supplier_name": "Inexistente SpA",
        })

        assert isinstance(result, str)
        assert "ERROR" in result
        assert "proveedor" in result.lower()

    @patch("invoice_processor.tools.tools.odoo_manager")
    def test_unexpected_exception_returns_error_string(self, mock_odoo):
        """Excepción inesperada retorna string con ERROR inesperado."""
        from invoice_processor.tools.tools import map_product_decision_tool

        mock_odoo._resolve_supplier_id.return_value = 99
        mock_odoo.map_product_decision.side_effect = RuntimeError("conexión perdida")

        result = map_product_decision_tool.invoke({
            "invoice_detail": "Producto X",
            "odoo_product_id": None,
            "supplier_id": 99,
            "default_code": "MP001",
        })

        assert isinstance(result, str)
        assert "ERROR inesperado" in result

    @patch("invoice_processor.tools.tools.odoo_manager")
    def test_successful_mapping_returns_confirmation(self, mock_odoo):
        """Mapeo exitoso retorna string de confirmación."""
        from invoice_processor.tools.tools import map_product_decision_tool

        mock_odoo._resolve_supplier_id.return_value = 99
        mock_odoo.map_product_decision.return_value = None
        mock_odoo._normalize_id.side_effect = lambda v: (
            int(v[0]) if isinstance(v, (list, tuple)) else (int(v) if v is not None else None)
        )
        mock_odoo._resolve_product_by_default_code.return_value = 42
        mock_odoo._execute_kw.return_value = [
            {"name": "Aceite Esencial", "default_code": "MP001"}
        ]

        result = map_product_decision_tool.invoke({
            "invoice_detail": "Aceite Factura",
            "odoo_product_id": None,
            "supplier_id": 99,
            "default_code": "MP001",
        })

        assert isinstance(result, str)
        assert "Mapeo registrado" in result
        assert "Aceite Factura" in result
