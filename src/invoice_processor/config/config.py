from functools import lru_cache
from env_manager import init_config, require_config, get_config
from pathlib import Path
import yaml
import os
import logging
from typing import Dict, List, Any

PROJECT_ROOT = Path().cwd()
CONFIG_PATH = PROJECT_ROOT / "config/config_vars.yaml"

init_config(
    CONFIG_PATH,
    secret_origin=None,
    gcp_project_id=None,
    strict=None,
    dotenv_path=None,
    debug=False,
)

DATA_PATH = Path(get_config("DATA_PATH"))
DEFAULT_INVOICE_PATH = DATA_PATH / "factura.jpg"


class Settings:
    def __init__(self):
        self.odoo_url = require_config("ODOO_PROD_URL")
        self.odoo_db = require_config("ODOO_PROD_DB")
        self.odoo_username = require_config("ODOO_PROD_USERNAME")
        self.odoo_password = require_config("ODOO_PROD_PASSWORD")
        self.openai_api_key = require_config("OPENAI_API_KEY")
        self.llm_model = get_config("LLM_MODEL", "gpt-5.2")
        self.embedding_model = get_config("EMBEDDING_MODEL", "text-embedding-3-small")
        self.slack_app_token = require_config("SLACK_APP_TOKEN")
        self.slack_bot_token = require_config("SLACK_BOT_TOKEN")
        self.slack_debug_logs = get_config("SLACK_DEBUG_LOGS", False)
        self.data_path = DATA_PATH
        self.default_invoice_path = DEFAULT_INVOICE_PATH
        self.allowed_users_file = get_config(
            "SLACK_ALLOWED_USERS_FILE", "config/allowed_users_config.yaml"
        )
        # Impuestos de compra por defecto (lista de IDs) si una línea no trae impuestos.
        self.default_purchase_tax_ids = get_config("DEFAULT_PURCHASE_TAX_IDS", "")
        # LangSmith tracing (solo variables LANGSMITH_*)
        self.langsmith_tracing = get_config("LANGSMITH_TRACING", False)
        self.langsmith_endpoint = get_config("LANGSMITH_ENDPOINT", None)
        self.langsmith_api_key = get_config("LANGSMITH_API_KEY", None)
        self.langsmith_project = get_config("LANGSMITH_PROJECT", None)
        self.langsmith_project_id = get_config("LANGSMITH_PROJECT_ID", None)
        self._configure_langsmith_env()

    def _configure_langsmith_env(self) -> None:
        """Configura variables de entorno para LangSmith sin sobrescribir si ya están seteadas."""
        if str(self.langsmith_tracing).lower() in {"true", "1", "yes"}:
            os.environ.setdefault("LANGSMITH_TRACING", "true")
            # Mapear a las variables que consume LangChain para activar tracing automático.
            os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
        if self.langsmith_endpoint:
            os.environ.setdefault("LANGSMITH_ENDPOINT", str(self.langsmith_endpoint))
            os.environ.setdefault("LANGCHAIN_ENDPOINT", str(self.langsmith_endpoint))
        if self.langsmith_api_key:
            os.environ.setdefault("LANGSMITH_API_KEY", str(self.langsmith_api_key))
            os.environ.setdefault("LANGCHAIN_API_KEY", str(self.langsmith_api_key))
        if self.langsmith_project:
            os.environ.setdefault("LANGSMITH_PROJECT", str(self.langsmith_project))
            os.environ.setdefault("LANGCHAIN_PROJECT", str(self.langsmith_project))
        if self.langsmith_project_id:
            os.environ.setdefault(
                "LANGSMITH_PROJECT_ID", str(self.langsmith_project_id)
            )


def load_allowed_users():
    settings = get_settings()
    path = Path(settings.allowed_users_file)
    if not path.exists():
        logging.warning("Archivo de usuarios permitidos no encontrado: %s", path)
        return {"enabled": False, "users": []}
    data = yaml.safe_load(path.read_text()) or {}
    slack_access = data.get("slack_access", {})
    enabled_flag = slack_access.get("enabled", slack_access.get("enable", False))
    users_raw = slack_access.get("users", [])
    users: List[Dict[str, str]] = []
    for entry in users_raw or []:
        if not isinstance(entry, dict):
            logging.warning(
                "Entrada de usuario inválida (no es dict), se ignora: %s", entry
            )
            continue
        uid = entry.get("id")
        name = entry.get("name", "")
        if not uid or not isinstance(uid, str):
            logging.warning("Entrada de usuario sin id válido, se ignora: %s", entry)
            continue
        users.append({"id": uid, "name": str(name)})
    return {
        "enabled": bool(enabled_flag),
        "users": users,
    }


def is_user_allowed(user_id: str) -> bool:
    info = load_allowed_users()
    if not info.get("enabled"):
        return False
    return any(user_id == u.get("id") for u in info.get("users", []))


def get_settings() -> Settings:
    return Settings()
