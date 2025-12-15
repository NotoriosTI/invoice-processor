from functools import lru_cache
from env_manager import init_config, require_config, get_config
from pathlib import Path
import yaml

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
        self.odoo_url = require_config("ODOO_TEST_URL")
        self.odoo_db = require_config("ODOO_TEST_DB")
        self.odoo_username = require_config("ODOO_TEST_USERNAME")
        self.odoo_password = require_config("ODOO_TEST_PASSWORD")
        self.openai_api_key = require_config("OPENAI_API_KEY")
        self.llm_model = get_config("LLM_MODEL", "gpt-4o-mini")
        self.slack_app_token = require_config("SLACK_APP_TOKEN")
        self.slack_bot_token = require_config("SLACK_BOT_TOKEN")
        self.slack_debug_logs = get_config("SLACK_DEBUG_LOGS", False)
        self.data_path = DATA_PATH
        self.default_invoice_path = DEFAULT_INVOICE_PATH
        self.allowed_users_file = get_config("SLACK_ALLOWED_USERS_FILE", "config/allowed_users_config.yaml")
        # Impuestos de compra por defecto (lista de IDs) si una línea no trae impuestos.
        self.default_purchase_tax_ids = get_config("DEFAULT_PURCHASE_TAX_IDS", "")

def load_allowed_users():
    settings = get_settings()
    path = Path(settings.allowed_users_file)
    if not path.exists():
        return {"enabled": False, "users": []}
    data = yaml.safe_load(path.read_text()) or {}
    slack_access = data.get("slack_access", {})
    enabled_flag = slack_access.get("enabled", slack_access.get("enable", False))
    return {
        "enabled": bool(enabled_flag),
        "users": slack_access.get("users", []),
    }

def get_settings() -> Settings:
    return Settings()
