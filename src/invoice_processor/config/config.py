from functools import lru_cache
from env_manager import init_config, require_config, get_config

init_config(
    "config/config_vars.yaml",
    secret_origin=None,
    gcp_project_id=None,
    strict=None,
    dotenv_path=None,
    debug=False,
)


class Settings:
    def __init__(self):
        self.odoo_url = require_config("ODOO_TEST_URL")
        self.odoo_db = require_config("ODOO_TEST_DB")
        self.odoo_username = require_config("ODOO_TEST_USERNAME")
        self.odoo_password = require_config("ODOO_TEST_PASSWORD")
        self.openai_api_key = require_config("OPENAI_API_KEY")
        self.openai_model = get_config("OPENAI_MODEL", "gpt-4o-mini")
        self.slack_app_token = require_config("SLACK_APP_TOKEN")
        self.slack_bot_token = require_config("SLACK_BOT_TOKEN")


@lru_cache
def get_settings() -> Settings:
    return Settings()
