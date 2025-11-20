from pydantic_settings import BaseSettings
from functools import lru_cache
from env_manager import init_config, required_config, get_config

init_config(
    "config/config_vars.yaml",
    secret_origin=None, 
    gcp_project_id=None,
    strict=None,
    dotenv_path=None,
    debug=False,
)

class Settings(self):
    self.odoo_url: str = required_config("ODOO_URL")
    self.odoo_db: str = required_config("ODOO_DB")
    self.odoo_username: str = required_config("ODOO_USERNAME")
    self.odoo_password: str = required_config("ODOO_PASSWORD")
    self.openai_api_key: str = required_config("OPENAI_API_KEY")
    self.openai_model: str = get_config("OPENAI_MODEL", default="gpt-4o-mini")
    

@lru_cache
def get_settings(): -> Settings:
    return Settings()