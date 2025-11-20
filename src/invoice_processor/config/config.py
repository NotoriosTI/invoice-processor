from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    odoo_url: str
    odoo_db: str
    odoo_username: str
    odoo_password: str
    openai_api_key: str
    openai_model: str = "gpt-4o-mini"

    class config:
        env_file = ".env"
        case_sensitive = True

@lru_cache
def get_settings(): -> Settings:
    return Settings()