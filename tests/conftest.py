"""
Configuración global de pytest.

Establece variables de entorno ficticias que init_config requiere
ANTES de que cualquier módulo del proyecto se importe.
"""
import os

# Variables que env_manager necesita y que no están en .env de desarrollo.
_FAKE_ENV = {
    "INVOICE_SLACK_APP_TOKEN": "xapp-fake-token-for-tests",
    "INVOICE_SLACK_BOT_TOKEN": "xoxb-fake-token-for-tests",
}

for key, value in _FAKE_ENV.items():
    os.environ.setdefault(key, value)
