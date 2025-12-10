# tests/deployment/check_gcp_secrets.py
import sys
import os
from invoice_processor.config.config import init_config, CONFIG_PATH

def main():
    print("🔍 Verificando conexión con GCP Secret Manager (cargando una variable)...")
    try:
        # Cada test debe inicializar la configuración para cargar las variables
        init_config(CONFIG_PATH)

        # Elige una variable que deba existir si la carga desde GCP fue exitosa
        secret_var = os.getenv("ODOO_TEST_URL")
        
        if secret_var:
            print("✅ Conexión con GCP Secret Manager exitosa (la variable ODOO_TEST_URL fue cargada).")
            return True
        else:
            raise ValueError("La variable de entorno ODOO_TEST_URL no se encontró. La carga desde GCP Secret Manager pudo haber fallado.")

    except Exception as e:
        print(f"❌ ERROR: Falló la prueba de GCP Secret Manager. {e}")
        return False

if __name__ == "__main__":
    if not main():
        sys.exit(1)