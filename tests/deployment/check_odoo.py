# tests/deployment/check_odoo.py
import sys
import ssl
from invoice_processor.config.config import init_config, CONFIG_PATH
from invoice_processor.infrastructure.services.odoo_connection_manager import OdooConnectionManager

def main():
    print("🔍 Verificando conexión a Odoo...")
    try:
        # Cada test debe inicializar la configuración para cargar las variables
        init_config(CONFIG_PATH)
        
        odoo_manager = OdooConnectionManager()
        
        # Se llama al método que agregamos para hacer una consulta real a Odoo.
        connection_status = odoo_manager.check_connection()

        if connection_status:
            print(f"✅ Conexión a Odoo exitosa.")
            return True
        else:
            raise Exception("El método 'check_connection' devolvió un estado de fallo.")
    except Exception as e:
        print(f"❌ ERROR: Falló la conexión a Odoo. {e}")
        return False

if __name__ == "__main__":
    if not main():
        sys.exit(1)
