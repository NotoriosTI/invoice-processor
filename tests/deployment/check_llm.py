# tests/deployment/check_llm.py
import sys
import os
from openai import OpenAI
from invoice_processor.config.config import init_config, CONFIG_PATH

def main():
    print("🔍 Verificando conexión con el LLM...")
    try:
        # Cada test debe inicializar la configuración para cargar las variables
        init_config(CONFIG_PATH)

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("La variable de entorno OPENAI_API_KEY no está definida.")

        client = OpenAI(api_key=api_key)
        
        # Realiza una llamada de prueba simple, pequeña y de bajo costo
        llm_model = os.getenv("LLM_MODEL")
        if not llm_model:
            print("⚠️ Advertencia: LLM_MODEL no encontrado en las variables de entorno, usando 'gpt-4o-mini' como default.")
            llm_model = "gpt-4o-mini"

        response = client.chat.completions.create(
            model=llm_model,
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=5,
        )
        
        if response.choices and response.choices[0].message.content:
            print("✅ Conexión e invocación simple al LLM exitosa.")
            return True
        else:
            raise Exception("La respuesta del LLM no fue la esperada.")

    except Exception as e:
        print(f"❌ ERROR: Falló la prueba del LLM. {e}")
        return False

if __name__ == "__main__":
    if not main():
        sys.exit(1)