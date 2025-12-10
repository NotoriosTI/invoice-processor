#!/bin/bash
set -e

echo "🚀 Iniciando despliegue y verificación local para invoice-processor..."

# Navegar al directorio raíz del proyecto
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
PROJECT_ROOT=$(dirname "$SCRIPT_DIR")
cd "$PROJECT_ROOT"

echo "📍 Directorio actual: $(pwd)"

# --- Paso 1: Construir la imagen sin caché ---
echo "🐳 Construyendo la imagen sin caché..."
docker-compose build --no-cache

echo "✅ Imagen Docker construida exitosamente."

# --- Paso 2: Levantar el contenedor en modo detached ---
echo "⬆️ Levantando contenedor con docker-compose en modo detached..."
docker-compose up -d

echo "✅ Contenedor 'invoice-processor' iniciado."
echo "💡 Para ver los logs de la aplicación: docker-compose logs -f"
echo "💡 Para detenerlo: docker-compose down"

# --- Paso 3: Esperar un poco para que el servicio inicie completamente ---
echo "⏳ Esperando 15 segundos para que el servicio principal (bot de Slack) inicie..."
sleep 15

# --- Paso 4: Ejecutar los health checks individuales dentro del contenedor ---
echo "🔍 Ejecutando health checks individuales dentro del contenedor 'invoice-processor'..."

TEST_RESULTS=()

# Test Odoo
echo "--- Ejecutando check_odoo.py ---"
if docker-compose exec invoice-processor python -m tests.deployment.check_odoo; then
    echo "✅ check_odoo.py PASSED"
    TEST_RESULTS+=("PASSED: check_odoo.py")
else
    echo "❌ check_odoo.py FAILED"
    TEST_RESULTS+=("FAILED: check_odoo.py")
fi

# Test LLM
echo "--- Ejecutando check_llm.py ---"
if docker-compose exec invoice-processor python -m tests.deployment.check_llm; then
    echo "✅ check_llm.py PASSED"
    TEST_RESULTS+=("PASSED: check_llm.py")
else
    echo "❌ check_llm.py FAILED"
    TEST_RESULTS+=("FAILED: check_llm.py")
fi

# Test GCP Secrets
echo "--- Ejecutando check_gcp_secrets.py ---"
if docker-compose exec invoice-processor python -m tests.deployment.check_gcp_secrets; then
    echo "✅ check_gcp_secrets.py PASSED"
    TEST_RESULTS+=("PASSED: check_gcp_secrets.py")
else
    echo "❌ check_gcp_secrets.py FAILED"
    TEST_RESULTS+=("FAILED: check_gcp_secrets.py")
fi

echo "--- Resumen de Health Checks ---"
for result in "${TEST_RESULTS[@]}"; do
    echo "$result"
done

# Verificar si todos los tests pasaron
if [[ " ${TEST_RESULTS[*]} " == *" FAILED: "* ]]; then
    echo "🔥 Uno o más health checks fallaron."
    exit 1
else
    echo "🎉 Todos los health checks pasaron exitosamente."
fi

echo "🎉 Verificación local completada."
echo "⬇️ Puedes detener el contenedor con 'docker-compose down' cuando termines."