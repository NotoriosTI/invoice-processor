# invoice-processor

Asistente automatizado (LangGraph + OpenAI) que lee facturas de compra (OCR), las compara contra Odoo, ajusta lineas de orden, y opcionalmente cierra el flujo completo (recepcion + factura) con minima intervencion humana. Interfaces via Slack y consola.

## Features

- **OCR con OpenAI Vision**: Extrae proveedor, RUT, folio, fecha de emision, descuento global, descuentos por linea, NETO, IVA 19%, TOTAL y lineas de detalle (cantidad, unidad, detalle, precio unitario, subtotal, SKU/default_code).
- **Integracion Odoo via XML-RPC**: Busca/crea orden de compra, ajusta cantidades y precios, recalcula totales, confirma recepcion con ruteo por prefijo de SKU (MP/ME), y crea factura.
- **Busqueda de productos por embeddings**: Similitud coseno con `text-embedding-3-small` para fuzzy matching de productos, con cache en disco y TTL configurable.
- **Mapeo de productos persistente**: Registra decisiones en `product.supplierinfo` de Odoo para resolver automaticamente en futuras facturas.
- **Retry con backoff exponencial**: Reintentos automaticos en llamadas XML-RPC transitorias (3 intentos, delays 1s/2s/4s).
- **Tres modos de ejecucion**: `slack` (produccion), `console` (desarrollo), `reader` (solo OCR).
- **Agente ReAct con LangGraph**: Respuestas estructuradas (`InvoiceResponseModel`), checkpointing SQLite para persistencia de estado entre turnos.
- **Flujo interactivo de dos fases**: Lectura (comparacion sin escritura) y escritura (creacion de OC + recepcion + factura) con aprobacion humana intermedia.

## Architecture

```
Slack User  -->  SlackBot (slack-api)  -->  Queue  -->  slack_handler loop
                                                              |
                                                 LangGraph ReAct Agent
                                                   (invoice_agent)
                                                         |
                                 +----------+----------+----------+----------+
                                 |          |          |          |          |
                           parse_invoice  process_  map_product  split_   receive/
                           _image        _purchase  _decision   purchase  finalize
                                 |        _flow      _tool      _line     _workflow
                                 |          |          |
                           OCR Client   Processor   OdooConnectionManager
                         (OpenAI Vision)    |       (XML-RPC + odoo-api)
                                            |
                                       Odoo Server
```

## Flow

1. **Entrada**: El usuario envia una imagen de factura al bot de Slack (o usa el modo consola/reader).
2. **OCR**: `parse_invoice_image` envia la imagen a OpenAI Vision con un prompt estructurado. Post-procesa el JSON: normaliza numeros CLP, valida fechas, verifica coherencia de subtotales. Si el OCR es dudoso, reintenta con una version recortada de la imagen.
3. **Comparacion (modo lectura)**: `process_invoice_purchase_flow(allow_odoo_write=false)` compara la factura contra Odoo sin escribir:
   - Resuelve proveedor por nombre/RUT (formato canonico con guion).
   - Para cada linea, busca mapeo existente en `product.supplierinfo`; si no existe, usa busqueda por embeddings.
   - Retorna `WAITING_FOR_HUMAN` si hay productos sin mapeo confirmado, o `WAITING_FOR_APPROVAL` si todo coincide.
4. **Confirmacion humana**: El bot muestra la lista de productos (Factura vs Odoo + SKU). El usuario puede:
   - Responder "Afirmativo" para confirmar toda la lista.
   - Corregir: "Cambiar {Producto Factura} por {SKU o nombre correcto}".
5. **Mapeo**: `map_product_decision_tool` registra la correccion en `product.supplierinfo`. Se reintenta la comparacion.
6. **Escritura**: Con aprobacion, `process_invoice_purchase_flow(allow_odoo_write=true)` ejecuta:
   - Crea/actualiza la Orden de Compra (PO).
   - Ajusta cantidades y precios de lineas para que coincidan con la factura.
   - Confirma la PO, recepciona (picking con ruteo por prefijo SKU), y crea/publica la factura.
7. **Respuesta**: El bot envia el resumen final y agrega una reaccion verde si el flujo se completo.

## Prerequisites

- Python >= 3.13, < 3.14
- Poetry
- Instancia Odoo accesible desde la aplicacion
- OpenAI API key (con acceso al modelo Vision configurado)
- Slack app/bot tokens (para modo Slack)

## Configuration

Las variables se definen en `config/config_vars.yaml` y se cargan via `env_manager`. Se pueden setear como variables de entorno o en un archivo `.env` en la raiz del proyecto.

### Required

| Variable | Descripcion |
|---|---|
| `ODOO_PROD_URL` | URL de la instancia Odoo |
| `ODOO_PROD_DB` | Base de datos Odoo |
| `ODOO_PROD_USERNAME` | Usuario Odoo |
| `ODOO_PROD_PASSWORD` | Password Odoo |
| `OPENAI_API_KEY` | API key de OpenAI |
| `SLACK_APP_TOKEN` | Token de app Slack (xapp-...) |
| `SLACK_BOT_TOKEN` | Token de bot Slack (xoxb-...) |

### Optional

| Variable | Default | Descripcion |
|---|---|---|
| `LLM_MODEL` | `gpt-5.2` | Modelo de OpenAI para OCR y agente |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Modelo de embeddings para busqueda de productos |
| `SLACK_DEBUG_LOGS` | `false` | Habilitar logs de debug para Slack |
| `DEFAULT_PURCHASE_TAX_IDS` | `""` | IDs de impuestos de compra por defecto (separados por coma) |
| `XMLRPC_TIMEOUT_SECONDS` | `30` | Timeout en segundos para llamadas XML-RPC a Odoo |
| `EMBEDDING_CACHE_TTL_HOURS` | `24` | TTL en horas para la cache de embeddings de productos |
| `ODOO_STOCK_LOCATION_MP_ME` | `JS/Stock/Materia Prima y Envases` | Ubicacion destino para SKUs con prefijo MP/ME |
| `ODOO_STOCK_LOCATION_DEFAULT` | `JS/Stock` | Ubicacion destino por defecto |
| `LANGSMITH_TRACING` | `false` | Habilitar tracing con LangSmith |
| `LANGSMITH_ENDPOINT` | `""` | Endpoint de LangSmith |
| `LANGSMITH_API_KEY` | `""` | API key de LangSmith |
| `LANGSMITH_PROJECT` | `""` | Nombre del proyecto en LangSmith |

### Example `.env`

```env
ODOO_PROD_URL=https://your-odoo
ODOO_PROD_DB=your_db
ODOO_PROD_USERNAME=user
ODOO_PROD_PASSWORD=pass
OPENAI_API_KEY=sk-...
SLACK_APP_TOKEN=xapp-...
SLACK_BOT_TOKEN=xoxb-...
LLM_MODEL=gpt-5.2
EMBEDDING_MODEL=text-embedding-3-small
SLACK_DEBUG_LOGS=false
XMLRPC_TIMEOUT_SECONDS=30
EMBEDDING_CACHE_TTL_HOURS=24
```

## Installation

```bash
poetry install
```

## Usage

### Slack mode (production)

```bash
poetry run python -m invoice_processor.app.main --mode slack
```

### Console mode (development)

```bash
poetry run python -m invoice_processor.app.main --mode console
```

### Reader mode (solo OCR)

```bash
poetry run python -m invoice_processor.app.main --mode reader --image /path/to/factura.jpg
```

## Project Structure

```
src/invoice_processor/
  app/
    main.py                 # Entry point (3 modos)
    slack_handler.py        # Loop principal Slack, descarga, routing
    slack_bot.py            # Re-export de slack-api
  agents/
    agent.py                # LangGraph ReAct agents + SQLite checkpointers
  config/
    config.py               # Settings, env loading, allowed users
  core/
    models.py               # Pydantic models (InvoiceData, InvoiceResponseModel, etc.)
    processor.py            # Orquestacion: OCR -> comparacion -> escritura Odoo
  infrastructure/services/
    ocr.py                  # OpenAI Vision OCR client
    odoo_connection_manager.py  # Singleton XML-RPC manager (~2700 lineas)
    embedding_service.py    # Embeddings + cosine similarity
  prompts/
    invoice_prompt.py       # System prompt del agente principal
    invoice_reader_prompt.py # System prompt del agente reader
    invoice_ocr_prompt.py   # Prompt para OpenAI Vision (schema JSON)
  tools/
    tools.py                # Tools del agente (parse, process, map)
    odoo_tools.py           # Tools Odoo (split, update, finalize, receive)
config/
  config_vars.yaml          # Definicion de variables de configuracion
  allowed_users_config.yaml # Control de acceso Slack
data/
  checkpoints.sqlite3       # Estado del agente (LangGraph)
  thread_status.json        # Estado de threads Slack
  product_embeddings.pkl    # Cache de embeddings
```

## Testing

```bash
poetry run pytest tests/
```

Tests cubren: OCR post-processing, persistencia de estado, deteccion de loops, mapeo de productos, validaciones financieras, y flujos e2e con mocks.

## Notes

- Librerias custom de NotoriosTI (`odoo-api`, `odoo-engine`, `slack-api`, `dev-utils`, `env-manager`) se instalan via git en `pyproject.toml`.
- El OCR usa OpenAI Vision; asegurar que la API key tenga acceso al modelo configurado.
- La app puede crear/modificar ordenes de compra, recepciones y facturas en Odoo; revisar permisos del usuario Odoo.
- El formato de RUT chileno se normaliza automaticamente (ej: `12345678-9`).
- Los embeddings de productos se cachean en disco con TTL configurable; usar `invalidate_product_embeddings()` para forzar regeneracion.

## License

MIT (default). Revisar `pyproject.toml` o los terminos del repositorio si difiere.

## Disclaimer

Invoice-processor esta actualmente en desarrollo activo.
