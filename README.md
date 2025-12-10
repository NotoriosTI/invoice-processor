# invoice-processor

Automated assistant (LangGraph + OpenAI) that reads purchase invoices (OCR), matches them against Odoo, adjusts order lines, and optionally closes the flow (receipts + invoice) with minimal human intervention. Provides Slack and console interfaces.

## Features
- OCR invoice reader (OpenAI Vision) → extracts NETO, IVA 19%, TOTAL and line items (cantidad, detalle, precio unitario, subtotal).
- Odoo integration: busca/crea orden de compra, ajusta cantidades/precios, recalcula totales, confirma recepción y crea factura cuando todo coincide.
- Slack and console modes for interactive runs; reader mode to just parse a single image.
- LangGraph ReAct agent with structured responses (`InvoiceResponseModel`).

## Prerequisites
- Python >= 3.13, < 3.14
- Poetry
- Odoo instance accessible from this app
- OpenAI API key
- Slack app/bot tokens (for Slack mode)

## Configuration
Variables are defined in `config/config_vars.yaml` and loaded via `env_manager`. You can set them as environment variables or in a `.env` file at the project root.

Required:
- `ODOO_TEST_URL`, `ODOO_TEST_DB`, `ODOO_TEST_USERNAME`, `ODOO_TEST_PASSWORD`
- `OPENAI_API_KEY`
- `SLACK_APP_TOKEN`, `SLACK_BOT_TOKEN`

Optional:
- `LLM_MODEL` (default `gpt-4o-mini`)
- `SLACK_DEBUG_LOGS` (default `false`)

Example `.env`:
```env
ODOO_TEST_URL=https://your-odoo
ODOO_TEST_DB=your_db
ODOO_TEST_USERNAME=user
ODOO_TEST_PASSWORD=pass
OPENAI_API_KEY=sk-...
SLACK_APP_TOKEN=xapp-...
SLACK_BOT_TOKEN=xoxb-...
LLM_MODEL=gpt-4o-mini
SLACK_DEBUG_LOGS=false
```

## Installation
```bash
poetry install
```

## Usage
Run the Slack bot:
```bash
poetry run python -m invoice_processor.app.main --mode slack
```

### Slack flow (summary)
1) User sends an image to the bot.
2) Bot downloads to `/tmp/invoices/<user_id>/`.
3) Agent runs OCR + Odoo comparison/adjustment.
4) Bot replies with the final summary (and an alert if manual review is needed).

## Notes
- Custom libraries from NotoriosTI (odoo-api, odoo-engine, slack-api, dev-utils, env-manager) are installed via git in `pyproject.toml`.
- OCR uses OpenAI Vision; ensure the API key has access to the selected model.
- The app may automatically update order lines in Odoo; review access/permissions accordingly.

## License
MIT (default). Review `pyproject.toml` or repository terms if different.

## Disclaimer
Invoice-processor is currently under active development
