from pathlib import Path
from queue import Queue
import logging
import requests
from langchain_core.messages import SystemMessage, HumanMessage

from .slack_bot import SlackBot
from ..config import get_settings
from ..prompts.prompts import INVOICE_PROMPT
from ..agents.agent import invoice_agent

settings = get_settings()
logger = logging.getLogger(__name__)
SLACK_POST_MESSAGE_URL = "https://slack.com/api/chat.postMessage"


def configure_logging() -> None:
    level = logging.DEBUG if settings.slack_debug_logs else logging.INFO
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
        root_logger.addHandler(handler)
    root_logger.setLevel(level)
    logger.setLevel(level)
    for noisy_logger in ("slack_bolt", "slack_sdk"):
        logging.getLogger(noisy_logger).setLevel(level)


def download_file(file_info, dest_dir: Path) -> Path:
    url = file_info["url_private_download"]
    filename = file_info["name"]
    destination = dest_dir / filename
    headers = {"Authorization": f"Bearer {settings.slack_bot_token}"}
    logger.info("Descargando archivo de Slack: %s -> %s", filename, destination)
    response = requests.get(url, headers=headers, timeout=60)
    response.raise_for_status()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(response.content)
    logger.info("Archivo descargado (%d bytes)", len(response.content))
    return destination


def format_messages(text: str, file_path: str, is_new: bool):
    message_text = text.strip() or "Procesa esta factura."
    content = f"{message_text}\nArchivo: {file_path}"

    if is_new:
        return [
            SystemMessage(content=INVOICE_PROMPT),
            HumanMessage(content=content),
        ]
    return [HumanMessage(content=content)]


def run_slack_bot():
    configure_logging()
    message_queue = Queue()
    slack_bot = SlackBot(
        app_token=settings.slack_app_token,
        bot_token=settings.slack_bot_token,
        message_queue=message_queue,
        debug=bool(settings.slack_debug_logs),
    )
    logger.info("Iniciando Slack bot (debug=%s)", settings.slack_debug_logs)
    slack_bot.start()

    while True:
        payload = message_queue.get()
        user_id = payload["user"]
        files = payload.get("files", [])
        channel = payload.get("channel")
        timestamp = payload.get("timestamp")
        logger.info(
            "Evento recibido de Slack user=%s channel=%s ts=%s adjuntos=%d texto=%s",
            user_id,
            channel,
            timestamp,
            len(files),
            "sí" if (payload.get("text") or "").strip() else "no",
        )

        if not files:
            logger.info("Evento sin adjuntos; solicitando nueva factura al usuario %s", user_id)
            post_message(channel, "Necesito que adjuntes una factura en formato PNG/JPG.", timestamp)
            continue

        try:
            file_path = download_file(files[0], Path("/tmp/invoices") / user_id)
        except Exception as exc:
            logger.exception("No se pudo descargar el archivo de Slack")
            post_message(channel, f"No pude descargar la factura: {exc}", timestamp)
            continue
        config = {"configurable": {"thread_id": user_id}}

        state = invoice_agent.get_state(config)
        is_new_thread = state is None or not state.values.get("messages")
        messages = format_messages(payload.get("text", ""), str(file_path), is_new_thread)

        try:
            logger.info("Procesando factura %s con el agente", file_path)
            result = invoice_agent.invoke({"messages": messages}, config=config)
        except Exception as exc:
            logger.exception("El agente falló al procesar la factura")
            post_message(
                channel,
                "Ocurrió un error al procesar la factura. Intenta nuevamente más tarde.",
                timestamp,
            )
            continue
        response = result.summary if hasattr(result, "summary") else str(result)

        if getattr(result, "needs_follow_up", False):
            response += "\n⚠️ Se detectaron discrepancias. Revisa los detalles anteriores."

        try:
            post_message(channel, response, timestamp)
        except Exception as exc:
            logger.error("No se pudo enviar mensaje a Slack: %s", exc)
            logger.debug("Respuesta que falló: %s", response)


def post_message(channel: str | None, text: str, thread_ts: str | None = None) -> None:
    if not channel:
        logger.warning("No se puede enviar mensaje a Slack: canal desconocido.")
        return
    payload = {"channel": channel, "text": text}
    if thread_ts:
        payload["thread_ts"] = thread_ts
    headers = {
        "Authorization": f"Bearer {settings.slack_bot_token}",
        "Content-Type": "application/json;charset=utf-8",
    }
    resp = requests.post(SLACK_POST_MESSAGE_URL, headers=headers, json=payload, timeout=15)
    data = resp.json() if resp.content else {}
    if not resp.ok or not data.get("ok"):
        raise RuntimeError(
            f"Slack API error: {data.get('error') if isinstance(data, dict) else resp.text}"
        )
