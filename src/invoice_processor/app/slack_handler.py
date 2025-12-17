from pathlib import Path
from queue import Queue
import logging
import requests
from langchain_core.messages import SystemMessage, HumanMessage
from rich.logging import RichHandler
from rich.traceback import install

from .slack_bot import SlackBot
from ..config import get_settings, load_allowed_users
from ..prompts import INVOICE_PROMPT
from ..agents.agent import invoice_agent
from ..core.models import InvoiceResponseModel

settings = get_settings()
logger = logging.getLogger(__name__)
SLACK_POST_MESSAGE_URL = "https://slack.com/api/chat.postMessage"
SLACK_REACTION_URL = "https://slack.com/api/reactions.add"
SLACK_HISTORY_URL = "https://slack.com/api/conversations.history"


def configure_logging() -> None:
    level = logging.DEBUG if settings.slack_debug_logs else logging.INFO
    install()
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[RichHandler(rich_tracebacks=True, markup=True)],
        force=True,
    )
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
    allowed = load_allowed_users()
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
        user_id = payload.get("user") or payload.get("user_id")
        event = payload.get("event") or payload  # algunos adaptadores entregan el evento anidado
        inner_message = event.get("message") or {}
        previous_message = inner_message.get("previous_message", {}) or event.get("previous_message", {}) or {}

        # Filtro de usuarios autorizados
        if allowed.get("enabled"):
            allowed_ids = {u["id"] if isinstance(u, dict) else u for u in allowed.get("users", [])}
            if user_id not in allowed_ids:
                logger.info("Usuario no autorizado: %s; se ignora el evento", user_id)
                continue

        # Datos base del evento
        user_id = event.get("user") or event.get("user_id") or inner_message.get("user") or previous_message.get("user")
        bot_id = event.get("bot_id") or inner_message.get("bot_id")
        subtype = event.get("subtype")
        hidden = event.get("hidden")
        inner_subtype = inner_message.get("subtype")
        inner_hidden = inner_message.get("hidden")
        files = event.get("files", [])
        # Algunos eventos (message_changed, file_share) traen los archivos anidados en message o previous_message.
        if not files and inner_message.get("files"):
            files = inner_message.get("files")
        if not files and inner_message.get("previous_message", {}).get("files"):
            files = inner_message.get("previous_message", {}).get("files")
        channel = event.get("channel") or inner_message.get("channel")
        timestamp = event.get("timestamp") or event.get("ts") or inner_message.get("ts") or previous_message.get("ts")
        incoming_text = (event.get("text") or inner_message.get("text") or previous_message.get("text") or "").strip()
        logger.info(
            "Evento recibido de Slack user=%s channel=%s ts=%s adjuntos=%d texto=%s",
            user_id,
            channel,
            timestamp,
            len(files),
            "sí" if incoming_text else "no",
        )

        # Ignora mensajes generados por el propio bot para evitar loops.
        if bot_id or (subtype and subtype == "bot_message") or (user_id and user_id.startswith("B")):
            logger.debug("Evento ignorado por ser del bot (bot_id=%s, subtype=%s, user=%s).", bot_id, subtype, user_id)
            continue

        # Solo procesamos mensajes de usuario con subtipos permitidos.
        allowed_subtypes = {None, "", "file_share"}
        sys_subtypes = {"message_changed", "message_deleted", "tombstone"}
        current_subtype = inner_subtype or subtype
        previous_subtype = previous_message.get("subtype")
        if (
            hidden
            or inner_hidden
            or current_subtype in sys_subtypes
            or previous_subtype in sys_subtypes
            or user_id == "USLACKBOT"
            or current_subtype not in allowed_subtypes
        ):
            logger.debug(
                "Evento ignorado por subtipo no permitido (hidden=%s/%s, subtype=%s, inner_subtype=%s, prev_subtype=%s, user=%s, files=%d).",
                hidden,
                inner_hidden,
                subtype,
                inner_subtype,
                previous_subtype,
                user_id,
                len(files),
            )
            continue

        if not files:
            # Intentar recuperar archivos del mensaje original (mensajes de Slack que llegan sin files en el evento).
            fetched_files = _fetch_files_from_slack(channel, timestamp)
            if fetched_files:
                files = fetched_files

        if not files:
            if not incoming_text:
                logger.debug("Evento sin adjuntos ni texto; se ignora.")
                continue
            logger.info("Evento sin adjuntos; solicitando nueva factura al usuario %s", user_id)
            post_message(channel, "Necesito que adjuntes una factura en formato PNG/JPG.", timestamp)
            continue

        # Valida que el archivo principal sea imagen; si no, solicita PNG/JPG.
        file_info = files[0]
        mimetype = (file_info.get("mimetype") or file_info.get("filetype") or "").lower()
        if "image" not in mimetype:
            logger.info("Adjunto ignorado por no ser imagen (mimetype=%s).", mimetype)
            post_message(channel, "Necesito que adjuntes una factura en formato PNG/JPG.", timestamp)
            continue

        try:
            file_path = download_file(file_info, Path("/tmp/invoices") / user_id)
        except Exception as exc:
            logger.exception("No se pudo descargar el archivo de Slack")
            post_message(channel, f"No pude descargar la factura: {exc}", timestamp)
            continue
        config = {"configurable": {"thread_id": user_id}}

        state = invoice_agent.get_state(config)
        is_new_thread = state is None or not state.values.get("messages")
        # Si no hay texto, usar un mensaje por defecto
        if not incoming_text:
            incoming_text = "Procesa esta factura."
        messages = format_messages(incoming_text, str(file_path), is_new_thread)

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
            try:
                add_reaction(channel, timestamp, "x")
            except Exception:
                logger.debug("No se pudo agregar reacción de error al mensaje original.")
            continue
        # Prefiere la respuesta estructurada del modelo si existe; evita la conversación interna.
        response_model: InvoiceResponseModel | None = _coerce_response_model(result)
        if response_model:
            response = response_model.summary
            needs_follow_up = bool(response_model.needs_follow_up)
        else:
            # Evita enviar conversación interna u objetos desconocidos.
            response = "No se pudo procesar la factura; revisa Odoo manualmente."
            needs_follow_up = True

        # Reacciona en el mensaje original: verde si todo OK y se cerró flujo; rojo + mensaje si hay follow-up.
        if not needs_follow_up:
            try:
                add_reaction(channel, timestamp, "white_check_mark")
            except Exception:
                logger.debug("No se pudo agregar reacción de éxito al mensaje original.")
            # Enviar también el resumen de lo que se hizo.
            try:
                post_message(channel, response, timestamp)
            except Exception as exc:
                logger.error("No se pudo enviar mensaje a Slack: %s", exc)
                logger.debug("Respuesta que falló: %s", response)
            continue
        else:
            try:
                add_reaction(channel, timestamp, "x")
            except Exception:
                logger.debug("No se pudo agregar reacción de error al mensaje original.")
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


def add_reaction(channel: str | None, timestamp: str | None, emoji: str) -> None:
    """Agrega una reacción al mensaje original del usuario."""
    if not channel or not timestamp:
        logger.debug("No se puede agregar reacción: faltan channel o timestamp.")
        return
    payload = {"channel": channel, "timestamp": timestamp, "name": emoji}
    headers = {
        "Authorization": f"Bearer {settings.slack_bot_token}",
        "Content-Type": "application/json;charset=utf-8",
    }
    resp = requests.post(SLACK_REACTION_URL, headers=headers, json=payload, timeout=10)
    data = resp.json() if resp.content else {}
    if not resp.ok or not data.get("ok"):
        raise RuntimeError(
            f"Slack reaction error: {data.get('error') if isinstance(data, dict) else resp.text}"
        )
def _fetch_files_from_slack(channel: str | None, ts: str | None):
    """Obtiene los archivos del mensaje original usando la Web API si no vienen en el evento."""
    if not channel or not ts:
        return []
    try:
        params = {
            "channel": channel,
            "latest": ts,
            "inclusive": "true",
            "limit": 1,
        }
        headers = {"Authorization": f"Bearer {settings.slack_bot_token}"}
        resp = requests.get(SLACK_HISTORY_URL, headers=headers, params=params, timeout=10)
        data = resp.json() if resp.content else {}
        messages = data.get("messages") or []
        if not messages:
            return []
        files = messages[0].get("files") or []
        return files
    except Exception as exc:
        logger.warning("No se pudieron recuperar archivos desde Slack: %s", exc)
        return []
def _coerce_response_model(result) -> InvoiceResponseModel | None:
    """Intenta extraer InvoiceResponseModel desde distintos formatos de resultado."""
    try:
        if isinstance(result, InvoiceResponseModel):
            return result

        # Caso atributo structured_response
        structured = getattr(result, "structured_response", None)
        if structured:
            if isinstance(structured, InvoiceResponseModel):
                return structured
            try:
                return InvoiceResponseModel.model_validate(structured)
            except Exception:
                pass

        # Caso dict con structured_response o campos del modelo
        if isinstance(result, dict):
            if "structured_response" in result:
                try:
                    return InvoiceResponseModel.model_validate(result["structured_response"])
                except Exception:
                    pass
            try:
                return InvoiceResponseModel.model_validate(result)
            except Exception:
                pass

        # Caso atributo dict interno
        if hasattr(result, "__dict__"):
            data = result.__dict__
            if "structured_response" in data:
                try:
                    return InvoiceResponseModel.model_validate(data["structured_response"])
                except Exception:
                    pass
            try:
                return InvoiceResponseModel.model_validate(data)
            except Exception:
                pass
    except Exception:
        return None
    return None
