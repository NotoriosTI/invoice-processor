from pathlib import Path
from queue import Queue
import requests
from langchain_core.messages import SystemMessage, HumanMessage

from slack_api import SlackBot
from .config import get_settings
from .prompts import INVOICE_PROMPT
from .agent import invoice_agent

settings = get_settings()


def download_file(file_info, dest_dir: Path) -> Path:
    url = file_info["url_private_download"]
    filename = file_info["name"]
    destination = dest_dir / filename
    headers = {"Authorization": f"Bearer {settings.slack_bot_token}"}
    response = requests.get(url, headers=headers, timeout=60)
    response.raise_for_status()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(response.content)
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
    message_queue = Queue()
    slack_bot = SlackBot(
        app_token=settings.slack_app_token,
        bot_token=settings.slack_bot_token,
        message_queue=message_queue,
        debug=False,
    )
    slack_bot.start()

    while True:
        payload = message_queue.get()
        user_id = payload["user"]
        files = payload.get("files", [])
        say = payload["say"]

        if not files:
            say("Necesito que adjuntes una factura en formato PNG/JPG.")
            continue

        file_path = download_file(files[0], Path("/tmp/invoices") / user_id)
        config = {"configurable": {"thread_id": user_id}}

        state = invoice_agent.get_state(config)
        is_new_thread = state is None or not state.values.get("messages")
        messages = format_messages(payload.get("text", ""), str(file_path), is_new_thread)

        result = invoice_agent.invoke({"messages": messages}, config=config)
        response = result.summary if hasattr(result, "summary") else str(result)

        if getattr(result, "needs_follow_up", False):
            response += "\n⚠️ Se detectaron discrepancias. Revisa los detalles anteriores."

        say(response)
