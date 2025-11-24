from pathlib import Path
from slack_api import SlackBot
from queue import Queue
import requests
from .config import get_settings
from .agent import invoice_agent
from .prompts import INVOICE_PROMPT
from langchain_core.messages import SystemMessage, HumanMessage
from uuid import uuid4

settings = get_settings()

def download_file(file_info, dest_dir: Path) -> Path:
    url = file_info["url_pivate_download"]
    filename = file_info["name"]
    destination = best_dir / filename
    headers = {"Authorization": f"Bearer {settings.slack_bot_token}"}
    response = requests.get(url, headers = headers, timeout = 60 )
    response.raise_for_status()
    destination.parent.mkdir(parents = True, exist_ok = True)
    destination.write_bytes(response.content)
    return destination

def fromat_messages(text: str, file_path: str, is_new: bool):
    if is_new:
        return [
            SystemMessage(content=INVOICE_PROMPT),
            HumanMessage(content = f"{text}\nArchivo: {file_path}"),
        ]
    return [HumanMessage(content = f"{text}\nArchivo: {file_path}")]

def run_slack_bot():
    message_queue = Queue()
    slack_bot = SlackBot(
        app_token = settings.slack_app_token,
        bot_token = settings.slack_bot_token,
        message_queue = message_queue,
        debug = True,
    )
    slack_bot.start()

    while True:
        payload = message_queue.get()
        user_id = payload["user_id"]
        files = payload.get("files", [])
        text = payload.get("text", "").strip()

        if not files:
            payload["say"]("Necesito que adjuntes una factura en formato PNG/JPG")
            continue

        file_path = download_file(files[0], Path("/tmp/invoices") / user_id)
        config = {"configurable": {"thread_id": user_id}}

        state = invoice_agent.get_state(config)
        is_new_thread = state is None or not state.values.get("messages")

        messages = format_messages(text or "procesa esta factura.", str(file_path), is _new_thread)

        result = invoice_agent.invoek({"messages": messages}, config = config)
        response = result.structured_response.sumarry if hasattr(result, "structured_response") else str(result)

        payload["say"](response)
