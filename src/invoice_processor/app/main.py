import argparse
from uuid import uuid4
from env_manager import get_config
from langchain_core.messages import SystemMessage, HumanMessage
from ..agents.agent import invoice_agent, invoice_reader_agent
from ..prompts.prompts import INVOICE_PROMPT, INVOICE_READER_PROMPT
from .slack_handler import run_slack_bot
import logging
from rich.logging import RichHandler
from rich.traceback import install
from rich.console import Console
from pathlib import Path
from ..config import get_settings

install()

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True, markup=True)],
)
logging.getLogger("langchain").setLevel(logging.ERROR)
logging.getLogger("langgraph").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.WARNING)

console = Console()


def console_main():
    thread_id = f"console-{uuid4()}"
    config = {"configurable": {"thread_id": thread_id}}

    while True:
        text = input(">> ")
        state = invoice_agent.get_state(config)
        is_new = state is None or not state.values.get("messages")

        messages = (
            [SystemMessage(content=INVOICE_PROMPT), HumanMessage(content=text)]
            if is_new
            else [HumanMessage(content=text)]
        )

        result = invoice_agent.invoke({"messages": messages}, config=config)
        if hasattr(result, "structured_response"):
            console.print(result.structured_response.summary)
        else:
            console.print(result)


def reader_main(image_path: str):
    """Ejecuta el agente de lectura simple sobre una única imagen."""
    thread_id = f"reader-{uuid4()}"
    config = {"configurable": {"thread_id": thread_id}}
    message_text = (
        "Lee la factura ubicada en '{path}'. "
        "Llama a parse_invoice_image(image_path='{path}') y devuelve los datos extraídos."
    ).format(path=image_path)
    messages = [
        SystemMessage(content=INVOICE_READER_PROMPT),
        HumanMessage(content=message_text),
    ]
    result = invoice_reader_agent.invoke({"messages": messages}, config=config)
    if hasattr(result, "model_dump_json"):
        console.print(result.model_dump_json(indent=2))
    else:
        console.print(result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=["console", "slack", "reader"], default="console"
    )
    parser.add_argument("--image", help="Ruta de la factura a leer (solo modo reader)")
    args = parser.parse_args()

    if args.mode == "slack":
        run_slack_bot()
    elif args.mode == "reader":
        settings = get_settings()
        path = args.image or settings.default_invoice_path
        reader_main(path)
    else:
        console_main()


if __name__ == "__main__":
    main()
