import sqlite3

from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_openai import ChatOpenAI
from ..config import get_settings, DATA_PATH
from ..core.models import InvoiceData, InvoiceResponseModel
from ..tools.tools import parse_invoice_image, process_invoice_purchase_flow, map_product_decision_tool
from ..tools.odoo_tools import (
    split_purchase_line,
    update_line_quantity,
    finalize_invoice_workflow,
    receive_order_by_sku_prefix,
)

tools = [
    parse_invoice_image,
    process_invoice_purchase_flow,
    map_product_decision_tool,
    split_purchase_line,
    update_line_quantity,
    finalize_invoice_workflow,
    receive_order_by_sku_prefix,
]

_db_path = DATA_PATH / "checkpoints.sqlite3"
_db_path.parent.mkdir(parents=True, exist_ok=True)
_conn = sqlite3.connect(str(_db_path), check_same_thread=False)
checkpointer = SqliteSaver(_conn)
checkpointer.setup()

reader_tools = [parse_invoice_image]
_reader_db_path = DATA_PATH / "reader_checkpoints.sqlite3"
_reader_conn = sqlite3.connect(str(_reader_db_path), check_same_thread=False)
reader_checkpointer = SqliteSaver(_reader_conn)
reader_checkpointer.setup()

def _get_llm():
    settings = get_settings()
    return ChatOpenAI(
        model = settings.llm_model,
        openai_api_key = settings.openai_api_key,
        max_tokens = 2048,
    )

invoice_agent = create_react_agent(
    _get_llm(),
    tools = tools,
    response_format = InvoiceResponseModel,
    checkpointer = checkpointer,
)

invoice_reader_agent = create_react_agent(
    _get_llm(),
    tools = reader_tools,
    response_format = InvoiceData,
    checkpointer = reader_checkpointer,
)
