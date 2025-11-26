from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from .config import get_settings
from .models import InvoiceData, InvoiceResponseModel
from .tools import parse_invoice_image, process_invoice_purchase_flow, request_human_input

tools = [parse_invoice_image, process_invoice_purchase_flow, request_human_input]
checkpointer = MemorySaver()
reader_tools = [parse_invoice_image]
reader_checkpointer = MemorySaver()

def _get_llm():
    settings = get_settings()
    return ChatOpenAI(
        model = settings.openai_model,
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
