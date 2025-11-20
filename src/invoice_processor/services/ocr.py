import base64
from pathlib import Path
from typing import Any, Dict, Union

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from ..config import get_settings
from ..prompts import INVOICE_OCR_PROMPT


class InvoiceOcrClient:
    def __init__(self):
        settings = get_settings()
        self.llm = ChatOpenAI(
            model=settings.openai_model,
            openai_api_key=settings.openai_api_key,
            max_tokens=2048,
        )

    @staticmethod
    def _encode_image(image_path: str) -> str:
        data = Path(image_path).read_bytes()
        return base64.b64encode(data).decode("utf-8")

    def extract(self, image_path: str) -> Union[str, Dict[str, Any]]:
        image_b64 = self._encode_image(image_path)
        message = HumanMessage(
            content=[
                {"type": "text", "text": INVOICE_OCR_PROMPT},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                },
            ]
        )
        response = self.llm.invoke([message])
        return response.content
