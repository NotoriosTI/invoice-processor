import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from invoice_processor.core.models import ProductCandidate  # noqa: E402


def test_product_candidate_accepts_default_code():
    cand = ProductCandidate(id=1, name="Producto", score=0.9, default_code="SKU123")
    assert cand.default_code == "SKU123"
    assert cand.id == 1
