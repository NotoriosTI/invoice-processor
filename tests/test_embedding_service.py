import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from invoice_processor.infrastructure.services.embedding_service import (  # noqa: E402
    EmbeddingService,
    choose_best_match_with_fallback,
)


class FakeEmbeddings:
    def embed_documents(self, texts):
        return [[1.0, 0.0], [0.0, 1.0]][: len(texts)]

    def embed_query(self, text):
        lower = (text or "").lower()
        if "banana" in lower:
            return [0.1, 0.9]
        return [0.9, 0.1]


def test_find_best_match_prefers_highest_similarity(tmp_path):
    service = EmbeddingService(
        model_name="fake",
        api_key="fake",
        cache_path=tmp_path / "cache.pkl",
        embeddings_client=FakeEmbeddings(),
    )
    ids = [11, 22]
    vectors = [[1.0, 0.0], [0.0, 1.0]]

    best = service.find_best_match("apple", ids, vectors)

    assert best is not None
    best_id, score = best
    assert best_id == 11
    assert score > 0.9


def test_cache_roundtrip(tmp_path):
    cache_file = tmp_path / "cache.pkl"
    service = EmbeddingService(
        model_name="fake",
        api_key="fake",
        cache_path=cache_file,
        embeddings_client=FakeEmbeddings(),
    )
    ids = [1, 2]
    vectors = [[0.1, 0.2], [0.3, 0.4]]

    service.save_cache(ids, vectors)
    cached = service.load_cache()

    assert cached is not None
    assert cached["ids"] == ids
    assert cached["vectors"] == vectors


def test_choose_best_match_with_fallback_prefers_second_cache():
    service = EmbeddingService(
        model_name="fake",
        api_key="fake",
        cache_path=Path("/tmp/unused.pkl"),
        embeddings_client=FakeEmbeddings(),
    )
    cache_supplier = {"ids": [11], "vectors": [[1.0, 0.0]]}  # similitud ~0.11
    cache_global = {"ids": [22], "vectors": [[0.0, 1.0]]}  # similitud ~0.99

    match = choose_best_match_with_fallback(
        service,
        "banana",
        [cache_supplier, cache_global],
        min_score=0.5,
    )

    assert match is not None
    prod_id, score = match
    assert prod_id == 22
    assert score > 0.9
