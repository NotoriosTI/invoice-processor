import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Capa auxiliar para generar y cachear embeddings de productos."""

    def __init__(
        self,
        model_name: str,
        api_key: str,
        cache_path: Path,
        embeddings_client: Optional[Embeddings] = None,
    ):
        self._client = embeddings_client or OpenAIEmbeddings(
            model=model_name,
            openai_api_key=api_key,
        )
        self._cache_path = Path(cache_path)
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)

    def load_cache(self, path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
        cache_path = Path(path) if path else self._cache_path
        if not cache_path.exists():
            return None
        try:
            with cache_path.open("rb") as f:
                data = pickle.load(f)
            if not data or "ids" not in data or "vectors" not in data:
                return None
            return data
        except Exception as exc:
            logger.warning("No se pudo cargar la caché de embeddings: %s", exc)
            return None

    def save_cache(
        self,
        product_ids: List[int],
        vectors: Sequence[Sequence[float]],
        path: Optional[Path] = None,
    ) -> None:
        cache_path = Path(path) if path else self._cache_path
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with cache_path.open("wb") as f:
                payload = {
                    "ids": product_ids,
                    "vectors": [list(map(float, vec)) for vec in vectors],
                }
                pickle.dump(payload, f)
        except Exception as exc:
            logger.warning("No se pudo guardar la caché de embeddings: %s", exc)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._client.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._client.embed_query(text)

    @staticmethod
    def _cosine_similarities(vectors: np.ndarray, query_vector: np.ndarray) -> np.ndarray:
        denom = np.linalg.norm(vectors, axis=1) * np.linalg.norm(query_vector)
        denom = np.where(denom == 0, 1e-12, denom)
        return vectors @ query_vector / denom

    def find_best_match(
        self,
        query_text: str,
        product_ids: List[int],
        vectors: Sequence[Sequence[float]],
    ) -> Optional[Tuple[int, float]]:
        if not query_text or not product_ids or not vectors:
            return None
        try:
            matrix = np.array(vectors, dtype=float)
            query_vec = np.array(self.embed_query(query_text), dtype=float)
            if matrix.ndim != 2 or matrix.shape[0] != len(product_ids):
                logger.warning("Dimensiones de embeddings inconsistentes para la búsqueda.")
                return None
            sims = self._cosine_similarities(matrix, query_vec)
            best_idx = int(np.argmax(sims))
            return product_ids[best_idx], float(sims[best_idx])
        except Exception as exc:
            logger.warning("Error calculando similitud por embeddings: %s", exc)
            return None


def choose_best_match_with_fallback(
    service: EmbeddingService,
    query_text: str,
    caches: List[Optional[Dict[str, Any]]],
    min_score: float,
) -> Optional[Tuple[int, float]]:
    """Intenta caches en orden y devuelve el primer match que alcance min_score."""
    for cache in caches:
        if not cache:
            continue
        ids = cache.get("ids") or []
        vectors = cache.get("vectors") or []
        match = service.find_best_match(query_text, ids, vectors)
        if match and match[1] >= min_score:
            return match
    return None
