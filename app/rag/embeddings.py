import logging
import time
from typing import List

from app.core.config import settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    def __init__(self):
        self.api_key = settings.GEMINI_API_KEY
        self.model = settings.EMBEDDING_MODEL
        self._client = None
        self._local_ef = None

        if self._is_local():
            logger.info(
                "EmbeddingService: Configured to use local ONNX engine (all-MiniLM-L6-v2)"
            )
        elif self.api_key:
            try:
                from google import genai

                self._client = genai.Client(api_key=self.api_key)
                logger.info(
                    f"EmbeddingService: Using official Google GenAI ({self.model})"
                )
            except Exception:
                logger.warning("Failed to configure Google GenAI client")

    def _is_local(self) -> bool:
        return (self.model or "").strip().lower() == "local"

    def _get_local_ef(self):
        if getattr(self, "_local_ef", None) is None:
            import chromadb.utils.embedding_functions as ef

            self._local_ef = ef.DefaultEmbeddingFunction()
            logger.info(
                "EmbeddingService: Using local all-MiniLM-L6-v2 ONNX embedding engine."
            )
        return self._local_ef

    def embed_texts(self, texts: List[str], batch_size: int = 25) -> List[List[float]]:
        """
        Embed a list of text strings into vectors.
        Uses local ONNX model when EMBEDDING_MODEL=local, otherwise calls Gemini API.
        """
        if not texts:
            return []
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        if not self._is_local():
            if self.api_key and not self._client:
                raise RuntimeError("Gemini embedding client unavailable")
            if self.api_key and self._client:
                embeddings = []
                for i in range(0, len(texts), batch_size):
                    batch = texts[i : i + batch_size]
                    max_retries = 3
                    for attempt in range(max_retries):
                        try:
                            res = self._client.models.embed_content(
                                model=self.model, contents=batch
                            )
                            if not res.embeddings or len(res.embeddings) != len(batch):
                                raise RuntimeError("Incomplete embedding response")
                            embeddings.extend(item.values for item in res.embeddings)
                            if i + batch_size < len(texts):
                                time.sleep(1.0)
                            break
                        except Exception as e:
                            if attempt == max_retries - 1:
                                raise RuntimeError("Embedding retries exhausted") from e
                            if (
                                "429" in str(e)
                                or "RESOURCE_EXHAUSTED" in str(e)
                                or "Quota" in str(e)
                            ):
                                wait_sec = 25 * (attempt + 1)
                                logger.warning(
                                    f"Gemini API rate limit reached. Waiting {wait_sec}s before retry (attempt {attempt + 1}/{max_retries})..."
                                )
                                time.sleep(wait_sec)
                            else:
                                logger.error("Embedding batch failed")
                                time.sleep(2.0)
                return embeddings

        # Local ONNX model (all-MiniLM-L6-v2)
        local_ef = self._get_local_ef()
        return [[float(x) for x in vec] for vec in local_ef(texts)]

    def embed_query(self, query: str) -> List[float]:
        """
        Embed a single query text for retrieval search.
        """
        if not self._is_local():
            if self.api_key and not self._client:
                raise RuntimeError("Gemini embedding client unavailable")
            if self.api_key and self._client:
                try:
                    res = self._client.models.embed_content(
                        model=self.model, contents=query
                    )
                    return res.embeddings[0].values
                except Exception:
                    logger.error("Error embedding query via Gemini")
                    raise RuntimeError("Query embedding unavailable") from None

        local_ef = self._get_local_ef()
        return [float(x) for x in local_ef([query])[0]]


embedding_service = EmbeddingService()