import time
import logging
from typing import List
from app.core.config import settings

logger = logging.getLogger(__name__)

class EmbeddingService:
    def __init__(self):
        self.api_key = settings.GEMINI_API_KEY
        self.model = settings.EMBEDDING_MODEL
        self._client = None
        self._local_ef = None
        
        if self.api_key:
            try:
                from google import genai
                self._client = genai.Client(api_key=self.api_key)
                logger.info(f"EmbeddingService: Using official Google GenAI ({self.model})")
            except Exception as e:
                logger.warning(f"Failed to configure Google GenAI client: {e}")

    def _get_local_ef(self):
        if self._local_ef is None:
            import chromadb.utils.embedding_functions as ef
            self._local_ef = ef.DefaultEmbeddingFunction()
            logger.info("EmbeddingService: Using local all-MiniLM-L6-v2 ONNX embedding engine.")
        return self._local_ef

    def embed_texts(self, texts: List[str], batch_size: int = 25) -> List[List[float]]:
        """
        Embed a list of text strings into vectors.
        Sends texts in batches in single API requests to avoid hitting rate limits.
        Automatically retries with exponential backoff if rate-limited.
        """
        if not texts:
            return []

        if self.api_key and self._client:
            embeddings = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        res = self._client.models.embed_content(
                            model=self.model,
                            contents=batch
                        )
                        for item in res.embeddings:
                            embeddings.append(item.values)
                        # Small polite pause between batches
                        if i + batch_size < len(texts):
                            time.sleep(1.0)
                        break
                    except Exception as e:
                        if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e) or "Quota" in str(e):
                            wait_sec = 25 * (attempt + 1)
                            logger.warning(f"Gemini API rate limit reached. Waiting {wait_sec}s before retry (attempt {attempt+1}/{max_retries})...")
                            time.sleep(wait_sec)
                        else:
                            logger.error(f"Error embedding batch ({i} to {i+len(batch)}): {e}")
                            if attempt == max_retries - 1:
                                raise e
                            time.sleep(2.0)
            return embeddings

        # Fallback to local ONNX model if no API key is configured
        local_ef = self._get_local_ef()
        return local_ef(texts)

    def embed_query(self, query: str) -> List[float]:
        """
        Embed a single query text for retrieval search.
        """
        if self.api_key and self._client:
            try:
                res = self._client.models.embed_content(
                    model=self.model,
                    contents=query
                )
                return res.embeddings[0].values
            except Exception as e:
                logger.error(f"Error embedding query via Gemini: {e}")
                raise e

        local_ef = self._get_local_ef()
        return local_ef([query])[0]

embedding_service = EmbeddingService()



