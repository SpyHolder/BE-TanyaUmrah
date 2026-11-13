import logging
from typing import List
from app.core.config import settings

logger = logging.getLogger(__name__)

class EmbeddingService:
    def __init__(self):
        self.api_key = settings.GEMINI_API_KEY
        self.model = settings.EMBEDDING_MODEL
        self._client = None
        
        if self.api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self._genai = genai
            except Exception as e:
                logger.warning(f"Failed to configure Google GenAI: {e}")

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of text strings into vectors.
        """
        if not texts:
            return []
            
        if not self.api_key:
            logger.warning("GEMINI_API_KEY is not set. Generating fallback zero-vector embeddings.")
            return [[0.0] * 768 for _ in texts]
            
        try:
            # Batch embedding via google.generativeai
            embeddings = []
            # API supports batching or single calls
            for text in texts:
                result = self._genai.embed_content(
                    model=self.model,
                    content=text,
                    task_type="retrieval_document"
                )
                embeddings.append(result['embedding'])
            return embeddings
        except Exception as e:
            logger.error(f"Error embedding texts via Gemini: {e}")
            raise e

    def embed_query(self, query: str) -> List[float]:
        """
        Embed a single query text for retrieval search.
        """
        if not self.api_key:
            logger.warning("GEMINI_API_KEY is not set. Returning fallback zero-vector.")
            return [0.0] * 768
            
        try:
            result = self._genai.embed_content(
                model=self.model,
                content=query,
                task_type="retrieval_query"
            )
            return result['embedding']
        except Exception as e:
            logger.error(f"Error embedding query via Gemini: {e}")
            raise e

embedding_service = EmbeddingService()
