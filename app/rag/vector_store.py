import os
import chromadb
from chromadb.config import Settings as ChromaSettings
from typing import List, Dict, Any, Optional
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self, persist_dir: str = settings.CHROMA_PERSIST_DIR):
        self.persist_dir = persist_dir
        os.makedirs(self.persist_dir, exist_ok=True)
        self.client = chromadb.PersistentClient(path=self.persist_dir)
        self.collection_name = "umrah_documents"
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def add_chunks(
        self,
        ids: List[str],
        documents: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]]
    ):
        """
        Add document chunks and their embeddings to ChromaDB.
        """
        if not ids:
            return
            
        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )
        logger.info(f"Added {len(ids)} chunks to ChromaDB collection '{self.collection_name}'.")

    def query_similar(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        where_filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for top_k most similar document chunks.
        Returns a list of dicts with document, score, and metadata.
        """
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where_filter
        )
        
        output = []
        if not results['documents'] or not results['documents'][0]:
            return output
            
        docs = results['documents'][0]
        metas = results['metadatas'][0] if results['metadatas'] else [{}] * len(docs)
        distances = results['distances'][0] if results['distances'] else [0.0] * len(docs)
        ids = results['ids'][0] if results['ids'] else [""] * len(docs)
        
        for doc_id, doc, meta, dist in zip(ids, docs, metas, distances):
            # Cosine distance in Chroma: 0 is identical, 2 is opposite.
            # Similarity score = 1 - (dist / 2) or 1 - dist
            similarity = 1.0 - (dist if dist <= 1.0 else 1.0)
            output.append({
                "id": doc_id,
                "content": doc,
                "metadata": meta,
                "similarity": round(similarity, 4),
                "distance": round(dist, 4)
            })
            
        return output

    def delete_by_document_id(self, document_id: int):
        """
        Delete all chunks associated with a specific document ID.
        """
        self.collection.delete(
            where={"document_id": document_id}
        )
        logger.info(f"Deleted chunks for document_id={document_id} from ChromaDB.")

    def reset_collection(self):
        """
        Deletes and recreates the collection with fresh embedding configuration.
        """
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        logger.info(f"Reset ChromaDB collection '{self.collection_name}'.")

    def count_chunks(self) -> int:
        return self.collection.count()

vector_store = VectorStore()
