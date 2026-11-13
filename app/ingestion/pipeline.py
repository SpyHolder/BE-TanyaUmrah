import os
import logging
from typing import Dict, Any, Optional
from sqlalchemy.orm import Session
from app.database.models import Dokumen
from app.ingestion.parser import pdf_parser
from app.ingestion.chunker import smart_chunker
from app.rag.embeddings import embedding_service
from app.rag.vector_store import vector_store

logger = logging.getLogger(__name__)

class IngestionPipeline:
    def ingest_pdf(
        self,
        pdf_path: str,
        db: Session,
        nama: Optional[str] = None,
        kategori: str = "akademik",
        tahun: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Orchestrates full ingestion of a PDF:
        Parsing -> OCR (if scanned) -> Chunking -> Embedding -> ChromaDB & SQLite storage.
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        filename = os.path.basename(pdf_path)
        if not nama:
            nama = os.path.splitext(filename)[0]

        logger.info(f"Starting ingestion pipeline for '{nama}'...")

        # 1. Register or get DB record
        doc_record = Dokumen(
            nama=nama,
            filename=filename,
            kategori=kategori,
            tahun=tahun,
            status_ingest="processing"
        )
        db.add(doc_record)
        db.commit()
        db.refresh(doc_record)

        try:
            # 2. Parse PDF (auto-detects digital vs scan)
            pages_data = pdf_parser.parse_pdf(pdf_path)
            total_pages = len(pages_data)
            has_scan = any(p["is_ocr"] for p in pages_data)
            doc_record.tipe_pdf = "scan" if has_scan else "text"
            doc_record.jumlah_halaman = total_pages

            # 3. Smart Chunking
            chunks = smart_chunker.chunk_pages(pages_data)
            logger.info(f"Generated {len(chunks)} chunks from {total_pages} pages.")

            if not chunks:
                logger.warning(f"No textual content extracted from '{filename}'.")
                doc_record.status_ingest = "error"
                db.commit()
                return {"status": "error", "message": "No text extracted", "document_id": doc_record.id}

            # 4. Prepare batch data for ChromaDB
            chunk_ids = []
            chunk_texts = []
            chunk_metas = []

            for c in chunks:
                c_id = f"doc_{doc_record.id}_chunk_{c['chunk_index']}"
                chunk_ids.append(c_id)
                chunk_texts.append(c["text"])
                chunk_metas.append({
                    "document_id": doc_record.id,
                    "document_name": doc_record.nama,
                    "filename": doc_record.filename,
                    "page": c["page"],
                    "section": c["section"],
                    "kategori": kategori,
                    "tahun": tahun or 2026
                })

            # 5. Generate Embeddings (batch)
            logger.info(f"Generating embeddings for {len(chunk_texts)} chunks...")
            embeddings = embedding_service.embed_texts(chunk_texts)

            # 6. Store in ChromaDB
            vector_store.add_chunks(
                ids=chunk_ids,
                documents=chunk_texts,
                embeddings=embeddings,
                metadatas=chunk_metas
            )

            # 7. Update DB record to done
            doc_record.jumlah_chunk = len(chunks)
            doc_record.status_ingest = "done"
            db.commit()

            logger.info(f"Ingestion completed for '{nama}'. {len(chunks)} chunks indexed.")
            return {
                "status": "success",
                "document_id": doc_record.id,
                "nama": doc_record.nama,
                "pages": total_pages,
                "chunks": len(chunks),
                "tipe": doc_record.tipe_pdf
            }

        except Exception as e:
            logger.error(f"Error ingesting '{nama}': {e}", exc_info=True)
            doc_record.status_ingest = "error"
            db.commit()
            raise e

ingestion_pipeline = IngestionPipeline()
