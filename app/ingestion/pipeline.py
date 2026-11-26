import logging
import os
from typing import Any, Dict, Optional

from sqlalchemy.orm import Session

from app.database.models import AcademicEvent, Dokumen
from app.ingestion.chunker import smart_chunker
from app.ingestion.parser import pdf_parser
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
        tahun: Optional[int] = None,
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
        doc_record = (
            db.query(Dokumen)
            .filter(Dokumen.filename == filename)
            .order_by(Dokumen.id.asc())
            .first()
        )
        if not doc_record:
            doc_record = Dokumen(filename=filename)
            db.add(doc_record)
        doc_record.nama = nama
        doc_record.kategori = kategori
        doc_record.tahun = tahun
        doc_record.status_ingest = "processing"
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
                return {
                    "status": "error",
                    "message": "No text extracted",
                    "document_id": doc_record.id,
                }

            # 4. Prepare batch data for ChromaDB
            chunk_ids = []
            chunk_texts = []
            chunk_metas = []

            for c in chunks:
                c_id = f"doc_{doc_record.id}_chunk_{c['chunk_index']}"
                chunk_ids.append(c_id)
                chunk_texts.append(c["text"])
                chunk_metas.append(
                    {
                        "document_id": doc_record.id,
                        "document_name": doc_record.nama,
                        "filename": doc_record.filename,
                        "page": c["page"],
                        "section": c["section"],
                        "kategori": kategori,
                        "tahun": tahun or 2026,
                    }
                )

            # 5. Generate Embeddings (batch)
            logger.info(f"Generating embeddings for {len(chunk_texts)} chunks...")
            embeddings = embedding_service.embed_texts(chunk_texts)

            if len(embeddings) != len(chunk_texts):
                raise RuntimeError("Embedding count does not match chunk count")

            # Replacing all chunks also removes stale tails when a PDF becomes shorter.
            # Prepare embeddings first so provider failures do not destroy the old index.
            duplicates = (
                db.query(Dokumen)
                .filter(Dokumen.filename == filename, Dokumen.id != doc_record.id)
                .all()
            )
            vector_store.delete_by_document_id(doc_record.id)
            for duplicate in duplicates:
                vector_store.delete_by_document_id(duplicate.id)
                db.query(AcademicEvent).filter_by(
                    source_document_id=duplicate.id
                ).update({"source_document_id": doc_record.id})
                db.delete(duplicate)
            # 6. Store in ChromaDB
            vector_store.add_chunks(
                ids=chunk_ids,
                documents=chunk_texts,
                embeddings=embeddings,
                metadatas=chunk_metas,
            )

            # 7. Update DB record to done
            doc_record.jumlah_chunk = len(chunks)
            doc_record.status_ingest = "done"
            db.commit()

            logger.info(
                f"Ingestion completed for '{nama}'. {len(chunks)} chunks indexed."
            )
            return {
                "status": "success",
                "document_id": doc_record.id,
                "nama": doc_record.nama,
                "pages": total_pages,
                "chunks": len(chunks),
                "tipe": doc_record.tipe_pdf,
            }

        except Exception as e:
            logger.error("Document ingestion failed")
            db.rollback()
            doc_record.status_ingest = "error"
            db.commit()
            raise


ingestion_pipeline = IngestionPipeline()
