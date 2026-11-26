from pathlib import Path

import pymupdf as fitz
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse, Response
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.security import require_document_reader, require_student
from app.database.models import Dokumen
from app.database.session import get_db
from app.ingestion.parser import pdf_parser

router = APIRouter(prefix="/documents", tags=["Documents"])


def _resolve_pdf_path(doc: Dokumen) -> Path:
    for directory in (settings.UPLOAD_DIR, settings.DATA_PDF_DIR):
        root = Path(directory).resolve()
        candidate = (root / doc.filename).resolve()
        if (
            candidate.is_relative_to(root)
            and candidate.is_file()
            and candidate.suffix.lower() == ".pdf"
        ):
            return candidate
    raise HTTPException(
        status_code=404, detail="File PDF fisik tidak ditemukan di server."
    )


@router.get("", dependencies=[Depends(require_student)])
def list_documents(db: Session = Depends(get_db)):
    """
    List all indexed documents in the knowledge base.
    """
    docs = db.query(Dokumen).order_by(Dokumen.created_at.desc()).all()
    return {
        "status": "success",
        "total": len(docs),
        "documents": [
            {
                "id": d.id,
                "nama": d.nama,
                "filename": d.filename,
                "kategori": d.kategori,
                "tahun": d.tahun,
                "jumlah_halaman": d.jumlah_halaman,
                "jumlah_chunk": d.jumlah_chunk,
                "tipe_pdf": d.tipe_pdf,
                "status_ingest": d.status_ingest,
                "file_url": f"/api/v1/documents/{d.id}/file",
            }
            for d in docs
        ],
    }


@router.get("/{document_id}/file", dependencies=[Depends(require_document_reader)])
def get_document_pdf(document_id: int, db: Session = Depends(get_db)):
    """
    Stream or download the actual PDF file for in-app viewing.
    """
    doc = db.query(Dokumen).filter(Dokumen.id == document_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Dokumen tidak ditemukan.")

    target_path = _resolve_pdf_path(doc)

    return FileResponse(
        path=target_path,
        media_type="application/pdf",
        filename=target_path.name,
        headers={
            "Cache-Control": "private, no-store",
            "X-Content-Type-Options": "nosniff",
        },
    )


@router.get("/{document_id}/pages/{page_number}", dependencies=[Depends(require_student)])
def get_document_page_info(
    document_id: int, page_number: int, db: Session = Depends(get_db)
):
    """
    Retrieve extracted text and metadata for a specific 1-based page of a PDF document.
    """
    doc = db.query(Dokumen).filter(Dokumen.id == document_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Dokumen tidak ditemukan.")

    target_path = _resolve_pdf_path(doc)

    try:
        pages_data = pdf_parser.parse_pdf(str(target_path))
        total_pages = len(pages_data) or (doc.jumlah_halaman or 1)
    except Exception:
        pages_data = []
        total_pages = doc.jumlah_halaman or 1

    if page_number < 1 or page_number > max(1, total_pages):
        raise HTTPException(
            status_code=404,
            detail=f"Halaman {page_number} di luar rentang dokumen (1-{total_pages}).",
        )

    page_text = ""
    is_ocr = False
    for p in pages_data:
        if p.get("page_number") == page_number:
            page_text = p.get("text", "")
            is_ocr = bool(p.get("is_ocr", False))
            break

    return {
        "status": "success",
        "document_id": doc.id,
        "nama": doc.nama,
        "filename": doc.filename,
        "kategori": doc.kategori,
        "tahun": doc.tahun,
        "page": page_number,
        "total_pages": total_pages,
        "is_ocr": is_ocr,
        "text": page_text,
        "image_url": f"/api/v1/documents/{doc.id}/pages/{page_number}/image",
    }


@router.get(
    "/{document_id}/pages/{page_number}/image",
    dependencies=[Depends(require_document_reader)],
)
def get_document_page_image(
    document_id: int, page_number: int, db: Session = Depends(get_db)
):
    """
    Render a specific 1-based page of a PDF document into a PNG image for in-app viewing.
    """
    doc = db.query(Dokumen).filter(Dokumen.id == document_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Dokumen tidak ditemukan.")

    target_path = _resolve_pdf_path(doc)

    pdf_doc = fitz.open(str(target_path))
    try:
        total_pages = len(pdf_doc)
        if page_number < 1 or page_number > total_pages:
            raise HTTPException(
                status_code=404,
                detail=f"Halaman {page_number} tidak ditemukan (total {total_pages} halaman).",
            )
        page = pdf_doc[page_number - 1]
        mat = fitz.Matrix(1.8, 1.8)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img_bytes = pix.tobytes("png")
    finally:
        pdf_doc.close()

    return Response(
        content=img_bytes,
        media_type="image/png",
        headers={
            "Cache-Control": "private, max-age=300",
            "X-Content-Type-Options": "nosniff",
        },
    )
