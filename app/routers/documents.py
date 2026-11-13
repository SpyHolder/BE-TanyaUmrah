import os
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from app.database.session import get_db
from app.database.models import Dokumen
from app.core.config import settings

router = APIRouter(prefix="/documents", tags=["Documents"])

@router.get("")
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
                "file_url": f"/api/v1/documents/{d.id}/file"
            }
            for d in docs
        ]
    }

@router.get("/{document_id}/file")
def get_document_pdf(document_id: int, db: Session = Depends(get_db)):
    """
    Stream or download the actual PDF file for in-app viewing.
    """
    doc = db.query(Dokumen).filter(Dokumen.id == document_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Dokumen tidak ditemukan.")

    # Search in upload dir or initial data dir
    upload_path = os.path.join(settings.UPLOAD_DIR, doc.filename)
    data_dir_path = os.path.join(settings.DATA_PDF_DIR, doc.filename)

    target_path = None
    if os.path.exists(upload_path):
        target_path = upload_path
    elif os.path.exists(data_dir_path):
        target_path = data_dir_path

    if not target_path or not os.path.exists(target_path):
        raise HTTPException(status_code=404, detail="File PDF fisik tidak ditemukan di server.")

    return FileResponse(
        path=target_path,
        media_type="application/pdf",
        filename=doc.filename
    )
