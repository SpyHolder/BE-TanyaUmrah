import logging
import os
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from pydantic import ValidationError
from sqlalchemy.orm import Session

from app.core.config import BACKEND_DIR, settings
from app.core.security import require_admin
from app.core.uploads import save_pdf_upload
from app.database.models import AcademicEvent, Dokumen, Feedback, Mahasiswa
from app.database.session import get_db
from app.ingestion.pipeline import ingestion_pipeline
from app.rag.vector_store import vector_store
from app.routers.auth import ProfileUpdateRequest
from app.routers.calendar import EventCreateRequest, event_values

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Admin Panel"], dependencies=[Depends(require_admin)])

# Setup Jinja2 templates
templates = Jinja2Templates(directory=str(BACKEND_DIR / "app" / "templates"))


@router.get("", response_class=HTMLResponse)
def admin_dashboard(request: Request, db: Session = Depends(get_db)):
    total_docs = db.query(Dokumen).count()
    total_chunks = vector_store.count_chunks()
    total_students = db.query(Mahasiswa).count()

    total_feedbacks = db.query(Feedback).count()
    positive_feedbacks = db.query(Feedback).filter(Feedback.rating == "helpful").count()
    feedback_pct = (
        int((positive_feedbacks / total_feedbacks * 100))
        if total_feedbacks > 0
        else 100
    )

    recent_docs = db.query(Dokumen).order_by(Dokumen.created_at.desc()).limit(5).all()

    return templates.TemplateResponse(
        name="admin/dashboard.html",
        request=request,
        context={
            "request": request,
            "active_page": "dashboard",
            "stats": {
                "total_documents": total_docs,
                "total_chunks": total_chunks,
                "total_students": total_students,
                "positive_feedback": feedback_pct,
            },
            "recent_docs": recent_docs,
        },
    )


@router.get("/documents", response_class=HTMLResponse)
def admin_documents(request: Request, db: Session = Depends(get_db)):
    docs = db.query(Dokumen).order_by(Dokumen.created_at.desc()).all()
    return templates.TemplateResponse(
        name="admin/documents.html",
        request=request,
        context={"request": request, "active_page": "documents", "documents": docs},
    )


@router.post("/documents/upload")
def upload_document(
    file: UploadFile = File(...),
    kategori: str = Form("akademik"),
    tahun: int = Form(2026, ge=1900, le=9999),
    db: Session = Depends(get_db),
):
    save_path = save_pdf_upload(file)
    try:
        result = ingestion_pipeline.ingest_pdf(
            pdf_path=str(save_path),
            nama=Path(file.filename.replace("\\", "/")).stem[:255],
            db=db,
            kategori=kategori,
            tahun=tahun,
        )
    except Exception:
        logger.error("Failed to ingest uploaded document")
        raise HTTPException(
            503,
            "Ingestion gagal. Periksa konfigurasi embedding dan status dokumen admin.",
        ) from None
    if result.get("status") != "success":
        raise HTTPException(422, "PDF tidak menghasilkan teks yang dapat diindeks.")

    return RedirectResponse(url="/admin/documents", status_code=303)


@router.post("/documents/sync-data-folder")
def sync_data_folder(db: Session = Depends(get_db)):
    """Ingest PDFs in configured DATA_PDF_DIR; reuse existing document records."""
    data_dir = settings.DATA_PDF_DIR
    if os.path.exists(data_dir):
        for fname in os.listdir(data_dir):
            if fname.endswith(".pdf"):
                # Check if already indexed
                existing = db.query(Dokumen).filter(Dokumen.filename == fname).first()
                if not existing or existing.status_ingest != "processing":
                    fpath = os.path.join(data_dir, fname)
                    try:
                        kategori = "akademik"
                        if "GURINDAM" in fname.upper():
                            kategori = "kemahasiswaan"
                        elif "ETIKA" in fname.upper():
                            kategori = "etika"
                        elif "SOP" in fname.upper():
                            kategori = "sop"

                        ingestion_pipeline.ingest_pdf(
                            pdf_path=fpath, db=db, kategori=kategori, tahun=2026
                        )
                    except Exception as err:
                        logger.error(f"Error syncing '{fname}': {err}")

    return RedirectResponse(url="/admin/documents", status_code=303)


@router.post("/documents/{document_id}/delete")
def delete_document(document_id: int, db: Session = Depends(get_db)):
    doc = db.query(Dokumen).filter(Dokumen.id == document_id).first()
    if doc:
        # 1. Remove chunks from ChromaDB
        vector_store.delete_by_document_id(doc.id)
        # 2. Remove DB record
        db.delete(doc)
        db.commit()
    return RedirectResponse(url="/admin/documents", status_code=303)


@router.get("/students", response_class=HTMLResponse)
def admin_students(request: Request, db: Session = Depends(get_db)):
    students = db.query(Mahasiswa).order_by(Mahasiswa.created_at.desc()).all()
    return templates.TemplateResponse(
        name="admin/students.html",
        request=request,
        context={"request": request, "active_page": "students", "students": students},
    )


@router.post("/students/save")
def save_student(
    nim: str = Form(...),
    nama: str = Form(...),
    prodi: str = Form("Teknik Informatika"),
    semester: int = Form(5, ge=1, le=30),
    total_sks: int = Form(95, ge=0, le=300),
    ipk: float = Form(3.45, ge=0, le=4),
    db: Session = Depends(get_db),
):
    import re

    if not re.fullmatch(r"[0-9]{1,20}", nim):
        raise HTTPException(422, "NIM harus seluruhnya numerik (maksimal 20 digit).")
    try:
        ProfileUpdateRequest(
            prodi=prodi, semester=semester, total_sks=total_sks, ipk=ipk
        )
    except ValidationError:
        raise HTTPException(422, "Data profil tidak valid.") from None
    student = db.query(Mahasiswa).filter(Mahasiswa.nim == nim).first()
    if not student:
        student = Mahasiswa(
            nim=nim,
            nama=nama,
            email=f"{nim}@student.umrah.ac.id",
            prodi=prodi,
            semester=semester,
            total_sks=total_sks,
            ipk=ipk,
            status="aktif",
        )
        db.add(student)
    else:
        student.nama = nama
        student.prodi = prodi
        student.semester = semester
        student.total_sks = total_sks
        student.ipk = ipk

    db.commit()
    return RedirectResponse(url="/admin/students", status_code=303)


@router.get("/calendar", response_class=HTMLResponse)
def admin_calendar(request: Request, db: Session = Depends(get_db)):
    events = db.query(AcademicEvent).order_by(AcademicEvent.start_date.asc()).all()
    return templates.TemplateResponse(
        name="admin/calendar.html",
        request=request,
        context={"request": request, "active_page": "calendar", "events": events},
    )


@router.post("/calendar/save")
def save_calendar_event(
    title: str = Form(...),
    start_date: str = Form(...),
    end_date: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
    category: str = Form("akademik"),
    db: Session = Depends(get_db),
):
    try:
        req = EventCreateRequest(
            title=title,
            start_date=start_date,
            end_date=end_date or None,
            description=description,
            category=category,
        )
    except ValidationError:
        raise HTTPException(
            422, "Tanggal, kategori, atau judul event tidak valid."
        ) from None
    ev = AcademicEvent(**event_values(req), is_approved=True)
    db.add(ev)
    db.commit()
    return RedirectResponse(url="/admin/calendar", status_code=303)


@router.post("/calendar/{event_id}/delete")
def delete_calendar_event(event_id: int, db: Session = Depends(get_db)):
    ev = db.query(AcademicEvent).filter(AcademicEvent.id == event_id).first()
    if ev:
        db.delete(ev)
        db.commit()
    return RedirectResponse(url="/admin/calendar", status_code=303)
