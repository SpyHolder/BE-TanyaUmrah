import os
import shutil
import logging
from typing import Optional
from fastapi import APIRouter, Depends, Request, Form, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from app.database.session import get_db
from app.database.models import Dokumen, Mahasiswa, Feedback, AcademicEvent
from app.core.config import settings
from app.ingestion.pipeline import ingestion_pipeline
from app.rag.vector_store import vector_store

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Admin Panel"])

# Setup Jinja2 templates
templates = Jinja2Templates(directory="app/templates")

@router.get("", response_class=HTMLResponse)
def admin_dashboard(request: Request, db: Session = Depends(get_db)):
    total_docs = db.query(Dokumen).count()
    total_chunks = vector_store.count_chunks()
    total_students = db.query(Mahasiswa).count()
    
    total_feedbacks = db.query(Feedback).count()
    positive_feedbacks = db.query(Feedback).filter(Feedback.rating == "helpful").count()
    feedback_pct = int((positive_feedbacks / total_feedbacks * 100)) if total_feedbacks > 0 else 100

    recent_docs = db.query(Dokumen).order_by(Dokumen.created_at.desc()).limit(5).all()

    return templates.TemplateResponse(
        "admin/dashboard.html",
        {
            "request": request,
            "active_page": "dashboard",
            "stats": {
                "total_documents": total_docs,
                "total_chunks": total_chunks,
                "total_students": total_students,
                "positive_feedback": feedback_pct
            },
            "recent_docs": recent_docs
        }
    )

@router.get("/documents", response_class=HTMLResponse)
def admin_documents(request: Request, db: Session = Depends(get_db)):
    docs = db.query(Dokumen).order_by(Dokumen.created_at.desc()).all()
    return templates.TemplateResponse(
        "admin/documents.html",
        {
            "request": request,
            "active_page": "documents",
            "documents": docs
        }
    )

@router.post("/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    kategori: str = Form("akademik"),
    tahun: int = Form(2026),
    db: Session = Depends(get_db)
):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Hanya file berekstensi .pdf yang diperbolehkan.")

    save_path = os.path.join(settings.UPLOAD_DIR, file.filename)
    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        ingestion_pipeline.ingest_pdf(
            pdf_path=save_path,
            db=db,
            kategori=kategori,
            tahun=tahun
        )
    except Exception as e:
        logger.error(f"Failed to ingest uploaded document: {e}")

    return RedirectResponse(url="/admin/documents", status_code=303)

@router.post("/documents/sync-data-folder")
def sync_data_folder(db: Session = Depends(get_db)):
    """
    Scans C:\HOLDER\KULIAH\Semester 5\PPB\data and ingests any PDFs not yet indexed.
    """
    data_dir = settings.DATA_PDF_DIR
    if os.path.exists(data_dir):
        for fname in os.listdir(data_dir):
            if fname.endswith(".pdf"):
                # Check if already indexed
                existing = db.query(Dokumen).filter(Dokumen.filename == fname).first()
                if not existing:
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
                            pdf_path=fpath,
                            db=db,
                            kategori=kategori,
                            tahun=2026
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
        "admin/students.html",
        {
            "request": request,
            "active_page": "students",
            "students": students
        }
    )

@router.post("/students/save")
def save_student(
    nim: str = Form(...),
    nama: str = Form(...),
    prodi: str = Form("Teknik Informatika"),
    semester: int = Form(5),
    total_sks: int = Form(95),
    ipk: float = Form(3.45),
    db: Session = Depends(get_db)
):
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
            status="aktif"
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
        "admin/calendar.html",
        {
            "request": request,
            "active_page": "calendar",
            "events": events
        }
    )

@router.post("/calendar/save")
def save_calendar_event(
    title: str = Form(...),
    start_date: str = Form(...),
    end_date: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
    category: str = Form("akademik"),
    db: Session = Depends(get_db)
):
    ev = AcademicEvent(
        title=title,
        start_date=start_date,
        end_date=end_date if end_date else None,
        description=description,
        category=category,
        is_approved=True
    )
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
