import re
import logging
from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, Header
from pydantic import BaseModel
from sqlalchemy.orm import Session
from app.database.session import get_db
from app.database.models import Mahasiswa

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Auth & Profile"])

class VerifyTokenRequest(BaseModel):
    id_token: Optional[str] = None
    email: str
    display_name: Optional[str] = "Mahasiswa UMRAH"
    photo_url: Optional[str] = None

class ProfileUpdateRequest(BaseModel):
    fakultas: Optional[str] = None
    prodi: Optional[str] = None
    semester: Optional[int] = None
    total_sks: Optional[int] = None
    ipk: Optional[float] = None
    mk_lulus: Optional[List[str]] = None

@router.post("/auth/verify")
def verify_student_auth(
    req: VerifyTokenRequest,
    db: Session = Depends(get_db)
):
    """
    Verifies student Google email domain (@student.umrah.ac.id) and loads or initializes profile.
    """
    email = req.email.strip().lower()

    # Domain verification
    if not (email.endswith("@student.umrah.ac.id") or email.endswith("@umrah.ac.id")):
        raise HTTPException(
            status_code=403,
            detail="Aplikasi ini khusus untuk mahasiswa aktif UMRAH. Harap gunakan akun Google @student.umrah.ac.id."
        )

    # Extract NIM from email prefix (e.g. '2401020163' from '2401020163@student.umrah.ac.id')
    nim_match = re.match(r"^(\d+)", email)
    nim = nim_match.group(1) if nim_match else email.split("@")[0]

    # Find existing student or create new profile
    student = db.query(Mahasiswa).filter(Mahasiswa.nim == nim).first()
    is_new = False

    if not student:
        is_new = True
        student = Mahasiswa(
            nim=nim,
            nama=req.display_name or f"Mahasiswa {nim}",
            email=email,
            photo_url=req.photo_url,
            fakultas="Teknik dan Ilmu Kelautan",
            prodi="Teknik Informatika",
            semester=1,
            total_sks=0,
            ipk=0.0,
            mk_lulus=[],
            status="aktif"
        )
        db.add(student)
        db.commit()
        db.refresh(student)
    else:
        # Update photo and name if changed
        if req.photo_url and student.photo_url != req.photo_url:
            student.photo_url = req.photo_url
        if req.display_name and student.nama != req.display_name:
            student.nama = req.display_name
        db.commit()

    return {
        "status": "success",
        "is_new_user": is_new,
        "profile": {
            "nim": student.nim,
            "nama": student.nama,
            "email": student.email,
            "photo_url": student.photo_url,
            "fakultas": student.fakultas,
            "prodi": student.prodi,
            "semester": student.semester,
            "total_sks": student.total_sks,
            "ipk": student.ipk,
            "mk_lulus": student.mk_lulus or [],
            "status": student.status
        }
    }

@router.get("/user/profile/{nim}")
def get_student_profile(
    nim: str,
    db: Session = Depends(get_db)
):
    student = db.query(Mahasiswa).filter(Mahasiswa.nim == nim).first()
    if not student:
        raise HTTPException(status_code=404, detail="Data mahasiswa tidak ditemukan.")
    return student

@router.put("/user/profile/{nim}")
def update_student_profile(
    nim: str,
    req: ProfileUpdateRequest,
    db: Session = Depends(get_db)
):
    student = db.query(Mahasiswa).filter(Mahasiswa.nim == nim).first()
    if not student:
        raise HTTPException(status_code=404, detail="Data mahasiswa tidak ditemukan.")

    if req.fakultas is not None:
        student.fakultas = req.fakultas
    if req.prodi is not None:
        student.prodi = req.prodi
    if req.semester is not None:
        student.semester = req.semester
    if req.total_sks is not None:
        student.total_sks = req.total_sks
    if req.ipk is not None:
        student.ipk = req.ipk
    if req.mk_lulus is not None:
        student.mk_lulus = req.mk_lulus

    db.commit()
    db.refresh(student)
    return {"status": "success", "message": "Profil berhasil diperbarui", "profile": student}
