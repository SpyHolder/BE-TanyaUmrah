from typing import Annotated, Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials
from pydantic import BaseModel, ConfigDict, Field, StringConstraints
from sqlalchemy.orm import Session

from app.core.security import (
    StudentIdentity,
    bearer,
    require_owner,
    require_student,
    verify_student_token,
)
from app.database.models import Mahasiswa
from app.database.session import get_db

router = APIRouter(tags=["Auth & Profile"])
ShortText = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=100)
]


class VerifyTokenRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id_token: str = Field(min_length=1, max_length=16384)


class ProfileUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    fakultas: Optional[ShortText] = None
    prodi: Optional[ShortText] = None
    semester: Optional[int] = Field(None, ge=1, le=30)
    total_sks: Optional[int] = Field(None, ge=0, le=300)
    ipk: Optional[float] = Field(None, ge=0, le=4)
    mk_lulus: Optional[list[ShortText]] = Field(None, max_length=150)


def profile_data(student):
    return {
        key: getattr(student, key)
        for key in (
            "nim",
            "nama",
            "email",
            "photo_url",
            "fakultas",
            "prodi",
            "semester",
            "total_sks",
            "ipk",
            "mk_lulus",
            "status",
        )
    }


@router.post("/auth/verify")
def verify_student_auth(
    req: Optional[VerifyTokenRequest] = None,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer),
    db: Session = Depends(get_db),
):
    if req and credentials and req.id_token != credentials.credentials:
        raise HTTPException(400, "Token pada body dan header harus sama.")
    token = req.id_token if req else credentials.credentials if credentials else None
    if not token:
        raise HTTPException(
            401, "Firebase ID token diperlukan.", headers={"WWW-Authenticate": "Bearer"}
        )
    identity = verify_student_token(token)
    student = db.query(Mahasiswa).filter(Mahasiswa.nim == identity.nim).first()
    is_new = student is None
    if is_new:
        student = Mahasiswa(
            nim=identity.nim,
            nama=identity.name,
            email=identity.email,
            photo_url=identity.picture,
            semester=1,
            total_sks=0,
            ipk=0,
            mk_lulus=[],
            status="aktif",
        )
        db.add(student)
    else:
        student.nama = identity.name
        student.email = identity.email
        student.photo_url = identity.picture
    db.commit()
    db.refresh(student)
    return {
        "status": "success",
        "is_new_user": is_new,
        "profile": profile_data(student),
    }


@router.get("/user/profile/{nim}")
def get_student_profile(
    nim: str,
    identity: StudentIdentity = Depends(require_student),
    db: Session = Depends(get_db),
):
    require_owner(nim, identity)
    student = db.query(Mahasiswa).filter(Mahasiswa.nim == identity.nim).first()
    if not student:
        raise HTTPException(
            404, "Data mahasiswa tidak ditemukan. Lakukan /auth/verify terlebih dahulu."
        )
    return profile_data(student)


@router.put("/user/profile/{nim}")
def update_student_profile(
    nim: str,
    req: ProfileUpdateRequest,
    identity: StudentIdentity = Depends(require_student),
    db: Session = Depends(get_db),
):
    require_owner(nim, identity)
    student = db.query(Mahasiswa).filter(Mahasiswa.nim == identity.nim).first()
    if not student:
        raise HTTPException(404, "Data mahasiswa tidak ditemukan.")
    for key, value in req.model_dump(exclude_none=True).items():
        setattr(student, key, value)
    db.commit()
    db.refresh(student)
    return {
        "status": "success",
        "message": "Profil berhasil diperbarui",
        "profile": profile_data(student),
    }
