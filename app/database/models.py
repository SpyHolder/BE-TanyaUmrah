from datetime import datetime
from sqlalchemy import Column, Integer, String, Text, Float, Boolean, DateTime, ForeignKey, JSON
from app.database.session import Base

class Mahasiswa(Base):
    __tablename__ = "mahasiswa"
    
    nim = Column(String(20), primary_key=True, index=True)
    nama = Column(String(100), nullable=False)
    email = Column(String(100), unique=True, index=True, nullable=False)
    photo_url = Column(Text, nullable=True)
    fakultas = Column(String(100), nullable=True, default="Teknik dan Ilmu Kelautan")
    prodi = Column(String(100), nullable=True, default="Teknik Informatika")
    semester = Column(Integer, default=1)
    total_sks = Column(Integer, default=0)
    ipk = Column(Float, default=0.0)
    mk_lulus = Column(JSON, default=list)  # list of strings e.g. ["Algoritma", "Kalkulus"]
    status = Column(String(20), default="aktif")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class AdminUser(Base):
    __tablename__ = "admin_users"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    role = Column(String(20), default="admin")
    created_at = Column(DateTime, default=datetime.utcnow)


class Dokumen(Base):
    __tablename__ = "dokumen"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    nama = Column(String(255), nullable=False)
    filename = Column(String(255), nullable=False)
    kategori = Column(String(50), default="akademik")
    tahun = Column(Integer, nullable=True)
    jumlah_halaman = Column(Integer, default=0)
    jumlah_chunk = Column(Integer, default=0)
    tipe_pdf = Column(String(20), default="text")  # 'text' or 'scan'
    status_ingest = Column(String(20), default="processing")  # 'processing', 'done', 'error'
    created_at = Column(DateTime, default=datetime.utcnow)


class Feedback(Base):
    __tablename__ = "feedback"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    nim = Column(String(20), ForeignKey("mahasiswa.nim"), nullable=True)
    conversation_id = Column(String(100), nullable=True)
    message_id = Column(String(100), nullable=True)
    rating = Column(String(20), nullable=False)  # 'helpful' or 'not_helpful'
    comment = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class AcademicEvent(Base):
    __tablename__ = "academic_events"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    title = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    start_date = Column(String(20), nullable=False)  # YYYY-MM-DD
    end_date = Column(String(20), nullable=True)    # YYYY-MM-DD
    category = Column(String(50), default="akademik")  # 'akademik', 'kemahasiswaan', 'umum'
    source_document_id = Column(Integer, ForeignKey("dokumen.id"), nullable=True)
    is_approved = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
