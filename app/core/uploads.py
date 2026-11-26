import hashlib
from pathlib import Path
from uuid import uuid4

import pymupdf
from fastapi import HTTPException, UploadFile

from app.core.config import settings


def save_pdf_upload(file: UploadFile) -> Path:
    """Bound the copy and never use client input as a filesystem path."""
    if (
        not file.filename
        or not file.filename.lower().endswith(".pdf")
        or file.content_type not in ("application/pdf", "application/octet-stream")
    ):
        raise HTTPException(400, "Hanya file PDF yang diperbolehkan.")
    root = Path(settings.UPLOAD_DIR).resolve()
    root.mkdir(parents=True, exist_ok=True)
    destination = root / f"{uuid4().hex}.pdf"
    try:
        size = 0
        digest = hashlib.sha256()
        with destination.open("xb") as output:
            while block := file.file.read(64 * 1024):
                size += len(block)
                if size > settings.MAX_UPLOAD_BYTES:
                    raise HTTPException(413, "Ukuran PDF melebihi MAX_UPLOAD_BYTES.")
                if size == len(block) and not block.startswith(b"%PDF-"):
                    raise HTTPException(400, "Konten file bukan PDF.")
                digest.update(block)
                output.write(block)
        try:
            with pymupdf.open(destination) as pdf:
                if pdf.needs_pass or len(pdf) == 0:
                    raise ValueError("empty or encrypted PDF")
        except Exception:
            raise HTTPException(400, "PDF rusak, kosong, atau terenkripsi.") from None
        # Identical uploads share one canonical file and ingestion record.
        canonical = root / f"{digest.hexdigest()}.pdf"
        destination.replace(canonical)
        return canonical
    except Exception:
        destination.unlink(missing_ok=True)
        raise
