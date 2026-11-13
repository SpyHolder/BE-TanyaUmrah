import os
import sys
import time

# Ensure project root is in sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.database.session import SessionLocal, engine, Base
from app.database.models import Dokumen
from app.ingestion.pipeline import ingestion_pipeline
from app.rag.vector_store import vector_store
from app.core.config import settings

DOCUMENTS_METADATA = [
    {
        "filename": "Peraturan-Rektor-UMRAH-Nomor-5-Pedoman-Etika-Akademik-UMRAH-Final.pdf",
        "nama": "Pedoman Etika Akademik UMRAH",
        "kategori": "etika",
        "tahun": 2015
    },
    {
        "filename": "PERATURAN REKTOR 2 TAHUN 2024 TENTANG PENGHARGAAN GELIGA GURINDAM.pdf",
        "nama": "Peraturan Penghargaan Mahasiswa Geliga Gurindam",
        "kategori": "kemahasiswaan",
        "tahun": 2024
    },
    {
        "filename": "PEDOMAN PENYELENGGARAAN PEMBELAJARAN 2026.pdf",
        "nama": "Pedoman Penyelenggaraan Pembelajaran 2026",
        "kategori": "akademik",
        "tahun": 2026
    },
    {
        "filename": "SOP AKADEMIK BAPKK.pdf",
        "nama": "Standar Operasional Prosedur (SOP) Akademik BAPKK",
        "kategori": "sop",
        "tahun": 2024
    },
    {
        "filename": "1. Peraturan Rektor Nomor 3 Tahun 2025 tentang Peraturan Akademik (1).pdf",
        "nama": "Peraturan Rektor No. 3 Tahun 2025 tentang Peraturan Akademik",
        "kategori": "akademik",
        "tahun": 2025
    }
]

def main():
    import sys
    do_reset = "--reset" in sys.argv

    start_time = time.time()
    print("=" * 65)
    print("🚀 MEMULAI INGESTION LENGKAP 5 DOKUMEN RESMI UMRAH")
    if do_reset:
        print("🔄 Mode Reset: Mengosongkan data lama & mengindeks ulang")
    print("=" * 65)

    Base.metadata.create_all(bind=engine)
    db = SessionLocal()

    if do_reset:
        vector_store.reset_collection()
        db.query(Dokumen).delete()
        db.commit()
        print("  -> ChromaDB & Dokumen Database telah di-reset.")

    data_dir = settings.DATA_PDF_DIR
    if not os.path.exists(data_dir):
        print(f"❌ Error: Direktori '{data_dir}' tidak ditemukan!")
        return

    total_chunks_all = 0
    total_pages_all = 0

    for idx, meta in enumerate(DOCUMENTS_METADATA, 1):
        fpath = os.path.join(data_dir, meta["filename"])
        if not os.path.exists(fpath):
            print(f"⚠️ [{idx}/5] File '{meta['filename']}' tidak ditemukan di folder data, dilewati.")
            continue

        # Check if already processed
        existing = db.query(Dokumen).filter(Dokumen.filename == meta["filename"]).first()
        if existing and existing.status_ingest == "done" and existing.jumlah_chunk > 0:
            print(f"⏩ [{idx}/5] '{meta['nama']}' sudah terindeks ({existing.jumlah_chunk} chunks). Dilewati.")
            total_chunks_all += existing.jumlah_chunk
            total_pages_all += existing.jumlah_halaman
            continue

        # If existing record has error/incomplete, delete previous chunks first
        if existing:
            vector_store.delete_by_document_id(existing.id)
            db.delete(existing)
            db.commit()

        print(f"\n📂 [{idx}/5] Memproses: {meta['nama']}")
        print(f"   File: {meta['filename']}")
        t0 = time.time()
        
        try:
            result = ingestion_pipeline.ingest_pdf(
                pdf_path=fpath,
                db=db,
                nama=meta["nama"],
                kategori=meta["kategori"],
                tahun=meta["tahun"]
            )
            elapsed = time.time() - t0
            print(f"   ✅ Sukses! {result['pages']} halaman | {result['chunks']} chunk | Tipe: {result['tipe']} ({elapsed:.1f} detik)")
            total_chunks_all += result['chunks']
            total_pages_all += result['pages']
        except Exception as e:
            print(f"   ❌ Gagal memproses '{meta['nama']}': {e}")

    db.close()
    total_time = time.time() - start_time
    print("\n" + "=" * 65)
    print(f"🎉 INGESTION SELESAI DALAM {total_time:.1f} DETIK")
    print(f"📊 Total Halaman Diproses : {total_pages_all} halaman")
    print(f"📊 Total Vektor di ChromaDB: {vector_store.count_chunks()} chunks")
    print("=" * 65)

if __name__ == "__main__":
    main()
