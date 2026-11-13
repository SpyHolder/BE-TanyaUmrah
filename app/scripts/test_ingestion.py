import os
import sys

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.database.session import SessionLocal, engine, Base
from app.database.models import Dokumen
from app.ingestion.parser import pdf_parser
from app.ingestion.chunker import smart_chunker
from app.rag.vector_store import vector_store
from app.core.config import settings

def run_test():
    print("=== TEST BACKEND INGESTION & OCR PIPELINE ===")
    
    # 1. Initialize DB
    print("[1/4] Initializing database tables...")
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    print("  -> SQLite Database initialized.")

    # 2. Check data directory
    data_dir = settings.DATA_PDF_DIR
    print(f"[2/4] Checking PDF files in '{data_dir}'...")
    if not os.path.exists(data_dir):
        print(f"  -> ERROR: Directory '{data_dir}' does not exist!")
        return

    pdf_files = [f for f in os.listdir(data_dir) if f.endswith(".pdf")]
    print(f"  -> Found {len(pdf_files)} PDF files.")

    # 3. Test parsing on digital PDF
    digital_sample = "Peraturan-Rektor-UMRAH-Nomor-5-Pedoman-Etika-Akademik-UMRAH-Final.pdf"
    if digital_sample in pdf_files:
        print(f"\n[3/4] Testing Digital PDF Parser: {digital_sample}")
        sample_path = os.path.join(data_dir, digital_sample)
        pages = pdf_parser.parse_pdf(sample_path)
        print(f"  -> Extracted {len(pages)} pages.")
        if pages:
            print(f"  -> Sample Page 1 Text snippet: {pages[0]['text'][:150]}...")
            chunks = smart_chunker.chunk_pages(pages)
            print(f"  -> Chunked into {len(chunks)} chunks.")
            if chunks:
                print(f"  -> Chunk 1 Section: '{chunks[0]['section']}', Length: {len(chunks[0]['text'])} chars")

    # 4. Test parsing & OCR on scanned PDF (Page 1 only for speed test)
    scanned_sample = "1. Peraturan Rektor Nomor 3 Tahun 2025 tentang Peraturan Akademik (1).pdf"
    if scanned_sample in pdf_files:
        print(f"\n[4/4] Testing Scanned PDF + OCR: {scanned_sample}")
        scanned_path = os.path.join(data_dir, scanned_sample)
        import fitz
        doc = fitz.open(scanned_path)
        print(f"  -> Total pages: {len(doc)}")
        first_page = doc[0]
        print(f"  -> Running OCR on Page 1...")
        ocr_result = pdf_parser._ocr_page(first_page)
        doc.close()
        print(f"  -> OCR Extracted {len(ocr_result)} chars.")
        print("  -> OCR Snippet:")
        print("  " + "-" * 40)
        for line in ocr_result.split("\n")[:8]:
            print(f"     {line}")
        print("  " + "-" * 40)

    print("\n=== TEST COMPLETED SUCCESSFULLY ===")

if __name__ == "__main__":
    run_test()
