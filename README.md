# tanyaUMRAH Backend API

Backend API dan Retrieval-Augmented Generation (RAG) Engine untuk aplikasi **tanyaUMRAH** (Chatbot Cerdas Berbasis Dokumen Resmi Kampus Universitas Maritim Raja Ali Haji).

Dikembangkan sebagai Tugas Akhir Mata Kuliah **Pemrograman Perangkat Bergerak (PPB)**  
Program Studi Teknik Informatika, Fakultas Teknik dan Teknologi Kemaritiman, Universitas Maritim Raja Ali Haji (UMRAH).  
**Dosen Pengampu:** Marisha Pertiwi, S.Tr.Kom, M.Kom.

---

## 🚀 Fitur Utama

- **RAG Engine Pintar:** Menggunakan ChromaDB vector store dengan embedding Gemini (`gemini-embedding-001`) dan reranking cerdas (BM25 + sinonim akademik kampus).
- **LLM Generator Berbasis Konteks:** Menghasilkan jawaban akurat, kutipan pasal/halaman resmi, dan rekomendasi personalisasi mahasiswa menggunakan `gemini-3.5-flash-lite` (fallback: `gemini-3.1-flash-lite`).
- **Streaming Response:** Mendukung Server-Sent Events (SSE) untuk efek teks mengetik real-time di aplikasi mobile Flutter.
- **Saran Pertanyaan Lanjutan:** Menghasilkan 3 pertanyaan relevan otomatis setelah setiap jawaban bot.
- **Multi-Dokumen Parser & RapidOCR:** Ekstraksi teks digital (PyMuPDF) dan pindaian/scanned OCR (RapidOCR ONNX) untuk dokumen peraturan bertanda tangan.
- **Web Admin Panel:** Panel kelola dokumen, kalender akademik, dan data profil mahasiswa berbasis Jinja2 + TailwindCSS.
- **Autentikasi Firebase & Role-Based Access:** Verifikasi ID Token akun mahasiswa Google Workspace UMRAH (`NIM@student.umrah.ac.id`) dan Basic Auth untuk administrator.

---

## 🛠️ Tech Stack

- **Framework:** FastAPI (Python 3.10+)
- **Vector Database:** ChromaDB
- **Database Relasional:** SQLite + SQLAlchemy ORM
- **OCR Engine:** RapidOCR ONNX Runtime + PyMuPDF
- **LLM & Embeddings:** Google GenAI SDK (`google-genai`)
- **Web Admin:** Jinja2 + TailwindCSS

---

## 📁 Struktur Direktori

```text
BE-TanyaUmrah/
├── app/
│   ├── core/           # Konfigurasi, keamanan, upload validator
│   ├── db/             # SQLAlchemy engine & model database
│   ├── ingestion/      # PDF parser, OCR, chunker, & pipeline
│   ├── rag/            # Embeddings, ChromaDB store, generator, & guardrails
│   ├── routers/        # Endpoint API (auth, chat, calendar, documents, admin)
│   ├── scripts/        # Skrip uji RAG & evaluasi
│   ├── templates/      # Template antarmuka Web Admin (Jinja2)
│   └── main.py         # Entrypoint aplikasi FastAPI
├── data/               # Dokumen PDF pedoman & peraturan resmi UMRAH
├── tests/              # Unit test keamanan & fungsionalitas backend
├── storage/            # Tempat upload dokumen & cache teks
├── .env.example        # Contoh variabel lingkungan
├── .gitignore          # File pengecualian Git
├── requirements.txt    # Daftar dependensi Python
└── run_backend.bat     # Launcher satu klik server backend
```

---

## ⚙️ Cara Instalasi & Menjalankan

### 1. Prasyarat
- Python 3.10 atau lebih baru
- Google Gemini API Key

### 2. Setup Virtual Environment
```bash
python -m venv venv
# Windows:
.\venv\Scripts\activate
```

### 3. Install Dependensi
```bash
pip install -r requirements.txt
```

### 4. Konfigurasi Lingkungan (.env)
Salin `.env.example` menjadi `.env` lalu sesuaikan isinya:
```bash
cp .env.example .env
```
Isi konfigurasi penting:
- `GEMINI_API_KEY`: API key dari Google AI Studio
- `ADMIN_USERNAME`: Username admin panel
- `ADMIN_PASSWORD`: Password admin panel

### 5. Menjalankan Server
Cukup klik ganda file `run_backend.bat`, atau jalankan via terminal:
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

- **Swagger UI API Docs:** `http://localhost:8000/docs`
- **Web Admin Panel:** `http://localhost:8000/admin`
- **Health Check:** `http://localhost:8000/health`

---

## 🧪 Menjalankan Uji Coba (Tests)

Jalankan test suite untuk memastikan seluruh keamanan akses dan endpoint berfungsi normal:
```bash
python -m unittest discover -s tests
```

Untuk menguji alur RAG secara langsung di terminal:
```bash
python app/scripts/test_chat.py
```

---

## 📄 Dokumen Knowledge Base

Sistem ini telah memuat 5 dokumen resmi Universitas Maritim Raja Ali Haji:
1. **Peraturan Rektor No. 3 Tahun 2025** tentang Peraturan Akademik
2. **Pedoman Penyelenggaraan Pembelajaran 2026**
3. **Peraturan Rektor No. 2 Tahun 2024** tentang Penghargaan Geliga Gurindam
4. **Peraturan Rektor No. 5** tentang Pedoman Etika Akademik UMRAH
5. **SOP Akademik BAPKK UMRAH**
