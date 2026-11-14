@echo off
title tanyaUMRAH Backend Server
echo ============================================================
echo   Menjalankan tanyaUMRAH Backend (FastAPI + ChromaDB)
echo ============================================================
echo.
cd /d "%~dp0"

echo Mengaktifkan virtual environment...
if not exist "venv\Scripts\uvicorn.exe" (
    echo [ERROR] Virtual environment tidak ditemukan!
    pause
    exit /b
)

echo.
echo Server berjalan di:
echo - Web Admin Panel : http://localhost:8000/admin
echo - API Docs (Swagger): http://localhost:8000/docs
echo - Android Emulator  : http://10.0.2.2:8000/api/v1
echo.
echo Tekan CTRL + C untuk menghentikan server.
echo ============================================================
echo.

.\venv\Scripts\uvicorn.exe app.main:app --reload --host 0.0.0.0 --port 8000
pause
