import json
import logging
from typing import List, Dict, Any, Optional, AsyncGenerator
from app.core.config import settings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """Anda adalah tanyaUMRAH, asisten kecerdasan buatan resmi untuk layanan akademik dan kemahasiswaan di Universitas Maritim Raja Ali Haji (UMRAH).

TUGAS ANDA:
1. Menjawab pertanyaan mahasiswa dengan ramah, jelas, sopan, dan terstruktur menggunakan Bahasa Indonesia yang baik.
2. JAWAB HANYA BERDASARKAN DOKUMEN RESMI UMRAH yang diberikan dalam [KONTEKS DOKUMEN].
3. Jika informasi TIDAK DITEMUKAN dalam dokumen yang diberikan, sampaikan secara jujur bahwa Anda belum memiliki informasi resmi mengenai hal tersebut dan sarankan mahasiswa untuk menghubungi fakultas atau Biro Akademik, Perencanaan, Kemahasiswaan, dan Kerjasama (BAPKK) UMRAH. JANGAN MENGARANG JAWABAN.
4. Jika terdapat data profil mahasiswa dalam [PROFIL MAHASISWA], gunakan data tersebut secara personal (misalnya memeriksa kecukupan SKS, IPK, semester, atau prasyarat untuk mengambil mata kuliah/magang/skripsi).
5. Pada akhir penjelasan, sebutkan pasal atau pedoman rujukan yang relevan.
"""

class ResponseGenerator:
    def __init__(self):
        self.api_key = settings.GEMINI_API_KEY
        self.model_name = settings.GEMINI_MODEL
        self._model = None
        
        if self.api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self._model = genai.GenerativeModel(
                    model_name=self.model_name,
                    system_instruction=SYSTEM_PROMPT
                )
            except Exception as e:
                logger.warning(f"Failed to initialize Gemini GenerativeModel: {e}")

    def build_prompt(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        student_profile: Optional[Dict[str, Any]] = None,
        chat_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        prompt_parts = []

        # 1. Student context (if available)
        if student_profile:
            prompt_parts.append("[PROFIL MAHASISWA PENANYA]")
            prompt_parts.append(f"- NIM: {student_profile.get('nim', '-')}")
            prompt_parts.append(f"- Nama: {student_profile.get('nama', '-')}")
            prompt_parts.append(f"- Program Studi: {student_profile.get('prodi', '-')}")
            prompt_parts.append(f"- Fakultas: {student_profile.get('fakultas', '-')}")
            prompt_parts.append(f"- Semester Saat Ini: {student_profile.get('semester', '-')}")
            prompt_parts.append(f"- Total SKS Lulus: {student_profile.get('total_sks', 0)} SKS")
            prompt_parts.append(f"- IPK: {student_profile.get('ipk', 0.0)}")
            mk_lulus = student_profile.get('mk_lulus', [])
            if mk_lulus:
                prompt_parts.append(f"- Mata Kuliah yang Sudah Lulus: {', '.join(mk_lulus)}")
            prompt_parts.append("")

        # 2. Context from documents
        prompt_parts.append("[KONTEKS DOKUMEN RESMI UMRAH]")
        if retrieved_chunks:
            for idx, chunk in enumerate(retrieved_chunks, 1):
                meta = chunk.get("metadata", {})
                doc_name = meta.get("document_name", "Dokumen UMRAH")
                page = meta.get("page", "-")
                section = meta.get("section", "-")
                prompt_parts.append(f"--- Potongan {idx} (Sumber: {doc_name}, Hal. {page}, Bagian: {section}) ---")
                prompt_parts.append(chunk.get("content", ""))
                prompt_parts.append("")
        else:
            prompt_parts.append("(Tidak ada potongan dokumen yang cocok ditemukan)")
            prompt_parts.append("")

        # 3. User Query
        prompt_parts.append("[PERTANYAAN MAHASISWA]")
        prompt_parts.append(query)
        prompt_parts.append("")
        prompt_parts.append("Berikan jawaban yang jelas, akurat, dan cantumkan rujukan pasal/halamannya.")

        return "\n".join(prompt_parts)

    async def generate_stream(
        self,
        prompt: str
    ) -> AsyncGenerator[str, None]:
        """
        Yields tokens for SSE streaming.
        """
        if not self.api_key or not self._model:
            # Fallback mockup response for offline testing without API key
            fallback_text = (
                "Ini adalah mode respon uji coba lokal (API Key Gemini belum diatur).\n\n"
                "Sistem RAG berhasil mengaitkan dokumen resmi UMRAH dengan pertanyaan Anda. "
                "Untuk mengaktifkan jawaban AI lengkap, silakan atur GEMINI_API_KEY di file .env backend."
            )
            for word in fallback_text.split(" "):
                yield word + " "
            return

        try:
            response = self._model.generate_content(prompt, stream=True)
            for chunk in response:
                if chunk.text:
                    yield chunk.text
        except Exception as e:
            logger.error(f"Error during Gemini generation: {e}")
            yield f"\n[Terjadi kesalahan saat memproses jawaban: {str(e)}]"

    def generate_suggestions(self, query: str, answer_snippet: str) -> List[str]:
        """
        Generate 2-3 relevant follow-up questions.
        """
        if not self.api_key or not self._model:
            return [
                "Bagaimana prosedur pendaftaran KRS?",
                "Berapa batas SKS yang boleh diambil?",
                "Kapan batas pengajuan cuti semester?"
            ]

        try:
            suggestion_prompt = (
                f"Berdasarkan pertanyaan: '{query}' dan rangkuman jawaban: '{answer_snippet[:200]}', "
                "berikan 3 pertanyaan lanjutan yang relevan dan singkat (maksimal 8 kata per pertanyaan). "
                "Output HANYA dalam format JSON array of strings, contoh: [\"Pertanyaan 1?\", \"Pertanyaan 2?\"]"
            )
            res = self._model.generate_content(suggestion_prompt)
            clean_res = res.text.strip()
            if "```json" in clean_res:
                clean_res = clean_res.split("```json")[1].split("```")[0].strip()
            elif "```" in clean_res:
                clean_res = clean_res.split("```")[1].split("```")[0].strip()
            suggestions = json.loads(clean_res)
            return suggestions if isinstance(suggestions, list) else []
        except Exception:
            return [
                "Bagaimana alur prosedurnya?",
                "Apa saja syarat dokumen yang diperlukan?",
                "Siapa unit/staf yang harus dihubungi?"
            ]

response_generator = ResponseGenerator()
