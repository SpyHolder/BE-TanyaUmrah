import logging
import re
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)


class Guardrail:
    # Profanity / toxic words filter (Indonesian + English)
    INAPPROPRIATE_WORDS = [
        "anjing",
        "babi",
        "bangsat",
        "kontol",
        "memek",
        "pantek",
        "ngentot",
        "tolol",
        "goblok",
        "fuck",
        "shit",
        "bitch",
        "asshole",
    ]

    # Prompt injection patterns
    INJECTION_PATTERNS = [
        r"ignore (all|previous|above) instructions",
        r"abaikan (semua|instruksi|perintah) sebelumnya",
        r"system prompt",
        r"you are now (unrestricted|jailbroken|dan)",
        r"tampilkan (prompt|instruksi rahasia)",
        r"mode developer",
        r"bypass",
    ]

    # Broad campus / academic keywords to identify relevance
    CAMPUS_KEYWORDS = [
        "umrah",
        "kuliah",
        "krs",
        "khs",
        "sks",
        "ipk",
        "ips",
        "cuti",
        "magang",
        "mbkm",
        "skripsi",
        "tugas akhir",
        "yudisium",
        "wisuda",
        "ukt",
        "biaya",
        "beasiswa",
        "dosen",
        "pembimbing",
        "fakultas",
        "prodi",
        "jurusan",
        "semester",
        "ujian",
        "uts",
        "uas",
        "remedial",
        "sp",
        "kurikulum",
        "mk",
        "mata kuliah",
        "prasyarat",
        "rektor",
        "bapkk",
        "geliga gurindam",
        "etika",
        "sanksi",
        "plagiarisme",
        "surat",
        "ijazah",
        "transkrip",
        "akademik",
        "kemahasiswaan",
        "alumni",
        "organisasi",
        "bem",
        "perpustakaan",
        "registrasi",
        "herregistrasi",
        "pembelajaran",
    ]

    def check_input(self, query: str) -> Tuple[bool, str]:
        """
        Layer 1 & 2 Guardrail:
        Checks for inappropriate content, prompt injection, and domain relevance.
        """
        lower_query = query.lower()

        # Check toxic/abusive words
        for word in self.INAPPROPRIATE_WORDS:
            if re.search(r"\b" + re.escape(word) + r"\b", lower_query):
                return (
                    False,
                    "Pertanyaan mengandung kata-kata yang tidak pantas. Mohon gunakan bahasa yang sopan.",
                )

        # Check prompt injection
        for pattern in self.INJECTION_PATTERNS:
            if re.search(pattern, lower_query):
                return (
                    False,
                    "Maaf, permintaan tidak dapat diproses demi keamanan sistem.",
                )

        return True, ""

    def is_campus_related(self, query: str) -> bool:
        """
        Checks whether the query is related to academic or university activities.
        """
        lower_query = query.lower()
        for kw in self.CAMPUS_KEYWORDS:
            if re.search(r"\b" + re.escape(kw) + r"\b", lower_query):
                return True
        return False

    def check_retrieval(
        self, retrieved_chunks: List[Dict[str, Any]], threshold: float = 0.30
    ) -> bool:
        """
        Layer 3: Check if retrieved chunks have sufficient relevance score.
        """
        if not retrieved_chunks:
            return False

        top_similarity = retrieved_chunks[0].get("similarity", 0.0)
        return top_similarity >= threshold


guardrail = Guardrail()
