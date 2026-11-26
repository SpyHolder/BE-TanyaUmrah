import logging
import math
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.core.config import settings

logger = logging.getLogger(__name__)

# Stopwords umum Bahasa Indonesia agar kata sambung tidak mengacaukan skor BM25
INDONESIAN_STOPWORDS = {
    "yang", "dan", "di", "ke", "dari", "ini", "itu", "untuk", "pada", "dengan",
    "adalah", "sebagai", "dalam", "atau", "jika", "bila", "apabila", "oleh",
    "dapat", "bisa", "akan", "telah", "sudah", "belum", "tidak", "bukan", "saya",
    "aku", "kamu", "anda", "kami", "kita", "mereka", "apa", "bagaimana", "kenapa",
    "mengapa", "siapa", "kapan", "dimana", "berapa", "saja", "hal", "tentang",
    "mengenai", "seputar", "cara", "mau", "ingin", "tanya", "tolong", "jelaskan",
    "sebutkan", "apakah", "ya", "dong", "kak", "min", "pak", "bu", "umrah",
    "universitas", "maritim", "raja", "ali", "haji", "mahasiswa", "kuliah",
    "kampus", "akademik",
}

# Kamus ekspansi sinonim istilah akademik UMRAH
ACADEMIC_SYNONYMS: Dict[str, List[str]] = {
    "cuti": ["cuti", "cuti akademik", "pengurusan cuti", "pasal 57", "berhenti sementara"],
    "krs": ["krs", "lirs", "lembar isian rencana studi", "kartu rencana studi", "rencana studi"],
    "lirs": ["lirs", "krs", "lembar isian rencana studi", "rencana studi"],
    "khs": ["khs", "lhsm", "kartu hasil studi", "lembar hasil studi"],
    "skripsi": ["skripsi", "tugas akhir", "karya ilmiah", "sidang", "seminar proposal"],
    "magang": ["magang", "kerja praktik", "kp", "mbkm", "merdeka belajar"],
    "kkn": ["kkn", "kukerta", "kuliah kerja nyata", "pengabdian"],
    "kukerta": ["kukerta", "kkn", "kuliah kerja nyata"],
    "ukt": ["ukt", "uang kuliah tunggal", "biaya kuliah", "spp", "pembayaran"],
    "spp": ["ukt", "uang kuliah tunggal", "biaya"],
    "do": ["berhenti studi", "putus studi", "drop out", "dikeluarkan", "batas masa studi"],
    "dropout": ["berhenti studi", "putus studi", "batas masa studi"],
    "mundur": ["berhenti studi", "mengundurkan diri", "pindah"],
    "pindah": ["pindah", "mutasi", "transfer", "alih kredit"],
    "ipk": ["ipk", "ips", "indeks prestasi", "predikat", "cumlaude", "pujian"],
    "cumlaude": ["pujian", "predikat", "kelulusan", "ipk"],
    "beasiswa": ["beasiswa", "penghargaan", "geliga gurindam", "bantuan"],
    "prestasi": ["penghargaan", "geliga gurindam", "prestasi", "kejuaraan"],
    "geliga": ["geliga gurindam", "penghargaan"],
    "etika": ["etika", "plagiarisme", "sanksi", "pelanggaran", "kode etik", "menyontek"],
    "nyontek": ["menyontek", "kecurangan", "plagiat", "sanksi", "etika"],
    "plagiat": ["plagiat", "plagiarisme", "karya ilmiah", "etika", "sanksi"],
    "sanksi": ["sanksi", "pelanggaran", "teguran", "skorsing", "pemberhentian"],
    "sks": ["sks", "satuan kredit semester", "beban belajar"],
    "nilai": ["nilai", "penilaian", "bobot", "indeks prestasi", "ujian", "remidi"],
    "ujian": ["ujian", "uts", "uas", "evaluasi", "susulan"],
    "absen": ["kehadiran", "presensi", "perkuliahan", "tatap muka"],
    "hadir": ["kehadiran", "presensi", "tatap muka"],
    "ijazah": ["ijazah", "transkrip", "wisuda", "kelulusan", "skpi"],
    "wisuda": ["wisuda", "yudisium", "kelulusan", "ijazah"],
    "yudisium": ["yudisium", "kelulusan", "wisuda"],
    "pa": ["penasihat akademik", "pembimbing akademik", "dosen wali", "konsultasi"],
    "wali": ["penasihat akademik", "pembimbing akademik", "dosen"],
    "sp": ["semester antara", "semester pendek", "perbaikan nilai"],
    "antara": ["semester antara", "semester pendek"],
    "registrasi": ["registrasi", "daftar ulang", "herregistrasi"],
}


def _extract_keywords(query_text: str) -> Tuple[List[str], List[str]]:
    """
    Mengekstrak kata kunci penting dan frasa sinonim dari query Bahasa Indonesia.
    Mengembalikan (core_tokens, expanded_phrases).
    """
    cleaned = re.sub(r"[^a-zA-Z0-9\s]", " ", (query_text or "").lower())
    raw_tokens = [t for t in cleaned.split() if len(t) >= 2]
    core_tokens = [t for t in raw_tokens if t not in INDONESIAN_STOPWORDS]
    if not core_tokens:
        core_tokens = raw_tokens

    expanded_phrases: List[str] = []
    for token in core_tokens:
        if token not in expanded_phrases:
            expanded_phrases.append(token)
        # Cek kamus sinonim akademik
        if token in ACADEMIC_SYNONYMS:
            for syn in ACADEMIC_SYNONYMS[token]:
                if syn not in expanded_phrases:
                    expanded_phrases.append(syn)
    return core_tokens, expanded_phrases


class VectorStore:
    def __init__(self, persist_dir: str = settings.CHROMA_PERSIST_DIR):
        self.persist_dir = persist_dir
        self.collection_name = "umrah_documents"
        self._client = None
        self._collection = None
        from threading import RLock

        self._lock = RLock()

    @property
    def client(self):
        with self._lock:
            if self._client is None:
                self._client = chromadb.PersistentClient(path=self.persist_dir)
            return self._client

    @property
    def collection(self):
        with self._lock:
            if self._collection is None:
                self._collection = self.client.get_or_create_collection(
                    name=self.collection_name, metadata={"hnsw:space": "cosine"}
                )
            return self._collection

    @collection.setter
    def collection(self, value):
        self._collection = value

    def add_chunks(
        self,
        ids: List[str],
        documents: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
    ):
        """
        Add document chunks and their embeddings to ChromaDB.
        """
        if not ids:
            return

        self.collection.add(
            ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas
        )
        logger.info(
            f"Added {len(ids)} chunks to ChromaDB collection '{self.collection_name}'."
        )

    def _compute_lexical_scores(
        self,
        query_text: str,
        all_docs: List[str],
        all_metas: List[Dict[str, Any]],
    ) -> List[float]:
        """
        Menghitung skor leksikal BM25 + Substring Match (tahan terhadap teks OCR yang menempel tanpa spasi)
        dan ekspansi sinonim akademik. Menghasilkan skor ternormalisasi [0.0, 1.0].
        """
        if not query_text or not all_docs:
            return [0.0] * len(all_docs)

        core_tokens, expanded_phrases = _extract_keywords(query_text)
        if not core_tokens and not expanded_phrases:
            return [0.0] * len(all_docs)

        n_docs = len(all_docs)
        lower_docs = [
            f"{(m or {}).get('document_name', '')} {(m or {}).get('section', '')} {d or ''}".lower()
            for d, m in zip(all_docs, all_metas)
        ]
        doc_lengths = [max(1, len(d)) for d in lower_docs]
        avg_dl = sum(doc_lengths) / max(1, n_docs)

        # Hitung Document Frequency (DF) untuk setiap istilah/frasa
        df_map: Dict[str, int] = {}
        for term in expanded_phrases:
            df = sum(1 for d in lower_docs if term in d)
            df_map[term] = df

        k1 = 1.5
        b = 0.75
        raw_scores = [0.0] * n_docs

        for idx, doc_text in enumerate(lower_docs):
            dl = doc_lengths[idx]
            score = 0.0
            matched_core = 0

            for token in core_tokens:
                if token in doc_text:
                    matched_core += 1

            for term in expanded_phrases:
                df = df_map.get(term, 0)
                if df == 0 or term not in doc_text:
                    continue
                tf = doc_text.count(term)
                idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
                tf_norm = (tf * (k1 + 1.0)) / (
                    tf + k1 * (1.0 - b + b * (dl / avg_dl))
                )
                # Beri bobot lebih tinggi jika istilah tersebut adalah kata inti dari user
                weight = 1.4 if term in core_tokens else 1.0
                if " " in term:
                    # Bonus untuk kecocokan frasa eksak (misal: "cuti akademik")
                    weight *= 1.35
                score += idf * tf_norm * weight

            # Bonus cakupan kata kunci inti (coverage bonus)
            if core_tokens and matched_core > 0:
                coverage = matched_core / len(core_tokens)
                score *= 1.0 + 0.5 * coverage

            # Penalti untuk halaman Daftar Isi (banyak titik-titik ".....") agar isi Pasal/SOP asli naik ke atas
            if "daftar isi" in doc_text or doc_text.count(".....") >= 2:
                score *= 0.25

            raw_scores[idx] = score

        max_score = max(raw_scores) if raw_scores else 0.0
        if max_score <= 0:
            return [0.0] * n_docs
        return [min(1.0, s / max_score) for s in raw_scores]

    def query_similar(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        where_filter: Optional[Dict[str, Any]] = None,
        query_text: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for top_k most similar document chunks.
        When query_text is provided, uses Hybrid Search (Vector Similarity + Local BM25/Synonym Reranking
        + Neighbor Chunk Context Expansion) so Indonesian academic queries work reliably offline.
        Returns a list of dicts with document, score, and metadata.
        """
        count = self.collection.count()
        if not count:
            return []

        # Jika query_text diberikan, ambil seluruh kandidat untuk di-rerank secara hybrid
        n_fetch = count if query_text else min(top_k, count)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_fetch,
            where=where_filter,
        )

        output = []
        if not results["documents"] or not results["documents"][0]:
            return output

        docs = results["documents"][0]
        metas = results["metadatas"][0] if results["metadatas"] else [{}] * len(docs)
        distances = (
            results["distances"][0] if results["distances"] else [1.0] * len(docs)
        )
        ids = results["ids"][0] if results["ids"] else [""] * len(docs)

        lexical_scores = (
            self._compute_lexical_scores(query_text, docs, metas)
            if query_text
            else [0.0] * len(docs)
        )
        has_any_lexical = any(s > 0 for s in lexical_scores)

        candidates = []
        chunk_lookup: Dict[str, Dict[str, Any]] = {}

        for doc_id, doc, meta, dist, lex_score in zip(
            ids, docs, metas, distances, lexical_scores
        ):
            vec_similarity = 1.0 - (dist if dist <= 1.0 else 1.0)
            if query_text and has_any_lexical:
                # Gabungkan skor vektor (35%) dan skor leksikal BM25 Bahasa Indonesia (65%)
                # Jika tidak ada kecocokan kata sama sekali, tetap gunakan vec_similarity tetapi sedikit direduksi
                if lex_score > 0:
                    hybrid_sim = 0.35 * vec_similarity + 0.65 * lex_score
                    similarity = max(vec_similarity * 0.85, hybrid_sim)
                else:
                    similarity = vec_similarity * 0.65
            else:
                similarity = vec_similarity

            item = {
                "id": doc_id,
                "content": doc,
                "metadata": meta,
                "similarity": round(min(1.0, similarity), 4),
                "distance": round(dist, 4),
                "_lex": lex_score,
            }
            candidates.append(item)
            if doc_id:
                chunk_lookup[doc_id] = item

        if query_text and has_any_lexical:
            candidates.sort(key=lambda x: (x["similarity"], x["_lex"]), reverse=True)

        # Pilih top kandidat dan lakukan Neighbor Chunk Expansion agar tabel SOP / ayat Pasal tidak terpotong
        selected: List[Dict[str, Any]] = []
        seen_ids = set()

        for item in candidates:
            if len(selected) >= top_k:
                break
            c_id = item["id"]
            if c_id in seen_ids:
                continue
            seen_ids.add(c_id)
            selected.append(item)

            # Jika chunk ini memiliki kecocokan kata kunci yang kuat, tarik juga chunk berikutnya (+1)
            # pada dokumen yang sama agar kelanjutan kalimat/tabel ikut terbawa
            if query_text and item.get("_lex", 0) >= 0.45 and len(selected) < top_k:
                m = re.match(r"^(doc_\d+_chunk_)(\d+)$", c_id or "")
                if m:
                    prefix, c_num = m.group(1), int(m.group(2))
                    next_id = f"{prefix}{c_num + 1}"
                    neighbor = chunk_lookup.get(next_id)
                    if (
                        neighbor
                        and next_id not in seen_ids
                        and neighbor["metadata"].get("document_id")
                        == item["metadata"].get("document_id")
                    ):
                        seen_ids.add(next_id)
                        # Pastikan similarity tetangga cukup untuk melewati threshold filter
                        boosted_neighbor = dict(neighbor)
                        boosted_neighbor["similarity"] = round(
                            max(neighbor["similarity"], item["similarity"] * 0.92), 4
                        )
                        selected.append(boosted_neighbor)

        for item in selected:
            item.pop("_lex", None)
            output.append(item)

        return output

    def delete_by_document_id(self, document_id: int):
        """
        Delete all chunks associated with a specific document ID.
        """
        self.collection.delete(where={"document_id": document_id})
        logger.info(f"Deleted chunks for document_id={document_id} from ChromaDB.")

    def reset_collection(self):
        """
        Deletes and recreates the collection with fresh embedding configuration.
        """
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name, metadata={"hnsw:space": "cosine"}
        )
        logger.info(f"Reset ChromaDB collection '{self.collection_name}'.")

    def count_chunks(self) -> int:
        return self.collection.count()


vector_store = VectorStore()
