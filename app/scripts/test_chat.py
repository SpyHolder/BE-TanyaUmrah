import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.rag.embeddings import embedding_service
from app.rag.vector_store import vector_store
from app.rag.generator import response_generator

def test_query(query: str):
    print("=" * 65)
    print(f"❓ PERTANYAAN: {query}")
    print("=" * 65)

    # 1. Similarity search
    q_vec = embedding_service.embed_query(query)
    chunks = vector_store.query_similar(q_vec, top_k=3)
    
    print(f"\n📚 Ditemukan {len(chunks)} Chunk Relevan:")
    for idx, c in enumerate(chunks, 1):
        meta = c["metadata"]
        print(f"  {idx}. [{meta.get('document_name')}] Halaman {meta.get('page')} (Bagian: {meta.get('section')})")
        print(f"     Skor Kemiripan: {c.get('similarity')}")
        print(f"     Cuplikan Teks : {c.get('content')[:120].strip()}...\n")

    # 2. Build prompt & generate response
    student_mock = {
        "nim": "2401020163",
        "nama": "Saif Alif Abyam",
        "prodi": "Teknik Informatika",
        "fakultas": "Teknik dan Ilmu Kelautan",
        "semester": 5,
        "total_sks": 95,
        "ipk": 3.45
    }

    prompt = response_generator.build_prompt(query, chunks, student_profile=student_mock)
    
    from google.genai import types
    config = types.GenerateContentConfig(temperature=0.3)
    
    print("🤖 JAWABAN tanyaUMRAH:")
    print("-" * 65)
    resp = response_generator._client.models.generate_content(
        model=response_generator.fallback_model,
        contents=prompt,
        config=config
    )
    print(resp.text.strip())
    print("-" * 65)

    # 3. Suggestions
    suggestions = response_generator.generate_suggestions(query, resp.text.strip())
    print("\n💡 Saran Pertanyaan Lanjutan:")
    for s in suggestions:
        print(f"  - {s}")
    print("=" * 65)

if __name__ == "__main__":
    test_query("Bagaimana prosedur dan syarat pengajuan cuti akademik di UMRAH?")
