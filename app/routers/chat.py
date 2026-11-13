import json
import logging
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Header
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.database.session import get_db
from app.database.models import Mahasiswa, Feedback
from app.rag.guardrail import guardrail
from app.rag.embeddings import embedding_service
from app.rag.vector_store import vector_store
from app.rag.generator import response_generator

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["Chat"])

class ChatRequest(BaseModel):
    query: str
    conversation_id: Optional[str] = None
    nim: Optional[str] = None  # Mahasiswa NIM if logged in

class FeedbackRequest(BaseModel):
    message_id: str
    conversation_id: Optional[str] = None
    nim: Optional[str] = None
    rating: str  # 'helpful' or 'not_helpful'
    comment: Optional[str] = None

@router.post("")
async def chat_stream(
    req: ChatRequest,
    db: Session = Depends(get_db)
):
    """
    RAG Chat endpoint with SSE (Server-Sent Events) streaming.
    """
    query = req.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Pertanyaan tidak boleh kosong.")

    # 1. Guardrail Layer 1: Inappropriate words & prompt injection
    is_safe, error_msg = guardrail.check_input(query)
    if not is_safe:
        async def guardrail_error_stream():
            yield f'data: {json.dumps({"type": "token", "content": error_msg})}\n\n'
            yield f'data: {json.dumps({"type": "done"})}\n\n'
        return StreamingResponse(guardrail_error_stream(), media_type="text/event-stream")

    # 2. Check student profile (if student NIM provided)
    student_profile = None
    if req.nim:
        mhs = db.query(Mahasiswa).filter(Mahasiswa.nim == req.nim).first()
        if mhs:
            student_profile = {
                "nim": mhs.nim,
                "nama": mhs.nama,
                "prodi": mhs.prodi,
                "fakultas": mhs.fakultas,
                "semester": mhs.semester,
                "total_sks": mhs.total_sks,
                "ipk": mhs.ipk,
                "mk_lulus": mhs.mk_lulus or []
            }

    # 3. Retrieve relevant chunks from ChromaDB
    try:
        query_vector = embedding_service.embed_query(query)
        retrieved_chunks = vector_store.query_similar(query_vector, top_k=5)
    except Exception as e:
        logger.error(f"Error retrieving chunks: {e}")
        retrieved_chunks = []

    # 4. Guardrail Layer 2 & 3: Off-topic / Retrieval check
    is_relevant = guardrail.check_retrieval(retrieved_chunks, threshold=0.25)
    is_campus = guardrail.is_campus_related(query)

    if not is_relevant and not is_campus:
        async def off_topic_stream():
            msg = "Maaf, saya adalah asisten akademik tanyaUMRAH dan hanya dapat menjawab pertanyaan seputar perkuliahan, peraturan, dan layanan mahasiswa UMRAH."
            yield f'data: {json.dumps({"type": "token", "content": msg})}\n\n'
            yield f'data: {json.dumps({"type": "done"})}\n\n'
        return StreamingResponse(off_topic_stream(), media_type="text/event-stream")

    # Format source cards
    sources = []
    seen_sources = set()
    for chunk in retrieved_chunks:
        meta = chunk.get("metadata", {})
        doc_name = meta.get("document_name", "Dokumen UMRAH")
        page = meta.get("page", 1)
        section = meta.get("section", "Umum")
        key = (doc_name, page, section)
        if key not in seen_sources and len(sources) < 3:
            seen_sources.add(key)
            sources.append({
                "document": doc_name,
                "page": page,
                "section": section,
                "similarity": chunk.get("similarity", 0.0)
            })

    # 5. Build prompt & generate streaming response
    prompt = response_generator.build_prompt(
        query=query,
        retrieved_chunks=retrieved_chunks,
        student_profile=student_profile
    )

    async def sse_generator():
        collected_tokens = []
        try:
            async for token in response_generator.generate_stream(prompt):
                collected_tokens.append(token)
                yield f'data: {json.dumps({"type": "token", "content": token})}\n\n'

            # Send source cards
            if sources:
                yield f'data: {json.dumps({"type": "sources", "data": sources})}\n\n'

            # Send suggested follow-up questions
            full_answer = "".join(collected_tokens)
            suggestions = response_generator.generate_suggestions(query, full_answer)
            if suggestions:
                yield f'data: {json.dumps({"type": "suggestions", "data": suggestions})}\n\n'

            # Signal completion
            yield f'data: {json.dumps({"type": "done"})}\n\n'

        except Exception as err:
            logger.error(f"Error in SSE generator: {err}")
            err_msg = f"\nTerjadi kesalahan: {str(err)}"
            yield f'data: {json.dumps({"type": "token", "content": err_msg})}\n\n'
            yield f'data: {json.dumps({"type": "done"})}\n\n'

    return StreamingResponse(sse_generator(), media_type="text/event-stream")

@router.post("/feedback")
def submit_feedback(
    fb: FeedbackRequest,
    db: Session = Depends(get_db)
):
    """
    Submit user feedback (helpful / not_helpful) for an AI response.
    """
    record = Feedback(
        nim=fb.nim,
        conversation_id=fb.conversation_id,
        message_id=fb.message_id,
        rating=fb.rating,
        comment=fb.comment
    )
    db.add(record)
    db.commit()
    return {"status": "success", "message": "Terima kasih atas penilaian Anda!"}
