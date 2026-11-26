import json
import logging
from typing import Annotated, Literal, Optional
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, StringConstraints
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.security import StudentIdentity, require_owner, require_student
from app.database.models import Feedback, Mahasiswa
from app.database.session import get_db
from app.rag.embeddings import embedding_service
from app.rag.generator import response_generator
from app.rag.guardrail import guardrail
from app.rag.vector_store import vector_store

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/chat", tags=["Chat"])
MessageText = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=4000)
]
Identifier = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=100)
]


class HistoryMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: Literal["user", "assistant"]
    content: MessageText


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: MessageText
    conversation_id: Optional[Identifier] = None
    nim: Optional[str] = Field(None, max_length=20)
    history: list[HistoryMessage] = Field(default_factory=list, max_length=10)


class FeedbackRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    message_id: Identifier
    conversation_id: Optional[Identifier] = None
    nim: Optional[str] = Field(None, max_length=20)
    rating: Literal["helpful", "not_helpful"]
    comment: Optional[str] = Field(None, max_length=2000)


def event(kind, **data):
    return f"data: {json.dumps({'type': kind, **data}, ensure_ascii=False)}\n\n"


def load_profile(db, nim):
    student = db.query(Mahasiswa).filter(Mahasiswa.nim == nim).first()
    if student:
        return {
            key: getattr(student, key)
            for key in (
                "nim",
                "nama",
                "prodi",
                "fakultas",
                "semester",
                "total_sks",
                "ipk",
                "mk_lulus",
            )
        }
    return None


def retrieve(query):
    vector = embedding_service.embed_query(query)
    try:
        raw_chunks = vector_store.query_similar(
            vector, top_k=8, query_text=query
        )
    except TypeError:
        raw_chunks = vector_store.query_similar(vector, top_k=5)
    return [
        chunk
        for chunk in raw_chunks
        if chunk.get("similarity", 0) >= settings.RETRIEVAL_MIN_SIMILARITY
    ]


@router.post("")
async def chat_stream(
    req: ChatRequest,
    identity: StudentIdentity = Depends(require_student),
    db: Session = Depends(get_db),
):
    if req.nim is not None:
        require_owner(req.nim, identity)
    # Resolve the profile before starting streaming; no session shared with the generator.
    profile = await run_in_threadpool(load_profile, db, identity.nim)
    conversation_id = (
        req.conversation_id if req.conversation_id is not None else str(uuid4())
    )

    async def sse_generator():
        yield event("meta", conversation_id=conversation_id)
        try:
            safe, message = guardrail.check_input(req.query)
            if safe:
                for previous in req.history:
                    safe, message = guardrail.check_input(previous.content)
                    if not safe:
                        break
            if not safe:
                yield event("token", content=message)
                return

            if not response_generator.api_key or not response_generator._client:
                yield event("meta", conversation_id=conversation_id, mode="unavailable")
                async for token in response_generator.generate_stream(""):
                    yield event("token", content=token)
                return

            chunks = await run_in_threadpool(retrieve, req.query)
            if (
                not chunks
                and not guardrail.is_campus_related(req.query)
                and not req.history
            ):
                yield event(
                    "token",
                    content="Maaf, tanyaUMRAH hanya menjawab pertanyaan seputar akademik dan layanan mahasiswa UMRAH.",
                )
                return
            prompt = response_generator.build_prompt(
                query=req.query,
                retrieved_chunks=chunks,
                student_profile=profile,
                chat_history=[item.model_dump() for item in req.history],
            )
            tokens = []
            async for token in response_generator.generate_stream(prompt):
                tokens.append(token)
                yield event("token", content=token)
            sources, seen = [], set()
            for chunk in chunks:
                meta = chunk.get("metadata", {})
                key = (
                    meta.get("document_name", "Dokumen UMRAH"),
                    meta.get("page", 1),
                    meta.get("section", "Umum"),
                )
                if key not in seen and len(sources) < 3:
                    seen.add(key)
                    source_item = dict(
                        zip(("document", "page", "section"), key),
                        similarity=chunk["similarity"],
                    )
                    if meta.get("document_id") is not None:
                        source_item["document_id"] = meta.get("document_id")
                    if chunk.get("content"):
                        source_item["excerpt"] = str(chunk.get("content", "")).strip()[:800]
                    sources.append(source_item)
            if sources:
                yield event("sources", data=sources)
            suggestions = await response_generator.generate_suggestions(
                req.query, "".join(tokens)
            )
            if suggestions:
                yield event("suggestions", data=suggestions)
        except Exception:
            logger.error("Chat processing failed")
            yield event(
                "error",
                content="Layanan jawaban sedang bermasalah. Silakan coba lagi nanti.",
            )
        finally:
            yield event("done")

    return StreamingResponse(
        sse_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/feedback")
def submit_feedback(
    fb: FeedbackRequest,
    identity: StudentIdentity = Depends(require_student),
    db: Session = Depends(get_db),
):
    if fb.nim is not None:
        require_owner(fb.nim, identity)
    if not db.query(Mahasiswa).filter(Mahasiswa.nim == identity.nim).first():
        raise HTTPException(404, "Lakukan /auth/verify terlebih dahulu.")
    record = Feedback(**fb.model_dump(exclude={"nim"}), nim=identity.nim)
    db.add(record)
    db.commit()
    return {"status": "success", "message": "Terima kasih atas penilaian Anda!"}
