"""Offline regression suite: in-memory SQL, temporary files, fake identity/AI/vector store."""

import asyncio
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pymupdf
from app.core.config import settings
from app.core.security import FirebaseVerifier, firebase_verifier
from app.core.uploads import save_pdf_upload
from app.database.models import AcademicEvent, Dokumen, Feedback, Mahasiswa
from app.database.session import Base, get_db
from app.ingestion.chunker import SmartChunker
from app.ingestion.parser import PDFParser
from app.ingestion.pipeline import IngestionPipeline
from app.main import app
from app.rag.embeddings import EmbeddingService
from app.rag.generator import ResponseGenerator
from app.rag.guardrail import guardrail
from app.routers import admin, chat
from fastapi import HTTPException
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from starlette.datastructures import Headers, UploadFile


def claims(**changes):
    return {
        "uid": "firebase-user-1",
        "email": "2401020163@student.umrah.ac.id",
        "email_verified": True,
        "name": "Verified Student",
        "picture": "https://example.test/photo",
        "firebase": {"sign_in_provider": "google.com"},
        **changes,
    }


def sse(response):
    return [
        json.loads(line[6:])
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]


class APITests(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine(
            "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
        )
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

        def db_override():
            with self.Session() as db:
                yield db

        app.dependency_overrides[get_db] = db_override
        # Do not enter TestClient lifespan, which initializes the real deployment DB.
        self.client = TestClient(app)
        self.verifier = patch.object(
            firebase_verifier, "verify", return_value=claims()
        ).start()
        self.addCleanup(patch.stopall)
        self.headers = {"Authorization": "Bearer fake-firebase-token"}

    def tearDown(self):
        self.client.close()
        app.dependency_overrides.clear()
        self.engine.dispose()

    def verify(self):
        response = self.client.post(
            "/api/v1/auth/verify", json={"id_token": "fake-firebase-token"}
        )
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def test_verify_uses_only_verified_claims_and_header(self):
        body = self.verify()
        self.assertEqual(body["profile"]["nama"], "Verified Student")
        self.assertEqual(body["profile"]["nim"], "2401020163")
        self.assertEqual(body["profile"]["photo_url"], "https://example.test/photo")
        response = self.client.post("/api/v1/auth/verify", headers=self.headers)
        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.json()["is_new_user"])
        self.assertEqual(
            self.client.post(
                "/api/v1/auth/verify",
                json={"id_token": "fake", "email": "forged@student.umrah.ac.id"},
            ).status_code,
            422,
        )
        self.assertEqual(
            self.client.post(
                "/api/v1/auth/verify",
                json={"id_token": "different"},
                headers=self.headers,
            ).status_code,
            400,
        )

    def test_exact_domain_verified_google_numeric_local_part(self):
        denied = [
            claims(email="2401020163@umrah.ac.id"),
            claims(email="2401020163@student.umrah.ac.id.evil.test"),
            claims(email="2401020163@evilstudent.umrah.ac.id"),
            claims(email="2401020163abc@student.umrah.ac.id"),
            claims(email="name@student.umrah.ac.id"),
            claims(email_verified=False),
            claims(firebase={"sign_in_provider": "password"}),
            claims(firebase={"sign_in_provider": "anonymous"}),
        ]
        for value in denied:
            with self.subTest(claims=value):
                self.verifier.return_value = value
                self.assertEqual(
                    self.client.post(
                        "/api/v1/auth/verify", json={"id_token": "fake"}
                    ).status_code,
                    403,
                )
        with self.Session() as db:
            self.assertEqual(db.query(Mahasiswa).count(), 0)

    def test_missing_invalid_tokens_and_fail_closed_config(self):
        self.assertEqual(self.client.post("/api/v1/auth/verify").status_code, 401)
        for path in (
            "/api/v1/user/profile/2401020163",
            "/api/v1/documents",
            "/api/v1/documents/1/file",
            "/api/v1/calendar/events",
        ):
            self.assertEqual(self.client.get(path).status_code, 401, path)
        self.assertEqual(
            self.client.post("/api/v1/chat", json={"query": "KRS?"}).status_code, 401
        )
        self.assertEqual(
            self.client.post(
                "/api/v1/chat/feedback", json={"message_id": "m", "rating": "helpful"}
            ).status_code,
            401,
        )
        self.verifier.side_effect = HTTPException(401, "invalid token")
        self.assertEqual(
            self.client.get("/api/v1/documents", headers=self.headers).status_code, 401
        )
        with (
            patch.object(settings, "FIREBASE_PROJECT_ID", ""),
            patch.object(settings, "FIREBASE_CREDENTIALS_PATH", ""),
            patch.object(settings, "GOOGLE_APPLICATION_CREDENTIALS", ""),
        ):
            with self.assertRaises(HTTPException) as error:
                FirebaseVerifier().verify("any")
            self.assertEqual(error.exception.status_code, 503)
            self.assertIn("FIREBASE_PROJECT_ID", error.exception.detail)

    def test_ownership_and_profile_validation(self):
        self.verify()
        own = "/api/v1/user/profile/2401020163"
        other = "/api/v1/user/profile/999"
        self.assertEqual(self.client.get(own, headers=self.headers).status_code, 200)
        self.assertEqual(self.client.get(other, headers=self.headers).status_code, 403)
        self.assertEqual(
            self.client.put(other, headers=self.headers, json={"ipk": 4}).status_code,
            403,
        )
        for body in (
            {"ipk": 4.1},
            {"ipk": -1},
            {"semester": 0},
            {"semester": 31},
            {"total_sks": -1},
            {"total_sks": 301},
            {"nim": "999"},
        ):
            self.assertEqual(
                self.client.put(own, headers=self.headers, json=body).status_code, 422
            )
        response = self.client.put(
            own,
            headers=self.headers,
            json={
                "semester": 5,
                "total_sks": 95,
                "ipk": 3.45,
                "mk_lulus": ["Algoritma"],
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["profile"]["ipk"], 3.45)

    def test_feedback_nim_is_derived_and_rating_enum(self):
        self.verify()
        body = {"message_id": "m", "conversation_id": "c", "rating": "helpful"}
        self.assertEqual(
            self.client.post(
                "/api/v1/chat/feedback",
                headers=self.headers,
                json={**body, "nim": "999"},
            ).status_code,
            403,
        )
        self.assertEqual(
            self.client.post(
                "/api/v1/chat/feedback",
                headers=self.headers,
                json={**body, "rating": "other"},
            ).status_code,
            422,
        )
        self.assertEqual(
            self.client.post(
                "/api/v1/chat/feedback", headers=self.headers, json=body
            ).status_code,
            200,
        )
        with self.Session() as db:
            self.assertEqual(db.query(Feedback).one().nim, "2401020163")

    def test_all_admin_routes_are_guarded_and_calendar_mutation(self):
        with (
            patch.object(settings, "ADMIN_USERNAME", ""),
            patch.object(settings, "ADMIN_PASSWORD", ""),
        ):
            for route in admin.router.routes:
                path = "/admin" + route.path
                path = path.replace("{document_id}", "1").replace("{event_id}", "1")
                method = "GET" if "GET" in route.methods else "POST"
                self.assertEqual(
                    self.client.request(method, path).status_code, 503, path
                )
            self.assertEqual(
                self.client.post(
                    "/api/v1/calendar/events",
                    json={"title": "T", "start_date": "2026-01-01"},
                ).status_code,
                503,
            )
        with (
            patch.object(settings, "ADMIN_USERNAME", "test-admin"),
            patch.object(settings, "ADMIN_PASSWORD", "strong-test-password"),
        ):
            self.assertEqual(self.client.get("/admin/students").status_code, 401)
            self.assertEqual(
                self.client.get(
                    "/admin/students", auth=("test-admin", "wrong")
                ).status_code,
                401,
            )
            self.assertEqual(
                self.client.get(
                    "/admin/students", auth=("test-admin", "strong-test-password")
                ).status_code,
                200,
            )
            self.assertEqual(
                self.client.post(
                    "/api/v1/calendar/events",
                    headers=self.headers,
                    json={"title": "T", "start_date": "2026-01-01"},
                ).status_code,
                401,
            )
            auth = ("test-admin", "strong-test-password")
            for body in (
                {"title": "T", "start_date": "2026-02-30"},
                {"title": "T", "start_date": "2026-02-02", "end_date": "2026-01-01"},
                {"title": "T", "start_date": "2026-01-01", "category": "invalid"},
            ):
                self.assertEqual(
                    self.client.post(
                        "/api/v1/calendar/events", auth=auth, json=body
                    ).status_code,
                    422,
                )
            self.assertEqual(
                self.client.post(
                    "/api/v1/calendar/events",
                    auth=auth,
                    json={"title": "T", "start_date": "2026-01-01"},
                ).status_code,
                200,
            )

    def test_approved_calendar_overlap(self):
        with self.Session() as db:
            db.add_all(
                [
                    AcademicEvent(
                        title="Spanning",
                        start_date="2026-10-25",
                        end_date="2026-12-01",
                        is_approved=True,
                    ),
                    AcademicEvent(
                        title="Hidden", start_date="2026-11-01", is_approved=False
                    ),
                    AcademicEvent(
                        title="Old", start_date="2025-11-01", is_approved=True
                    ),
                ]
            )
            db.commit()
        body = self.client.get(
            "/api/v1/calendar/events?month=11&year=2026", headers=self.headers
        ).json()
        self.assertEqual([ev["title"] for ev in body["events"]], ["Spanning"])
        body = self.client.get(
            "/api/v1/calendar/events?month=11", headers=self.headers
        ).json()
        self.assertEqual(body["total"], 2)

    def test_chat_ownership_history_bounds_and_honest_unavailable(self):
        self.verify()
        with (
            patch.object(chat.response_generator, "api_key", ""),
            patch.object(chat.embedding_service, "embed_query") as embed,
        ):
            response = self.client.post(
                "/api/v1/chat",
                headers=self.headers,
                json={"query": "KRS?", "conversation_id": "existing-id"},
            )
            events = sse(response)
            self.assertEqual(
                events[0], {"type": "meta", "conversation_id": "existing-id"}
            )
            self.assertEqual(events[-1]["type"], "done")
            self.assertIn(
                "belum tersedia", "".join(ev.get("content", "") for ev in events)
            )
            self.assertNotIn("sources", [ev["type"] for ev in events])
            embed.assert_not_called()
            generated = sse(
                self.client.post(
                    "/api/v1/chat", headers=self.headers, json={"query": "KRS?"}
                )
            )[0]["conversation_id"]
            from uuid import UUID

            self.assertEqual(str(UUID(generated)), generated)
        self.assertEqual(
            self.client.post(
                "/api/v1/chat",
                headers=self.headers,
                json={"query": "KRS?", "nim": "999"},
            ).status_code,
            403,
        )
        for history in (
            [{"role": "system", "content": "X"}],
            [{"role": "user", "content": "x"}] * 11,
            [{"role": "user", "content": "x" * 4001}],
        ):
            self.assertEqual(
                self.client.post(
                    "/api/v1/chat",
                    headers=self.headers,
                    json={"query": "KRS?", "history": history},
                ).status_code,
                422,
            )

    def test_chat_prompt_own_profile_filtered_sources_and_history(self):
        self.verify()
        captured = []

        async def generate(prompt):
            captured.append(prompt)
            yield "Jawaban"

        chunks = [
            {
                "content": "Bukti terpercaya",
                "similarity": 0.9,
                "metadata": {"document_name": "Valid", "page": 2},
            },
            {
                "content": "BUKTI LEMAH",
                "similarity": 0.01,
                "metadata": {"document_name": "Weak"},
            },
        ]
        with (
            patch.object(chat.response_generator, "api_key", "fake"),
            patch.object(chat.response_generator, "_client", object()),
            patch.object(chat.response_generator, "generate_stream", generate),
            patch.object(
                chat.response_generator,
                "generate_suggestions",
                AsyncMock(return_value=[]),
            ),
            patch.object(chat.embedding_service, "embed_query", return_value=[0.1]),
            patch.object(chat.vector_store, "query_similar", return_value=chunks),
        ):
            events = sse(
                self.client.post(
                    "/api/v1/chat",
                    headers=self.headers,
                    json={
                        "query": "Lalu apa?",
                        "history": [
                            {"role": "user", "content": "Syarat KRS?"},
                            {"role": "assistant", "content": "Lihat pedoman"},
                        ],
                    },
                )
            )
        self.assertIn("Syarat KRS?", captured[0])
        self.assertIn("Verified Student", captured[0])
        self.assertNotIn("BUKTI LEMAH", captured[0])
        sources = next(ev["data"] for ev in events if ev["type"] == "sources")
        self.assertEqual([source["document"] for source in sources], ["Valid"])

    def test_chat_sse_error_is_sanitized_and_done(self):
        with (
            patch.object(chat.response_generator, "api_key", "fake"),
            patch.object(chat.response_generator, "_client", object()),
            patch.object(
                chat.embedding_service,
                "embed_query",
                side_effect=RuntimeError("PRIVATE_PROVIDER_SECRET"),
            ),
        ):
            events = sse(
                self.client.post(
                    "/api/v1/chat", headers=self.headers, json={"query": "KRS?"}
                )
            )
        self.assertEqual([ev["type"] for ev in events], ["meta", "error", "done"])
        self.assertNotIn("PRIVATE_PROVIDER_SECRET", str(events))


class FirebaseServiceTests(unittest.TestCase):
    def test_sdk_checks_revocation_and_never_logs_credentials(self):
        import sys
        import types

        sdk = types.ModuleType("firebase_admin")
        invalid = type("InvalidIdTokenError", (Exception,), {})
        expired = type("ExpiredIdTokenError", (Exception,), {})
        revoked = type("RevokedIdTokenError", (Exception,), {})
        disabled = type("UserDisabledError", (Exception,), {})
        sdk.auth = SimpleNamespace(
            verify_id_token=MagicMock(return_value=claims()),
            InvalidIdTokenError=invalid,
            ExpiredIdTokenError=expired,
            RevokedIdTokenError=revoked,
            UserDisabledError=disabled,
        )
        sdk.credentials = SimpleNamespace(
            Certificate=MagicMock(side_effect=ValueError("PRIVATE_CREDENTIAL_VALUE"))
        )
        verifier = FirebaseVerifier()
        with (
            patch.dict(sys.modules, {"firebase_admin": sdk}),
            patch.dict("os.environ", {"FIREBASE_AUTH_EMULATOR_HOST": ""}),
        ):
            with patch.object(verifier, "_get_app", return_value="fake-app"):
                self.assertEqual(verifier.verify("signed-token"), claims())
                sdk.auth.verify_id_token.assert_called_once_with(
                    "signed-token", app="fake-app", check_revoked=True
                )
                for exception, status in (
                    (invalid("SECRET"), 401),
                    (expired("SECRET"), 401),
                    (revoked("SECRET"), 401),
                    (disabled("SECRET"), 401),
                    (RuntimeError("SECRET"), 503),
                ):
                    sdk.auth.verify_id_token.side_effect = exception
                    with self.assertRaises(HTTPException) as error:
                        verifier.verify("bad-token")
                    self.assertEqual(error.exception.status_code, status)
                    self.assertNotIn("SECRET", error.exception.detail)
            with (
                patch.object(
                    settings, "FIREBASE_CREDENTIALS_PATH", "credentials/fake.json"
                ),
                patch.object(settings, "GOOGLE_APPLICATION_CREDENTIALS", ""),
                patch.object(settings, "FIREBASE_PROJECT_ID", "fake-project"),
            ):
                with self.assertRaises(HTTPException) as error:
                    verifier.verify("token")
                self.assertEqual(error.exception.status_code, 503)
                self.assertNotIn("PRIVATE_CREDENTIAL_VALUE", error.exception.detail)

    def test_emulator_is_never_an_auth_bypass(self):
        with patch.dict(
            "os.environ", {"FIREBASE_AUTH_EMULATOR_HOST": "localhost:9099"}
        ):
            with self.assertRaises(HTTPException) as error:
                FirebaseVerifier().verify("unsigned-emulator-token")
            self.assertEqual(error.exception.status_code, 503)


class ProcessingTests(unittest.TestCase):
    def test_chunk_max_size_sections_and_progress(self):
        chunker = SmartChunker(chunk_size=100, chunk_overlap=20)
        chunks = chunker.chunk_pages(
            [
                {
                    "page_number": 1,
                    "text": "BAB I\n" + "a" * 1000 + "\nPasal 2\n" + "b " * 150,
                },
                {"page_number": 2, "text": "lanjutan " * 20},
            ]
        )
        self.assertTrue(chunks)
        self.assertTrue(all(0 < len(c["text"]) <= 100 for c in chunks))
        self.assertEqual(chunks[0]["section"], "BAB I")
        self.assertEqual(chunks[-1]["section"], "Pasal 2")
        self.assertEqual(
            [c["chunk_index"] for c in chunks], list(range(1, len(chunks) + 1))
        )
        with self.assertRaises(ValueError):
            SmartChunker(100, 100)

    def test_academic_keywords_have_word_boundaries(self):
        for text in ("tips memasak", "inspirasi olahraga", "transportasi", "chips"):
            self.assertFalse(guardrail.is_campus_related(text), text)
        for text in ("Bagaimana KRS?", "Batas SKS", "syarat MK", "semester pendek SP"):
            self.assertTrue(guardrail.is_campus_related(text), text)

    def test_cache_key_depends_on_content_not_filename(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = root / "cache"
            parser = PDFParser(str(cache))
            pdf_path = root / "same.pdf"

            def write_pdf(text):
                with pymupdf.open() as pdf:
                    page = pdf.new_page()
                    page.insert_text((72, 72), text)
                    pdf.save(str(pdf_path))

            write_pdf("First content long enough to avoid OCR. " * 3)
            first = parser.parse_pdf(str(pdf_path))
            self.assertEqual(len(list(cache.glob("*.json"))), 1)
            with patch(
                "app.ingestion.parser.fitz.open",
                side_effect=AssertionError("cache miss"),
            ):
                self.assertEqual(parser.parse_pdf(str(pdf_path)), first)
            pdf_path.unlink()
            write_pdf("Second entirely different content for this PDF. " * 3)
            second = parser.parse_pdf(str(pdf_path))
            self.assertNotEqual(first, second)
            self.assertEqual(len(list(cache.glob("*.json"))), 2)

    def test_embedding_retry_exhaustion_and_partial_response(self):
        service = EmbeddingService.__new__(EmbeddingService)
        service.api_key, service.model = "fake", "fake"
        service._client = MagicMock()
        for error in (RuntimeError("429 RESOURCE_EXHAUSTED"), RuntimeError("other")):
            service._client.models.embed_content.reset_mock()
            service._client.models.embed_content.side_effect = error
            with (
                patch("app.rag.embeddings.time.sleep"),
                self.assertRaisesRegex(RuntimeError, "retries exhausted"),
            ):
                service.embed_texts(["text"])
            self.assertEqual(service._client.models.embed_content.call_count, 3)
        service._client.models.embed_content.side_effect = None
        service._client.models.embed_content.return_value = SimpleNamespace(
            embeddings=[]
        )
        with patch("app.rag.embeddings.time.sleep"), self.assertRaises(RuntimeError):
            service.embed_texts(["text"])

    def test_reingest_reuses_record_and_replaces_chunks(self):
        engine = create_engine("sqlite://")
        Base.metadata.create_all(engine)
        with (
            tempfile.TemporaryDirectory() as directory,
            sessionmaker(bind=engine)() as db,
        ):
            pdf = Path(directory) / "doc.pdf"
            pdf.write_bytes(b"fake PDF mocked parser")
            with (
                patch(
                    "app.ingestion.pipeline.pdf_parser.parse_pdf",
                    return_value=[
                        {"page_number": 1, "text": "Pasal 1\nKRS", "is_ocr": False}
                    ],
                ),
                patch(
                    "app.ingestion.pipeline.embedding_service.embed_texts",
                    return_value=[[0.1]],
                ),
                patch("app.ingestion.pipeline.vector_store") as store,
            ):
                first = IngestionPipeline().ingest_pdf(str(pdf), db)
                second = IngestionPipeline().ingest_pdf(str(pdf), db)
                self.assertEqual(first["document_id"], second["document_id"])
                self.assertEqual(db.query(Dokumen).count(), 1)
                self.assertEqual(store.delete_by_document_id.call_count, 2)
                self.assertEqual(store.add_chunks.call_count, 2)
        engine.dispose()

    def test_upload_generated_name_content_and_limit(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(settings, "UPLOAD_DIR", directory),
        ):
            with pymupdf.open() as pdf:
                pdf.new_page()
                valid = pdf.tobytes()

            def upload(data):
                return UploadFile(
                    io.BytesIO(data),
                    filename="../../outside.PDF",
                    headers=Headers({"content-type": "application/pdf"}),
                )

            path = save_pdf_upload(upload(valid))
            self.assertEqual(path.parent, Path(directory).resolve())
            self.assertNotIn("outside", path.name)
            self.assertEqual(save_pdf_upload(upload(valid)), path)
            with self.assertRaises(HTTPException) as error:
                save_pdf_upload(upload(b"not a pdf"))
            self.assertEqual(error.exception.status_code, 400)
            with (
                patch.object(settings, "MAX_UPLOAD_BYTES", 10),
                self.assertRaises(HTTPException) as error,
            ):
                save_pdf_upload(upload(valid))
            self.assertEqual(error.exception.status_code, 413)
            self.assertEqual(len(list(Path(directory).glob("*.pdf"))), 1)

    def test_generation_uses_async_sdk_and_propagates_failure(self):
        generator = ResponseGenerator.__new__(ResponseGenerator)
        generator.api_key = "fake"
        generator.model_name = "primary"
        generator.fallback_model = "secondary"
        generator._client = MagicMock()

        async def stream():
            yield SimpleNamespace(text="async response")

        generator._client.aio.models.generate_content_stream = AsyncMock(
            return_value=stream()
        )

        async def collect():
            return [item async for item in generator.generate_stream("test")]

        self.assertEqual(asyncio.run(collect()), ["async response"])
        generator._client.models.generate_content_stream.assert_not_called()
        generator._client.aio.models.generate_content_stream = AsyncMock(
            side_effect=RuntimeError("SECRET")
        )
        with self.assertRaisesRegex(RuntimeError, "AI generation unavailable"):
            asyncio.run(collect())


if __name__ == "__main__":
    unittest.main()
