"""Download authorization and browser CSRF checks without network or deployment data."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.core.config import settings
from app.core.security import firebase_verifier
from app.database.models import AcademicEvent, Dokumen, Mahasiswa
from app.database.session import Base, get_db
from app.main import app
from app.routers import admin
from fastapi import HTTPException
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool


class AccessSecurityTests(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine(
            "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
        )
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

        def memory_db():
            with self.Session() as db:
                yield db

        app.dependency_overrides[get_db] = memory_db
        self.addCleanup(app.dependency_overrides.clear)
        self.addCleanup(self.engine.dispose)
        # No lifespan: never initialize deployment SQLite or storage directories.
        self.client = TestClient(app)
        self.addCleanup(self.client.close)
        self.verifier = self.enter_patch(
            patch.object(
                firebase_verifier,
                "verify",
                return_value={
                    "uid": "fake-uid",
                    "email": "2401020163@student.umrah.ac.id",
                    "email_verified": True,
                    "name": "Test Student",
                    "firebase": {"sign_in_provider": "google.com"},
                },
            )
        )
        self.enter_patch(patch.object(settings, "ADMIN_USERNAME", "fake-admin"))
        self.enter_patch(patch.object(settings, "ADMIN_PASSWORD", "fake-password"))
        self.auth = ("fake-admin", "fake-password")
        self.bearer = {"Authorization": "Bearer fake-id-token"}
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.uploads = self.root / "uploads"
        self.uploads.mkdir()
        self.enter_patch(patch.object(settings, "UPLOAD_DIR", str(self.uploads)))
        self.enter_patch(
            patch.object(settings, "DATA_PDF_DIR", str(self.root / "data"))
        )
        self.pdf_bytes = b"%PDF-1.4\nFake PDF download fixture\n%%EOF"
        (self.uploads / "fixture.pdf").write_bytes(self.pdf_bytes)
        with self.Session() as db:
            doc = Dokumen(nama="Fixture", filename="fixture.pdf", status_ingest="done")
            db.add(doc)
            db.commit()
            self.pdf_url = f"/api/v1/documents/{doc.id}/file"

    def enter_patch(self, patcher):
        value = patcher.start()
        self.addCleanup(patcher.stop)
        return value

    def mutation_paths(self):
        return [
            "/admin"
            + route.path.replace("{document_id}", "1").replace("{event_id}", "1")
            for route in admin.router.routes
            if "POST" in route.methods
        ] + ["/api/v1/calendar/events"]

    def test_pdf_accepts_student_or_admin_and_challenges_browser(self):
        response = self.client.get(self.pdf_url)
        self.assertEqual(response.status_code, 401)
        self.assertIn(
            'Basic realm="tanyaUMRAH Admin"', response.headers["www-authenticate"]
        )
        self.assertIn("Bearer", response.headers["www-authenticate"])
        response = self.client.get(self.pdf_url, auth=self.auth)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, self.pdf_bytes)
        self.verifier.assert_not_called()
        for response in (response, self.client.get(self.pdf_url, headers=self.bearer)):
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.content, self.pdf_bytes)
            self.assertEqual(response.headers["content-type"], "application/pdf")
            self.assertEqual(response.headers["cache-control"], "private, no-store")
            self.assertEqual(response.headers["x-content-type-options"], "nosniff")

    def test_pdf_bad_credentials_and_unavailable_verifier_never_fall_back(self):
        self.assertEqual(
            self.client.get(self.pdf_url, auth=("fake-admin", "wrong")).status_code, 401
        )
        self.verifier.assert_not_called()
        for code in (401, 503):
            self.verifier.side_effect = HTTPException(code, "fake verification failure")
            self.assertEqual(
                self.client.get(self.pdf_url, headers=self.bearer).status_code, code
            )
        self.verifier.side_effect = None
        self.verifier.return_value["email_verified"] = False
        self.assertEqual(
            self.client.get(self.pdf_url, headers=self.bearer).status_code, 403
        )

    def test_pdf_unconfigured_admin_is_503_but_student_access_still_works(self):
        for missing in ("ADMIN_USERNAME", "ADMIN_PASSWORD"):
            with self.subTest(missing=missing), patch.object(settings, missing, ""):
                self.assertEqual(
                    self.client.get(self.pdf_url, auth=self.auth).status_code, 503
                )
                self.assertEqual(
                    self.client.get(self.pdf_url, headers=self.bearer).status_code, 200
                )

    def test_basic_permission_is_not_extended_to_other_student_endpoints(self):
        for path in (
            "/api/v1/documents",
            "/api/v1/user/profile/2401020163",
            "/api/v1/calendar/events",
        ):
            self.assertEqual(
                self.client.get(path, auth=self.auth).status_code, 401, path
            )
        self.assertEqual(
            self.client.post(
                "/api/v1/chat", auth=self.auth, json={"query": "KRS?"}
            ).status_code,
            401,
        )
        self.assertEqual(
            self.client.post(
                "/api/v1/chat/feedback",
                auth=self.auth,
                json={"message_id": "m", "rating": "helpful"},
            ).status_code,
            401,
        )
        self.assertEqual(
            self.client.get("/admin/students", headers=self.bearer).status_code, 401
        )
        self.assertEqual(
            self.client.post(
                "/api/v1/calendar/events",
                headers=self.bearer,
                json={"title": "T", "start_date": "2026-11-01"},
            ).status_code,
            401,
        )

    def test_pdf_cannot_escape_storage_even_for_admin(self):
        (self.root / "outside.pdf").write_bytes(b"PRIVATE OUTSIDE STORAGE")
        with self.Session() as db:
            db.query(Dokumen).one().filename = "../outside.pdf"
            db.commit()
        for options in ({"auth": self.auth}, {"headers": self.bearer}):
            response = self.client.get(self.pdf_url, **options)
            self.assertEqual(response.status_code, 404)
            self.assertNotIn("PRIVATE OUTSIDE STORAGE", response.text)

    def test_cross_site_or_sibling_site_rejected_on_every_admin_mutation(self):
        with (
            patch("app.routers.admin.ingestion_pipeline") as ingestion,
            patch("app.routers.admin.vector_store") as vectors,
        ):
            for path in self.mutation_paths():
                for headers in (
                    {"Origin": "https://evil.example"},
                    {"Sec-Fetch-Site": "cross-site"},
                    {"Sec-Fetch-Site": "same-site"},
                    {"Origin": "http://testserver", "Sec-Fetch-Site": "cross-site"},
                ):
                    with self.subTest(path=path, headers=headers):
                        response = self.client.post(
                            path,
                            auth=self.auth,
                            headers=headers,
                            follow_redirects=False,
                        )
                        self.assertEqual(response.status_code, 403, response.text)
            ingestion.ingest_pdf.assert_not_called()
            vectors.delete_by_document_id.assert_not_called()
        with self.Session() as db:
            self.assertEqual(db.query(AcademicEvent).count(), 0)
            self.assertEqual(db.query(Mahasiswa).count(), 0)
            self.assertEqual(db.query(Dokumen).count(), 1)

    def test_origin_validation_rejects_null_other_ports_schemes_and_malformed(self):
        for origin in (
            "null",
            "http://testserver.evil",
            "http://testserver:8080",
            "https://testserver",
            "http://testserver@evil.example",
            "http://evil.example@testserver",
            "http://testserver/path",
            "http://testserver?query=1",
            "http://testserver#fragment",
            "http://testserver:bad",
            "http://testserver https://evil.example",
            "",
        ):
            with self.subTest(origin=origin):
                response = self.client.post(
                    "/api/v1/calendar/events",
                    auth=self.auth,
                    headers={"Origin": origin, "Sec-Fetch-Site": "same-origin"},
                    json={"title": "T", "start_date": "2026-11-01"},
                )
                self.assertEqual(response.status_code, 403)

    def test_same_origin_forms_and_headerless_curl_remain_allowed(self):
        for headers in (
            {},
            {"Origin": "http://testserver", "Sec-Fetch-Site": "same-origin"},
            {"Origin": "http://TESTSERVER:80"},
            {"Sec-Fetch-Site": "same-origin"},
            {"Sec-Fetch-Site": "none"},
        ):
            with self.subTest(headers=headers):
                response = self.client.post(
                    "/admin/calendar/save",
                    auth=self.auth,
                    headers=headers,
                    data={"title": "Fixture", "start_date": "2026-11-01"},
                    follow_redirects=False,
                )
                self.assertEqual(response.status_code, 303, response.text)
        response = self.client.post(
            "/api/v1/calendar/events",
            auth=self.auth,
            headers={"Origin": "http://testserver", "Sec-Fetch-Site": "same-origin"},
            json={"title": "API fixture", "start_date": "2026-11-02"},
        )
        self.assertEqual(response.status_code, 200)
        with self.Session() as db:
            self.assertEqual(db.query(AcademicEvent).count(), 6)

    def test_unconfigured_admin_remains_503_even_for_cross_site_mutations(self):
        with patch.object(settings, "ADMIN_PASSWORD", ""):
            for path in self.mutation_paths():
                response = self.client.post(
                    path,
                    auth=self.auth,
                    headers={
                        "Origin": "https://evil.example",
                        "Sec-Fetch-Site": "cross-site",
                    },
                )
                self.assertEqual(response.status_code, 503, path)
                self.assertIn("ADMIN_PASSWORD", response.json()["detail"])


if __name__ == "__main__":
    unittest.main()
