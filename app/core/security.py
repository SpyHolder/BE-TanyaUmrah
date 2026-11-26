"""Fail-closed Firebase student identity and separately configured admin access."""

import os
import re
import secrets
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Optional
from urllib.parse import urlsplit

from fastapi import Depends, HTTPException, Request
from fastapi.security import (
    HTTPAuthorizationCredentials,
    HTTPBasic,
    HTTPBasicCredentials,
    HTTPBearer,
)

from app.core.config import BACKEND_DIR, settings

bearer = HTTPBearer(auto_error=False)
ADMIN_CHALLENGE = 'Basic realm="tanyaUMRAH Admin"'
basic = HTTPBasic(auto_error=False, realm="tanyaUMRAH Admin")


@dataclass(frozen=True)
class StudentIdentity:
    uid: str
    nim: str
    email: str
    name: str
    picture: Optional[str] = None


class FirebaseVerifier:
    def __init__(self):
        self._app = None
        self._lock = Lock()

    def _get_app(self):
        if self._app is not None:
            return self._app
        with self._lock:
            if self._app is not None:
                return self._app
            path = (
                settings.GOOGLE_APPLICATION_CREDENTIALS
                or settings.FIREBASE_CREDENTIALS_PATH
            )
            if not path or not settings.FIREBASE_PROJECT_ID:
                raise HTTPException(
                    503,
                    "Firebase belum dikonfigurasi: set GOOGLE_APPLICATION_CREDENTIALS atau FIREBASE_CREDENTIALS_PATH dan FIREBASE_PROJECT_ID.",
                )
            try:
                import firebase_admin
                from firebase_admin import credentials

                credential_path = Path(path).expanduser()
                if not credential_path.is_absolute():
                    credential_path = BACKEND_DIR / credential_path
                self._app = firebase_admin.initialize_app(
                    credentials.Certificate(str(credential_path)),
                    {"projectId": settings.FIREBASE_PROJECT_ID},
                    name="tanyaumrah-student-auth",
                )
            except Exception:
                # Never expose credential contents, filesystem paths, or SDK errors.
                raise HTTPException(
                    503,
                    "Firebase tidak tersedia. Periksa service account, firebase-admin, dan FIREBASE_PROJECT_ID.",
                ) from None
        return self._app

    def verify(self, token: str) -> dict:
        # Admin SDK skips signature verification when the emulator variable is set.
        if os.getenv("FIREBASE_AUTH_EMULATOR_HOST"):
            raise HTTPException(
                503,
                "Nonaktifkan FIREBASE_AUTH_EMULATOR_HOST; autentikasi memerlukan Firebase produksi.",
            )
        app = self._get_app()
        from firebase_admin import auth

        try:
            return auth.verify_id_token(token, app=app, check_revoked=True)
        except (
            auth.InvalidIdTokenError,
            auth.ExpiredIdTokenError,
            auth.RevokedIdTokenError,
            auth.UserDisabledError,
            ValueError,
        ):
            raise HTTPException(
                401,
                "Firebase ID token tidak valid atau kedaluwarsa.",
                headers={"WWW-Authenticate": "Bearer"},
            ) from None
        except Exception:
            raise HTTPException(
                503,
                "Verifikasi Firebase tidak tersedia. Periksa koneksi dan konfigurasi Firebase.",
            ) from None


firebase_verifier = FirebaseVerifier()


def verify_student_token(token: str) -> StudentIdentity:
    claims = firebase_verifier.verify(token)
    email = claims.get("email", "")
    match = (
        re.fullmatch(r"([0-9]{1,20})@student\.umrah\.ac\.id", email)
        if isinstance(email, str)
        else None
    )
    provider_claims = claims.get("firebase")
    provider = (
        provider_claims.get("sign_in_provider")
        if isinstance(provider_claims, dict)
        else None
    )
    if (
        not match
        or claims.get("email_verified") is not True
        or provider != "google.com"
        or not claims.get("uid", claims.get("sub"))
    ):
        raise HTTPException(
            403,
            "Gunakan akun Google terverifikasi dengan NIM numerik @student.umrah.ac.id.",
        )
    return StudentIdentity(
        uid=claims.get("uid") or claims["sub"],
        nim=match.group(1),
        email=email,
        name=claims.get("name") or f"Mahasiswa {match.group(1)}",
        picture=claims.get("picture"),
    )


def require_student(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer),
) -> StudentIdentity:
    if not credentials:
        raise HTTPException(
            401,
            "Authorization: Bearer <Firebase ID token> diperlukan.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return verify_student_token(credentials.credentials)


def require_owner(nim: str, identity: StudentIdentity) -> None:
    if nim != identity.nim:
        raise HTTPException(403, "Akses hanya diizinkan untuk profil sendiri.")


def enforce_admin_same_origin(request: Request) -> None:
    if request.method in {"GET", "HEAD", "OPTIONS"}:
        return
    # Reject sibling subdomains as well: same-site is not necessarily same-origin.
    fetch_site = request.headers.get("sec-fetch-site")
    if fetch_site is not None and fetch_site.lower() not in {"same-origin", "none"}:
        raise HTTPException(403, "Mutasi admin hanya diizinkan dari origin yang sama.")
    origin = request.headers.get("origin")
    if origin is None:
        # Non-browser clients (curl) and legacy same-origin forms may omit these headers.
        return
    try:
        parsed = urlsplit(origin)
        target = urlsplit(str(request.url))
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.path
            or parsed.query
            or parsed.fragment
            or any(character.isspace() or ord(character) < 32 for character in origin)
        ):
            raise ValueError("Invalid Origin")

        def origin_key(url):
            port = (
                url.port
                if url.port is not None
                else (443 if url.scheme == "https" else 80)
            )
            return url.scheme, url.hostname, port

        if origin_key(parsed) != origin_key(target):
            raise ValueError("Origin mismatch")
    except ValueError:
        raise HTTPException(
            403, "Mutasi admin hanya diizinkan dari origin yang sama."
        ) from None


def require_admin(
    request: Request, credentials: Optional[HTTPBasicCredentials] = Depends(basic)
) -> str:
    if not settings.ADMIN_USERNAME or not settings.ADMIN_PASSWORD:
        raise HTTPException(
            503, "Admin belum dikonfigurasi: set ADMIN_USERNAME dan ADMIN_PASSWORD."
        )
    valid_user = secrets.compare_digest(
        (credentials.username if credentials else "").encode(),
        settings.ADMIN_USERNAME.encode(),
    )
    valid_password = secrets.compare_digest(
        (credentials.password if credentials else "").encode(),
        settings.ADMIN_PASSWORD.encode(),
    )
    if not (credentials and valid_user and valid_password):
        raise HTTPException(
            401,
            "Kredensial admin diperlukan.",
            headers={"WWW-Authenticate": ADMIN_CHALLENGE},
        )
    enforce_admin_same_origin(request)
    return credentials.username


def require_document_reader(
    request: Request,
    student_credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer),
    admin_credentials: Optional[HTTPBasicCredentials] = Depends(basic),
) -> StudentIdentity | str:
    # Dispatch on the provided scheme; never downgrade an invalid Bearer to admin access.
    if student_credentials is not None:
        return require_student(student_credentials)
    if admin_credentials is not None:
        return require_admin(request, admin_credentials)
    # The Basic challenge lets a browser follow /admin links outside its cached auth path.
    raise HTTPException(
        401,
        "Firebase Bearer atau Basic admin diperlukan untuk mengunduh PDF.",
        headers={"WWW-Authenticate": f"{ADMIN_CHALLENGE}, Bearer"},
    )
