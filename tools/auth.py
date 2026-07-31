"""
Authentication core — user store, password hashing, and signed access tokens.

Local-first / household scale. CairnIQ was built single-user: a request's
active profile is resolved from an *unauthenticated* ``profile`` cookie (see
``tools.user_profile`` and the ``profile_middleware`` in ``server.py``). This
module adds the missing piece — a real login that binds the active profile to
an authenticated identity, so several household members can each chat against
their own portfolio / memory / history.

Model (v1):
  - Users live in a GLOBAL json store (NOT per-profile) keyed by username,
    mapping username -> password hash + profile + role + token_version.
  - One user maps to one profile (defaults to the username).
  - Login issues an HS256 JWT carrying the ``profile`` claim; the profile
    middleware binds the active profile from that claim when a valid token is
    present (Authorization: Bearer ... for the iOS app, or the httponly
    ``cairniq_token`` cookie for the web UI).

Design choices:
  - Password hashing uses stdlib PBKDF2-HMAC-SHA256 — no native build deps, so
    the existing mac/windows/linux installers keep working untouched.
  - Tokens are JWT HS256 via PyJWT (pure-Python).
  - The signing secret lives in the OS keychain (via ``tools.secrets_store``)
    when available, with a durable file fallback so tokens survive restarts on
    headless boxes where no keyring backend exists.
  - Auth is OFF by default (``CAIRNIQ_AUTH_REQUIRED``). Existing single-user
    setups keep working until the login UI / iOS app are ready; flipping the
    flag on then enforces tokens on protected routes.

Token revocation: each user record carries a ``token_version``; issued tokens
embed it as the ``tv`` claim. Bumping a user's version (e.g. on password
change) invalidates all their outstanding tokens without a denylist.
"""

import hashlib
import hmac
import json
import os
import re
import secrets
import time
from datetime import UTC, datetime

import jwt

from tools.secrets_store import get_secret, set_secret

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
AUTH_DB_ENV = "CAIRNIQ_AUTH_DB"          # override store path (tests / custom)
AUTH_REQUIRED_ENV = "CAIRNIQ_AUTH_REQUIRED"
TOKEN_TTL_ENV = "CAIRNIQ_TOKEN_TTL"      # seconds; default 30 days
JWT_SECRET_NAME = "CAIRNIQ_JWT_SECRET"
JWT_ALG = "HS256"

DEFAULT_TOKEN_TTL = 30 * 24 * 60 * 60    # 30 days — household convenience
PBKDF2_ITERATIONS = 600_000

_TRUTHY = {"1", "true", "yes", "y", "on"}
_USERNAME_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,63}$")
_PROFILE_NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,80}$")

_jwt_secret_cache: str | None = None


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
def _project_root() -> str:
    return os.path.dirname(os.path.dirname(__file__))


def _user_data_root() -> str:
    return os.path.join(_project_root(), "user_data")


def _auth_db_path() -> str:
    override = os.environ.get(AUTH_DB_ENV)
    if override:
        return override
    return os.path.join(_user_data_root(), "auth.json")


def _jwt_secret_file() -> str:
    # Sits beside the auth store so test overrides keep everything isolated.
    return os.path.join(os.path.dirname(_auth_db_path()) or ".", ".jwt_secret")


# ---------------------------------------------------------------------------
# Flags
# ---------------------------------------------------------------------------
def _env_truthy(name: str) -> bool:
    return str(os.environ.get(name, "")).strip().lower() in _TRUTHY


def auth_required() -> bool:
    """Return True when protected routes must carry a valid token."""
    raw = os.environ.get(AUTH_REQUIRED_ENV, "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


def token_ttl_seconds() -> int:
    raw = os.environ.get(TOKEN_TTL_ENV, "")
    try:
        ttl = int(raw)
        return ttl if ttl > 0 else DEFAULT_TOKEN_TTL
    except (TypeError, ValueError):
        return DEFAULT_TOKEN_TTL


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------
def normalize_username(username: str | None) -> str:
    """Lowercase + validate a username; returns "" for invalid input."""
    name = str(username or "").strip().lower()
    return name if _USERNAME_RE.fullmatch(name) else ""


def _default_profile_for(username: str) -> str:
    """Map a username to a profile name (1:1 by default)."""
    candidate = username
    return candidate if _PROFILE_NAME_RE.fullmatch(candidate) else "default"


def _normalize_profile(profile: str | None, username: str) -> str:
    name = str(profile or "").strip()
    if not name:
        return _default_profile_for(username)
    if name == "default":
        return "default"
    return name if _PROFILE_NAME_RE.fullmatch(name) else _default_profile_for(username)


# ---------------------------------------------------------------------------
# Password hashing (stdlib PBKDF2)
# ---------------------------------------------------------------------------
def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PBKDF2_ITERATIONS)
    return f"pbkdf2_sha256${PBKDF2_ITERATIONS}${salt.hex()}${dk.hex()}"


def verify_password(password: str, encoded: str) -> bool:
    try:
        algo, iters_s, salt_hex, hash_hex = encoded.split("$")
        if algo != "pbkdf2_sha256":
            return False
        dk = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), bytes.fromhex(salt_hex), int(iters_s)
        )
        return hmac.compare_digest(dk, bytes.fromhex(hash_hex))
    except Exception:
        return False


# ---------------------------------------------------------------------------
# JWT signing secret
# ---------------------------------------------------------------------------
def _read_jwt_secret_file() -> str:
    try:
        with open(_jwt_secret_file(), encoding="utf-8") as fh:
            return fh.read().strip()
    except OSError:
        return ""


def _write_jwt_secret_file(value: str) -> None:
    path = _jwt_secret_file()
    try:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(value)
        os.chmod(path, 0o600)
    except OSError:
        pass


def _get_jwt_secret() -> str:
    """Resolve the HS256 signing secret, generating+persisting on first use.

    Order: env/keychain (via secrets_store) -> durable file fallback ->
    generate. The generated value is mirrored to the keychain when available
    and to a 0600 file otherwise so tokens stay valid across restarts.
    """
    val = get_secret(JWT_SECRET_NAME)
    if val:
        return val

    global _jwt_secret_cache
    if _jwt_secret_cache:
        return _jwt_secret_cache

    val = _read_jwt_secret_file()
    if not val:
        val = secrets.token_urlsafe(48)
        stored = set_secret(JWT_SECRET_NAME, val)  # keychain + os.environ
        # No keychain backend (headless box): persist to a file so a restart
        # doesn't silently invalidate every issued token. Skipped under pytest.
        if not stored and "PYTEST_CURRENT_TEST" not in os.environ:
            _write_jwt_secret_file(val)
    _jwt_secret_cache = val
    return val


# ---------------------------------------------------------------------------
# User store (global json)
# ---------------------------------------------------------------------------
def _load_store() -> dict:
    try:
        with open(_auth_db_path(), encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return {"version": 1, "users": {}}
    if not isinstance(data, dict):
        return {"version": 1, "users": {}}
    data.setdefault("version", 1)
    data.setdefault("users", {})
    return data


def _save_store(store: dict) -> None:
    path = _auth_db_path()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(store, fh, indent=2)
    os.replace(tmp, path)
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


def _public_view(record: dict) -> dict:
    return {
        "username": record.get("username"),
        "profile": record.get("profile"),
        "role": record.get("role", "user"),
        "created_at": record.get("created_at"),
    }


def get_user(username: str) -> dict | None:
    """Return the full stored record (incl. hash) or None."""
    name = normalize_username(username)
    if not name:
        return None
    return _load_store()["users"].get(name)


def list_users() -> list[dict]:
    return [_public_view(r) for r in _load_store()["users"].values()]


def create_user(
    username: str,
    password: str,
    profile: str | None = None,
    role: str = "user",
) -> dict:
    """Create a user. Raises ValueError on invalid input or duplicate."""
    name = normalize_username(username)
    if not name:
        raise ValueError(
            "Invalid username: use 1-64 chars of a-z, 0-9, '_', '.', '-' "
            "(must start alphanumeric)."
        )
    if not password or len(password) < 8:
        raise ValueError("Password must be at least 8 characters.")

    store = _load_store()
    if name in store["users"]:
        raise ValueError(f"User '{name}' already exists.")

    record = {
        "username": name,
        "password_hash": hash_password(password),
        "profile": _normalize_profile(profile, name),
        "role": role if role in ("user", "admin") else "user",
        "token_version": 0,
        "created_at": datetime.now(UTC).isoformat(),
    }
    store["users"][name] = record
    _save_store(store)
    return _public_view(record)


def set_password(username: str, password: str) -> bool:
    """Change a password and bump token_version (invalidates old tokens)."""
    if not password or len(password) < 8:
        raise ValueError("Password must be at least 8 characters.")
    name = normalize_username(username)
    store = _load_store()
    record = store["users"].get(name)
    if record is None:
        return False
    record["password_hash"] = hash_password(password)
    record["token_version"] = int(record.get("token_version", 0)) + 1
    _save_store(store)
    return True


def delete_user(username: str) -> bool:
    name = normalize_username(username)
    store = _load_store()
    if name in store["users"]:
        del store["users"][name]
        _save_store(store)
        return True
    return False


def verify_credentials(username: str, password: str) -> dict | None:
    """Return the public user record on a correct password, else None."""
    record = get_user(username)
    if record is None:
        # Run a dummy hash to keep timing roughly uniform for unknown users.
        verify_password(password, "pbkdf2_sha256$1$00$00")
        return None
    if verify_password(password, record.get("password_hash", "")):
        return _public_view(record)
    return None


# ---------------------------------------------------------------------------
# Tokens
# ---------------------------------------------------------------------------
def issue_token(user: dict) -> tuple[str, int]:
    """Issue an HS256 access token for a (public) user record.

    Returns (token, expires_in_seconds).
    """
    now = int(time.time())
    ttl = token_ttl_seconds()
    record = get_user(user["username"]) or {}
    payload = {
        "sub": user["username"],
        "profile": user["profile"],
        "role": user.get("role", "user"),
        "tv": int(record.get("token_version", 0)),
        "iat": now,
        "exp": now + ttl,
        "jti": secrets.token_hex(8),
    }
    token = jwt.encode(payload, _get_jwt_secret(), algorithm=JWT_ALG)
    return token, ttl


def extract_bearer(authorization: str | None, cookie: str | None) -> str | None:
    """Pull a token from an Authorization header (Bearer) or a cookie value."""
    if authorization and authorization.lower().startswith("bearer "):
        candidate = authorization[7:].strip()
        if candidate:
            return candidate
    if cookie:
        candidate = cookie.strip()
        if candidate:
            return candidate
    return None


def verify_token(token: str) -> dict | None:
    """Validate a token and return its claims, or None if invalid/expired.

    Also enforces token_version: a token whose ``tv`` no longer matches the
    stored user is rejected (covers password changes / forced logout).
    """
    if not token:
        return None
    try:
        claims = jwt.decode(token, _get_jwt_secret(), algorithms=[JWT_ALG])
    except Exception:
        return None
    record = get_user(claims.get("sub", ""))
    if record is not None and int(record.get("token_version", 0)) != int(claims.get("tv", 0)):
        return None
    return claims
