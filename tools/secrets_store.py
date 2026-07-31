"""
Cross-platform secrets store for CairnIQ.

Strategy
--------
Sensitive credentials (API keys, brokerage tokens) are stored in the OS-native
keychain instead of plaintext in `user_data/.env`:

- macOS:   Keychain Access            (encrypted, tied to the login keychain)
- Windows: Credential Manager / DPAPI (encrypted, tied to the user account)
- Linux:   Secret Service / kwallet   (encrypted, when a session bus is available)

Non-secret configuration (BASE_CURRENCY, LLM_PROVIDER, AWS_REGION, model IDs,
toggles) continues to live in `user_data/.env`.

At server startup we run two steps in order:

  1. `migrate_env_to_keyring(env_path)`
     One-time: if `.env` still contains a plaintext value for any known secret,
     we move it to the keychain and blank the value in `.env`. Idempotent.

  2. `load_secrets_into_env()`
     Read every known secret from the keychain and populate `os.environ` so
     that the rest of the codebase keeps working with plain
     `os.environ.get("OPENAI_API_KEY")` reads — no caller changes needed.

Fallback behavior
-----------------
If the `keyring` package is missing, or the platform has no usable backend
(common on headless Linux / inside Docker / CI), we transparently degrade to
plain environment variables. Power users can still do
`OPENAI_API_KEY=... python server.py` and everything works.
"""

from __future__ import annotations

import logging
import os
import sys

logger = logging.getLogger(__name__)

# Service name used as the "service" / "target" in every OS keychain.
# All CairnIQ secrets live under this single namespace.
KEYRING_SERVICE = "cairniq"

# The full set of credentials we treat as secrets. Everything NOT in this set
# stays in `user_data/.env` as readable configuration.
SECRET_KEYS: tuple[str, ...] = (
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_API_KEY_FAST",
    "GOOGLE_API_KEY",
    # Vertex AI service-account key (the full JSON blob). Treated as a normal
    # keychain secret — pasted in Settings like any other key — and turned into
    # credentials at call time so no key file is ever written to disk.
    "GOOGLE_SERVICE_ACCOUNT_KEY",
    "ALPHA_VANTAGE_API_KEY",
    "FMP_API_KEY",
    "FRED_API_KEY",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "TAVILY_API_KEY",
    "FINNHUB_API_KEY",
    "POLYGON_API_KEY",
    "QUESTRADE_REFRESH_TOKEN",
    "ALPACA_API_KEY",
    "ALPACA_SECRET_KEY",
)


import re as _re

# Match `<KNOWN_SECRET>_<N>` failover variants like ALPHA_VANTAGE_API_KEY_2.
# credential_manager.py supports rate-limit rotation across these, so when
# a user has them configured they ARE real credentials and belong in the
# keychain alongside the primary.
_FAILOVER_SUFFIX_RE = _re.compile(r"_(\d+)$")


def is_secret_key(name: str) -> bool:
    """Return True if `name` is a credential that should live in the keychain.

    Covers both primary names in `SECRET_KEYS` and numeric-suffix failover
    variants of the same names (e.g. `FMP_API_KEY_2`, `FRED_API_KEY_3`).
    """
    if name in SECRET_KEYS:
        return True
    m = _FAILOVER_SUFFIX_RE.search(name)
    if m:
        base = name[: m.start()]
        return base in SECRET_KEYS
    return False


# ---------------------------------------------------------------------------
# keyring availability detection
# ---------------------------------------------------------------------------

_keyring = None
_keyring_available: bool | None = None
_keyring_unavailable_reason: str = ""


def _in_test_mode() -> bool:
    """Avoid touching the real OS keychain during pytest runs."""
    if "PYTEST_CURRENT_TEST" in os.environ:
        return True
    if os.environ.get("ACTIVE_PROFILE", "").startswith("pytest_"):
        return True
    if os.environ.get("CAIRNIQ_DISABLE_KEYRING") == "1":
        return True
    return False


def _probe_keyring() -> bool:
    """Lazily import keyring and verify a usable backend exists."""
    global _keyring, _keyring_available, _keyring_unavailable_reason

    # Always re-check test mode FIRST — never cached. This guarantees that no
    # amount of leftover module state from earlier (non-test) calls can let a
    # test under pytest accidentally talk to the real OS keychain.
    if _in_test_mode():
        _keyring_unavailable_reason = "test mode"
        return False

    if _keyring_available is not None:
        return _keyring_available

    try:
        import keyring as _kr  # type: ignore[import-not-found]
        from keyring.errors import NoKeyringError  # type: ignore[import-not-found]
    except ImportError as e:
        _keyring_available = False
        _keyring_unavailable_reason = f"keyring not installed: {e}"
        return False

    # Probe with a no-op read; some Linux setups raise NoKeyringError here.
    try:
        _kr.get_password(KEYRING_SERVICE, "__cairniq_probe__")
    except NoKeyringError as e:
        _keyring_available = False
        _keyring_unavailable_reason = f"no usable backend: {e}"
        return False
    except Exception as e:  # noqa: BLE001 — some backends raise odd things on probe
        _keyring_available = False
        _keyring_unavailable_reason = f"backend error: {e}"
        return False

    _keyring = _kr
    _keyring_available = True
    return True


def keyring_status() -> dict:
    """Return a small dict describing the current secret-store backend.
    Useful for diagnostics in the settings page or `cairniq doctor` output."""
    available = _probe_keyring()
    platform = sys.platform
    backend_name = ""
    if available and _keyring is not None:
        try:
            backend_name = _keyring.get_keyring().__class__.__name__
        except Exception:  # noqa: BLE001
            backend_name = "unknown"
    return {
        "available": available,
        "platform": platform,
        "backend": backend_name,
        "reason": _keyring_unavailable_reason if not available else "",
    }


# ---------------------------------------------------------------------------
# Public read / write API
# ---------------------------------------------------------------------------


def get_secret(name: str) -> str:
    """Return the secret value for `name`.

    Lookup order:
      1. `os.environ` — already populated by `load_secrets_into_env()` at
         startup, and used directly when an env var is explicitly set
         (Docker / CI / power users).
      2. OS keychain — for the typical desktop case after migration.

    Returns "" when not found anywhere. Never raises.
    """
    env_val = os.environ.get(name, "")
    if env_val:
        return env_val

    if not _probe_keyring() or _keyring is None:
        return ""

    try:
        val = _keyring.get_password(KEYRING_SERVICE, name)
        return val or ""
    except Exception as e:  # noqa: BLE001
        logger.warning("keyring read failed for %s: %s", name, e)
        return ""


def set_secret(name: str, value: str) -> bool:
    """Persist `name` → `value` to the OS keychain and to `os.environ`.

    Returns True on success, False if the keyring backend is unavailable
    (caller may decide to fall back to writing `.env`). Always mirrors to
    `os.environ` so the running process sees the change immediately.
    """
    os.environ[name] = value or ""

    if not value:
        # Empty value means "clear it" — also remove from keyring if present.
        return delete_secret(name)

    if not _probe_keyring() or _keyring is None:
        return False

    try:
        _keyring.set_password(KEYRING_SERVICE, name, value)
        return True
    except Exception as e:  # noqa: BLE001
        logger.warning("keyring write failed for %s: %s", name, e)
        return False


def delete_secret(name: str) -> bool:
    """Remove `name` from the keychain. Returns True if the keychain was
    reachable (regardless of whether the entry existed)."""
    os.environ.pop(name, None)

    if not _probe_keyring() or _keyring is None:
        return False

    try:
        _keyring.delete_password(KEYRING_SERVICE, name)
    except Exception:  # noqa: BLE001
        # Most backends raise when the entry doesn't exist; that's fine.
        pass
    return True


def clear_incompatible_aws_session_token() -> dict:
    """Drop stale temporary AWS state when static IAM-user keys are active.

    Long-term IAM-user access keys have the AKIA prefix and must not be signed
    with AWS_SESSION_TOKEN. A stale SSO/session token can leak in from the
    shell, .env, or keychain and make Bedrock reject otherwise valid keys with
    UnrecognizedClientException.
    """
    access_key = os.environ.get("AWS_ACCESS_KEY_ID", "") or ""
    session_token = os.environ.get("AWS_SESSION_TOKEN", "") or ""
    if not session_token or not access_key.startswith("AKIA"):
        return {"cleared": False, "token_len": 0, "access_key_fingerprint": ""}

    os.environ.pop("AWS_SESSION_TOKEN", None)
    os.environ.pop("AWS_PROFILE", None)
    return {
        "cleared": True,
        "token_len": len(session_token),
        "access_key_fingerprint": f"{access_key[:4]}...{access_key[-4:]}",
    }


def load_secrets_into_env() -> int:
    """Populate `os.environ` with every known secret from the OS keychain.

    Looks up:
      - every primary name in `SECRET_KEYS`
      - every numeric-suffix failover variant that exists as a (blank) key in
        the current `os.environ` — these are markers left by the migration
        step or written there by load_dotenv reading a blanked `.env` entry.

    Existing non-empty `os.environ` values always win — that lets Docker / CI
    / a power user override a stored secret by exporting the env var before
    launch.

    Returns the number of secrets pulled from the keychain.
    """
    if not _probe_keyring() or _keyring is None:
        return 0

    # Set of names to attempt: all primaries + any failover variants we can
    # see in os.environ. Using a set avoids double-lookup of primaries.
    names: set[str] = set(SECRET_KEYS)
    for env_name in list(os.environ.keys()):
        if is_secret_key(env_name):
            names.add(env_name)

    loaded = 0
    failed = 0
    for name in names:
        if os.environ.get(name):
            continue
        try:
            val = _keyring.get_password(KEYRING_SERVICE, name)
        except Exception:  # noqa: BLE001
            # Intentionally avoid logging the key name or exception details — both
            # could expose secret-key identifiers/values in plaintext log sinks.
            failed += 1
            continue
        if val:
            os.environ[name] = val
            loaded += 1
    if failed:
        logger.warning("keyring read failed for %d secret(s) during bulk load", failed)
    return loaded


# ---------------------------------------------------------------------------
# One-time migration from plaintext .env → keychain
# ---------------------------------------------------------------------------


def migrate_env_to_keyring(env_path: str) -> dict:
    """Move plaintext secrets out of `.env` and into the OS keychain.

    For every key in `SECRET_KEYS` that currently has a non-empty value in
    `.env`, we:
      1. Write it to the keychain.
      2. Blank the value in `.env` (the key stays so users see what was
         configured, but the secret bytes are gone).

    Safe to call on every startup — it only acts on keys that still have
    plaintext values. If keyring is unavailable, this is a no-op so the user
    can still run with plaintext on platforms without a keychain.

    Returns a small report: {"migrated": [...], "skipped_no_backend": bool}.
    """
    if not os.path.exists(env_path):
        return {"migrated": [], "skipped_no_backend": False}

    if not _probe_keyring() or _keyring is None:
        return {"migrated": [], "skipped_no_backend": True}

    # Read the file as raw lines so we can preserve comments / order / spacing.
    try:
        with open(env_path, encoding="utf-8") as f:
            lines = f.readlines()
    except OSError as e:
        logger.warning("could not read %s for migration: %s", env_path, e)
        return {"migrated": [], "skipped_no_backend": False}

    migrated: list[str] = []
    changed = False
    new_lines: list[str] = []

    for raw in lines:
        stripped = raw.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            new_lines.append(raw)
            continue

        key, _, value = stripped.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")

        if is_secret_key(key) and value:
            try:
                _keyring.set_password(KEYRING_SERVICE, key, value)
                # Replace the line with a blanked version. Keep the key so
                # the user can still see "this is configured" in their .env
                # without exposing the value.
                new_lines.append(f"{key}=\n")
                migrated.append(key)
                changed = True
                continue
            except Exception as e:  # noqa: BLE001
                logger.warning("migration failed for %s: %s", key, e)
                new_lines.append(raw)
                continue

        new_lines.append(raw)

    if changed:
        try:
            with open(env_path, "w", encoding="utf-8") as f:
                f.writelines(new_lines)
        except OSError as e:
            logger.warning("could not rewrite %s after migration: %s", env_path, e)

    if migrated:
        logger.info(
            "Migrated %d secret(s) from plaintext .env into the OS keychain: %s",
            len(migrated),
            ", ".join(migrated),
        )

    return {"migrated": migrated, "skipped_no_backend": False}
