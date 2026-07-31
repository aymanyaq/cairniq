"""Tests for tools.secrets_store.

The module's `_in_test_mode()` helper auto-detects pytest and refuses to touch
the real OS keychain — that's verified directly in `test_test_mode_blocks_real_keyring`.
Tests that exercise the keychain code path inject an in-memory fake keyring
object into the module's globals.
"""
from __future__ import annotations

import os

import pytest

from tools import secrets_store

# ---------------------------------------------------------------------------
# Fakes & fixtures
# ---------------------------------------------------------------------------


class _FakeKeyring:
    """In-memory stand-in for the `keyring` module — just enough surface to
    satisfy `tools.secrets_store` (`get_password`, `set_password`,
    `delete_password`, `get_keyring`)."""

    class _Backend:
        pass

    def __init__(self) -> None:
        self._store: dict[tuple[str, str], str] = {}

    def get_password(self, service: str, name: str) -> str | None:
        return self._store.get((service, name))

    def set_password(self, service: str, name: str, value: str) -> None:
        self._store[(service, name)] = value

    def delete_password(self, service: str, name: str) -> None:
        self._store.pop((service, name), None)

    def get_keyring(self) -> _FakeKeyring._Backend:
        return self._Backend()


@pytest.fixture(autouse=True)
def _reset_module_state_and_env(monkeypatch):
    """Reset cached probe state and AWS env vars before each test."""
    monkeypatch.setattr(secrets_store, "_keyring", None)
    monkeypatch.setattr(secrets_store, "_keyring_available", None)
    monkeypatch.setattr(secrets_store, "_keyring_unavailable_reason", "")
    for k in (
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "AWS_PROFILE",
        "OPENAI_API_KEY",
        "FRED_API_KEY",
        "FMP_API_KEY",
        "FMP_API_KEY_2",
        "ANTHROPIC_API_KEY",
    ):
        monkeypatch.delenv(k, raising=False)
    yield


@pytest.fixture
def fake_keychain(monkeypatch):
    """Replace the real keyring backend with an in-memory fake, mark the probe
    as already succeeded, AND temporarily disable the test-mode short-circuit
    in `_probe_keyring`. Tests using this fixture are explicitly opting into
    "pretend a keychain is available" — the safety net (which otherwise blocks
    pytest from ever touching the real OS keychain) is opted out only here."""
    fk = _FakeKeyring()
    monkeypatch.setattr(secrets_store, "_keyring", fk)
    monkeypatch.setattr(secrets_store, "_keyring_available", True)
    monkeypatch.setattr(secrets_store, "_keyring_unavailable_reason", "")
    monkeypatch.setattr(secrets_store, "_in_test_mode", lambda: False)
    return fk


# ---------------------------------------------------------------------------
# is_secret_key — primary + _N failover variants
# ---------------------------------------------------------------------------


def test_primary_keys_are_recognized_as_secrets():
    for name in secrets_store.SECRET_KEYS:
        assert secrets_store.is_secret_key(name), name


def test_numeric_failover_variants_are_recognized():
    # Match credential_manager.py's `_2` rotation pattern — these are real
    # credentials and must end up in the keychain alongside primaries.
    assert secrets_store.is_secret_key("FMP_API_KEY_2")
    assert secrets_store.is_secret_key("FRED_API_KEY_3")
    assert secrets_store.is_secret_key("TAVILY_API_KEY_42")


def test_non_secrets_are_not_recognized():
    for name in ("BASE_CURRENCY", "AWS_REGION", "LLM_PROVIDER", "QUESTRADE_ENABLED"):
        assert not secrets_store.is_secret_key(name), name


def test_underscore_digit_on_non_secret_is_not_a_secret():
    # SOMETHING_ELSE_2 shouldn't promote a random env var to "secret" status.
    assert not secrets_store.is_secret_key("SOMETHING_ELSE_2")


# ---------------------------------------------------------------------------
# Test-mode safety — refuses to talk to the real keychain under pytest
# ---------------------------------------------------------------------------


def test_test_mode_blocks_real_keyring(monkeypatch):
    # pytest sets PYTEST_CURRENT_TEST automatically, but make it explicit.
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "active")
    monkeypatch.setattr(secrets_store, "_keyring", None)
    monkeypatch.setattr(secrets_store, "_keyring_available", None)
    assert secrets_store._probe_keyring() is False
    status = secrets_store.keyring_status()
    assert status["available"] is False
    assert "test mode" in status["reason"]


def test_explicit_disable_via_env_var(monkeypatch):
    # Documented escape hatch: CAIRNIQ_DISABLE_KEYRING=1
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.setenv("CAIRNIQ_DISABLE_KEYRING", "1")
    assert secrets_store._probe_keyring() is False


def test_cached_availability_cannot_bypass_test_mode(monkeypatch):
    # Regression guard: a stale `_keyring_available = True` left over from
    # a previous (non-test) call must NEVER let pytest leak through to the
    # real OS keychain. The probe must re-check test mode every call.
    monkeypatch.setattr(secrets_store, "_keyring", object())  # would look "loaded"
    monkeypatch.setattr(secrets_store, "_keyring_available", True)  # cached as ready
    # PYTEST_CURRENT_TEST is set automatically; that's the point.
    assert secrets_store._probe_keyring() is False
    # And the downstream functions stay safe too.
    assert secrets_store.load_secrets_into_env() == 0
    assert secrets_store.get_secret("AWS_ACCESS_KEY_ID") == ""


# ---------------------------------------------------------------------------
# get/set/delete with the in-memory fake backend
# ---------------------------------------------------------------------------


def test_set_secret_writes_to_keychain_and_mirrors_to_env(fake_keychain):
    assert secrets_store.set_secret("OPENAI_API_KEY", "sk-test-roundtrip") is True
    assert fake_keychain._store[(secrets_store.KEYRING_SERVICE, "OPENAI_API_KEY")] == "sk-test-roundtrip"
    assert os.environ["OPENAI_API_KEY"] == "sk-test-roundtrip"


def test_get_secret_prefers_env_over_keychain(fake_keychain, monkeypatch):
    # Power-user override: AWS_ACCESS_KEY_ID=... exported in the shell should
    # always win, even if the keychain has a different value.
    fake_keychain.set_password(secrets_store.KEYRING_SERVICE, "AWS_ACCESS_KEY_ID", "from-keychain")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "from-env")
    assert secrets_store.get_secret("AWS_ACCESS_KEY_ID") == "from-env"


def test_get_secret_falls_through_to_keychain_when_env_empty(fake_keychain):
    fake_keychain.set_password(secrets_store.KEYRING_SERVICE, "AWS_ACCESS_KEY_ID", "from-keychain")
    assert secrets_store.get_secret("AWS_ACCESS_KEY_ID") == "from-keychain"


def test_get_secret_returns_empty_when_missing_everywhere(fake_keychain):
    assert secrets_store.get_secret("FRED_API_KEY") == ""


def test_delete_secret_removes_from_env_and_keychain(fake_keychain):
    secrets_store.set_secret("FRED_API_KEY", "value")
    assert os.environ.get("FRED_API_KEY") == "value"
    secrets_store.delete_secret("FRED_API_KEY")
    assert "FRED_API_KEY" not in os.environ
    assert fake_keychain.get_password(secrets_store.KEYRING_SERVICE, "FRED_API_KEY") is None


def test_set_secret_with_empty_value_clears_existing(fake_keychain):
    secrets_store.set_secret("FRED_API_KEY", "value")
    secrets_store.set_secret("FRED_API_KEY", "")
    assert fake_keychain.get_password(secrets_store.KEYRING_SERVICE, "FRED_API_KEY") is None


# ---------------------------------------------------------------------------
# Fallback when keyring is unavailable (headless Linux / Docker / CI)
# ---------------------------------------------------------------------------


def test_set_secret_returns_false_when_backend_unavailable(monkeypatch):
    # Probe stays unavailable — the function must mirror to env but report False
    # so callers (e.g. the settings save handler) know to fall back to .env.
    monkeypatch.setattr(secrets_store, "_keyring", None)
    monkeypatch.setattr(secrets_store, "_keyring_available", False)
    assert secrets_store.set_secret("FMP_API_KEY", "x") is False
    assert os.environ["FMP_API_KEY"] == "x"


def test_load_secrets_into_env_is_noop_when_backend_unavailable(monkeypatch):
    monkeypatch.setattr(secrets_store, "_keyring", None)
    monkeypatch.setattr(secrets_store, "_keyring_available", False)
    assert secrets_store.load_secrets_into_env() == 0


# ---------------------------------------------------------------------------
# load_secrets_into_env — keychain → os.environ hydration
# ---------------------------------------------------------------------------


def test_load_secrets_into_env_populates_known_primaries(fake_keychain):
    fake_keychain.set_password(secrets_store.KEYRING_SERVICE, "AWS_ACCESS_KEY_ID", "AKIAFAKE")
    fake_keychain.set_password(secrets_store.KEYRING_SERVICE, "AWS_SECRET_ACCESS_KEY", "secret40")

    n = secrets_store.load_secrets_into_env()
    assert n == 2
    assert os.environ["AWS_ACCESS_KEY_ID"] == "AKIAFAKE"
    assert os.environ["AWS_SECRET_ACCESS_KEY"] == "secret40"


def test_load_secrets_into_env_picks_up_failover_variants_from_env(fake_keychain, monkeypatch):
    # When the migrated .env contains a blank `FMP_API_KEY_2=`, load_dotenv puts
    # an empty string in os.environ. load_secrets_into_env must notice and pull
    # the real value from the keychain.
    monkeypatch.setenv("FMP_API_KEY_2", "")
    fake_keychain.set_password(secrets_store.KEYRING_SERVICE, "FMP_API_KEY_2", "fmp-failover-key")

    secrets_store.load_secrets_into_env()
    assert os.environ["FMP_API_KEY_2"] == "fmp-failover-key"


def test_existing_env_value_wins_over_keychain(fake_keychain, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "from-shell")
    fake_keychain.set_password(secrets_store.KEYRING_SERVICE, "OPENAI_API_KEY", "from-keychain")

    secrets_store.load_secrets_into_env()
    assert os.environ["OPENAI_API_KEY"] == "from-shell"


# ---------------------------------------------------------------------------
# migrate_env_to_keyring — plaintext .env → keychain, blank in file
# ---------------------------------------------------------------------------


def test_migrate_missing_file_is_noop(tmp_path):
    report = secrets_store.migrate_env_to_keyring(str(tmp_path / "does-not-exist.env"))
    assert report == {"migrated": [], "skipped_no_backend": False}


def test_migrate_skipped_when_backend_unavailable(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("FRED_API_KEY=plaintext-secret\n")
    monkeypatch.setattr(secrets_store, "_keyring", None)
    monkeypatch.setattr(secrets_store, "_keyring_available", False)

    report = secrets_store.migrate_env_to_keyring(str(env_file))
    assert report["migrated"] == []
    assert report["skipped_no_backend"] is True
    # File must be untouched so the user can still run from plaintext.
    assert "FRED_API_KEY=plaintext-secret" in env_file.read_text()


def test_migrate_moves_secrets_to_keychain_and_blanks_the_file(tmp_path, fake_keychain):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "# my config\n"
        "FRED_API_KEY=plaintext-secret-xyz\n"
        "BASE_CURRENCY=USD\n"
        "AWS_ACCESS_KEY_ID=AKIAEXAMPLE123\n"
    )

    report = secrets_store.migrate_env_to_keyring(str(env_file))
    assert sorted(report["migrated"]) == ["AWS_ACCESS_KEY_ID", "FRED_API_KEY"]
    assert report["skipped_no_backend"] is False

    # Keychain has the values
    assert fake_keychain.get_password(secrets_store.KEYRING_SERVICE, "FRED_API_KEY") == "plaintext-secret-xyz"
    assert fake_keychain.get_password(secrets_store.KEYRING_SERVICE, "AWS_ACCESS_KEY_ID") == "AKIAEXAMPLE123"

    # File: secrets blanked, non-secrets and comments preserved
    contents = env_file.read_text()
    assert "plaintext-secret-xyz" not in contents
    assert "AKIAEXAMPLE123" not in contents
    assert "FRED_API_KEY=\n" in contents       # key remains as a "configured" marker
    assert "AWS_ACCESS_KEY_ID=\n" in contents
    assert "BASE_CURRENCY=USD" in contents     # non-secret untouched
    assert "# my config" in contents           # comment preserved


def test_migrate_strips_surrounding_quotes(tmp_path, fake_keychain):
    # .env files often quote values: KEY="value" or KEY='value'. Migration must
    # store the clean value, not the quoted form.
    env_file = tmp_path / ".env"
    env_file.write_text('FMP_API_KEY="quoted-value"\nPOLYGON_API_KEY=\'single-quoted\'\n')

    secrets_store.migrate_env_to_keyring(str(env_file))
    assert fake_keychain.get_password(secrets_store.KEYRING_SERVICE, "FMP_API_KEY") == "quoted-value"
    assert fake_keychain.get_password(secrets_store.KEYRING_SERVICE, "POLYGON_API_KEY") == "single-quoted"


def test_migrate_skips_empty_secret_lines(tmp_path, fake_keychain):
    # Placeholder lines like `OPENAI_API_KEY=` or `ALPACA_API_KEY=""` must not
    # cause the migration to write empty strings to the keychain.
    env_file = tmp_path / ".env"
    env_file.write_text('OPENAI_API_KEY=\nALPACA_API_KEY=""\n')

    report = secrets_store.migrate_env_to_keyring(str(env_file))
    assert report["migrated"] == []
    assert fake_keychain.get_password(secrets_store.KEYRING_SERVICE, "OPENAI_API_KEY") is None


def test_migrate_handles_failover_variants(tmp_path, fake_keychain):
    env_file = tmp_path / ".env"
    env_file.write_text("FMP_API_KEY_2=failover-secret\n")

    report = secrets_store.migrate_env_to_keyring(str(env_file))
    assert report["migrated"] == ["FMP_API_KEY_2"]
    assert fake_keychain.get_password(secrets_store.KEYRING_SERVICE, "FMP_API_KEY_2") == "failover-secret"


def test_migrate_is_idempotent(tmp_path, fake_keychain):
    env_file = tmp_path / ".env"
    env_file.write_text("FRED_API_KEY=secret-once\n")

    first = secrets_store.migrate_env_to_keyring(str(env_file))
    assert first["migrated"] == ["FRED_API_KEY"]

    # Second pass: file already blanked, nothing left to do.
    second = secrets_store.migrate_env_to_keyring(str(env_file))
    assert second["migrated"] == []


# ---------------------------------------------------------------------------
# clear_incompatible_aws_session_token
# ---------------------------------------------------------------------------


def test_session_token_cleared_when_paired_with_iam_user_key(monkeypatch):
    # Long-term IAM keys start with "AKIA" — combining them with a session
    # token (which only goes with temporary STS creds) makes AWS reject the
    # request with UnrecognizedClientException.
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIAEXAMPLE123456789")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "FQoGZXIv...stale...")
    monkeypatch.setenv("AWS_PROFILE", "old-sso-profile")

    report = secrets_store.clear_incompatible_aws_session_token()
    assert report["cleared"] is True
    assert "AWS_SESSION_TOKEN" not in os.environ
    assert "AWS_PROFILE" not in os.environ
    assert "AWS_ACCESS_KEY_ID" in os.environ  # the IAM user key stays


def test_session_token_preserved_for_temporary_keys(monkeypatch):
    # Temporary STS keys start with "ASIA" — those legitimately pair with a
    # session token and must NOT be stripped.
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "ASIAEXAMPLE123456789")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "real-sts-session-token")

    report = secrets_store.clear_incompatible_aws_session_token()
    assert report["cleared"] is False
    assert os.environ["AWS_SESSION_TOKEN"] == "real-sts-session-token"


def test_no_op_when_no_session_token(monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIAEXAMPLE")
    # AWS_SESSION_TOKEN not set
    report = secrets_store.clear_incompatible_aws_session_token()
    assert report["cleared"] is False
