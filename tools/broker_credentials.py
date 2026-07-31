"""
Per-profile broker credentials.

CairnIQ historically stored broker credentials process-globally — a single
keychain namespace plus os.environ loaded once at boot. Switching the active
profile therefore switched the user's DATA files (portfolio, history, memory)
but NOT which brokerage account was queried. In a household where several
people each link their own Alpaca/Questrade account, that means one member's
chat would read another member's live positions, and saving a key would clobber
everyone else's.

This module scopes broker credentials to the active profile:

  - The ``default`` profile keeps the legacy GLOBAL behaviour untouched
    (secrets in the global keychain; session/config in user_data/.env), so the
    existing single user needs no migration and nothing about their setup
    changes.
  - Any NAMED profile gets isolated storage:
      * secrets  -> per-profile keychain entry ``<profile>::NAME``
      * settings -> user_data/profiles/<profile>/broker_config.json
    A named profile NEVER falls back to the global keys — an unconfigured
    profile simply has no broker access (rather than silently using the
    operator's account).

Value classes:
  - SECRETS  (API keys, refresh tokens): long-term credentials -> keychain.
  - SETTINGS (paper-mode, enabled, account owner, and Questrade session state:
    access token / api server / token expiry): non-secret or short-lived,
    -> .env (default) or the per-profile json (named).
"""

import json
import os

from dotenv import load_dotenv, set_key

from tools.secrets_store import delete_secret, get_secret, set_secret
from tools.user_profile import get_active_profile, get_data_path

DEFAULT_PROFILE = "default"

# Long-term credentials -> keychain. Matched by prefix so numeric failover
# variants (e.g. QUESTRADE_REFRESH_TOKEN_2) are covered too.
_BROKER_SECRET_PREFIXES = (
    "ALPACA_API_KEY",
    "ALPACA_SECRET_KEY",
    "QUESTRADE_REFRESH_TOKEN",
)
# Non-secret config / session state -> .env (default) or per-profile json (named).
_BROKER_SETTING_PREFIXES = (
    "ALPACA_PAPER_MODE",
    "QUESTRADE_ENABLED",
    "QUESTRADE_ACCOUNT_OWNER",
    "QUESTRADE_ACCESS_TOKEN",
    "QUESTRADE_API_SERVER",
    "QUESTRADE_TOKEN_EXPIRY",
    "SCHEDULER_ENABLED",
)


def _project_root() -> str:
    return os.path.dirname(os.path.dirname(__file__))


def _env_path() -> str:
    return os.path.join(_project_root(), "user_data", ".env")


def is_broker_secret(name: str) -> bool:
    return any(name == p or name.startswith(p) for p in _BROKER_SECRET_PREFIXES)


def is_broker_setting(name: str) -> bool:
    return any(name == p or name.startswith(p) for p in _BROKER_SETTING_PREFIXES)


def is_broker_credential(name: str) -> bool:
    return is_broker_secret(name) or is_broker_setting(name)


def _active() -> str:
    return get_active_profile() or DEFAULT_PROFILE


def _ns(profile: str, name: str) -> str:
    return f"{profile}::{name}"


# ---------------------------------------------------------------------------
# Per-profile non-secret config (named profiles only)
# ---------------------------------------------------------------------------
def _config_path() -> str:
    return get_data_path("broker_config.json")


def _load_config() -> dict:
    try:
        with open(_config_path(), encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _save_config(cfg: dict) -> None:
    path = _config_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=2)
    os.replace(tmp, path)
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Public accessors
# ---------------------------------------------------------------------------
def get_broker_secret(name: str) -> str:
    """Return a broker secret for the active profile ("" if unset)."""
    profile = _active()
    if profile == DEFAULT_PROFILE:
        return get_secret(name)
    return get_secret(_ns(profile, name))


def set_broker_secret(name: str, value: str) -> None:
    """Persist a broker secret for the active profile (empty value clears it)."""
    profile = _active()
    key = name if profile == DEFAULT_PROFILE else _ns(profile, name)
    if value:
        set_secret(key, value)
    else:
        delete_secret(key)


def get_broker_setting(name: str, default: str = "") -> str:
    """Return a non-secret broker setting for the active profile."""
    profile = _active()
    if profile == DEFAULT_PROFILE:
        val = os.environ.get(name)
        return val if val is not None else default
    val = _load_config().get(name)
    return val if val is not None else default


def set_broker_setting(name: str, value: str) -> None:
    """Persist a non-secret broker setting for the active profile."""
    profile = _active()
    if profile == DEFAULT_PROFILE:
        os.environ[name] = value
        try:
            set_key(_env_path(), name, value)
        except Exception:
            pass
        return
    cfg = _load_config()
    cfg[name] = value
    _save_config(cfg)


def broker_lock_path() -> str:
    """Per-profile lock path for Questrade token-refresh coordination."""
    if _active() == DEFAULT_PROFILE:
        return _env_path() + ".lock"
    return _config_path() + ".lock"


def refresh_profile_state() -> None:
    """Reload persisted broker state so a sibling worker's refresh is visible.

    For the default profile this re-reads .env into os.environ (the legacy
    coordination mechanism). For named profiles the json is read fresh on every
    ``get_broker_setting`` call, so this is a no-op.
    """
    if _active() == DEFAULT_PROFILE:
        load_dotenv(_env_path(), override=True)
