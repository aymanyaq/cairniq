"""
Profile Resolution
------------------
Resolves the active user profile for the current request context.

Resolution order (per request):
  1. ContextVar set by FastAPI middleware (populated from X-Profile header or cookie)
  2. ACTIVE_PROFILE environment variable (server-wide default) — single-user only
  3. "default" literal fallback

The ContextVar approach is async/thread-safe: each request has its own
profile without any locking. A future login page simply calls
set_active_profile() in middleware and all downstream tools pick it up.

Multi-user safety: in a server that serves more than one profile per process,
the ACTIVE_PROFILE environment variable is process-global and therefore unsafe
as a fallback — a detached worker thread that lost its ContextVar would read
whichever profile last touched the env var, leaking one user's data into
another's analysis. server.py calls enable_multiuser_guard() at startup; once
on, a lost ContextVar resolves to the isolated, data-less UNBOUND_PROFILE
sentinel (and logs it to logs/tools/tools.jsonl) instead of the global env var.
The real fix is to re-bind the profile inside every worker (see
run_under_profile); the guard is the fail-safe for any boundary that slips
through.
"""
import csv
import json
import logging
import os
import re
import shutil
import threading
import traceback
from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import date, timedelta

from tools.exception_logger import log_exceptions
from tools.json_store import write_json_atomic

# -----------------------------------------------------------------------
# Per-request profile context (thread & async safe via contextvars)
# -----------------------------------------------------------------------
_profile_ctx: ContextVar[str | None] = ContextVar("active_profile", default=None)
_PROFILE_NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,80}$")
_TRUTHY = {"1", "true", "yes", "y", "on"}
DEMO_PROFILE_NAME = "demo"

# Isolated, intentionally empty profile returned when the active-profile
# ContextVar is lost inside a multi-user server (see enable_multiuser_guard).
# It maps to user_data/profiles/_unbound/ — a worker that lands here reads no
# holdings rather than another user's, so a binding bug degrades to "no data"
# instead of a cross-user leak.
UNBOUND_PROFILE = "_unbound"

# Set True by enable_multiuser_guard() when the process serves per-request
# profiles. Plain module global (NOT a ContextVar) so it is visible from every
# thread, including detached workers.
_multiuser_guard = False

# --- Unbound-fallback diagnostic -----------------------------------------
# The fallback warning used to go to logging.getLogger(__name__), which wrote
# nowhere: this module attaches a NullHandler (the library convention), that
# NullHandler counts as "a handler was found", and so logging never falls back
# to lastResort/stderr — while nothing in this app configures the root logger.
# Result: four weeks of the guard firing in production with 0 lines in
# cairniq.stderr.log and 0 in logs/. It now goes to the component channel every
# other diagnostic here uses (own FileHandler, propagate=False), which no
# uvicorn/litellm logging setup can reconfigure away.
_UNBOUND_LOG_COMPONENT = "tools"
_UNBOUND_LOG_PHASE = "Profile"
_UNBOUND_LOG_MESSAGE = (
    "Active-profile ContextVar missing inside a multi-user process; refusing the "
    "process-global ACTIVE_PROFILE fallback and resolving to '%s' (no data). A worker "
    "thread is reading the profile without re-binding it — wrap it with run_under_profile()."
) % UNBOUND_PROFILE

# One lost binding means every get_active_profile() call in that worker takes the
# fallback — dozens per tool run. Emitting each one would bury logs/tools/tools.jsonl
# under thousands of copies of a single fact, so per call site only the 1st, 10th,
# 100th ... occurrence is written. Every record carries the running count, so the
# channel is bounded without the diagnostic ever going quiet again.
_unbound_warn_counts: dict[str, int] = {}
_unbound_warn_lock = threading.Lock()
_THIS_FILE = os.path.abspath(__file__)


def _unbound_caller_frames(limit: int = 5) -> list[str]:
    """Return the innermost non-user_profile frames, nearest caller first.

    The docstring's promise is that the offending *worker boundary* is
    diagnosable; "it happened" is not enough to find it, so the record carries
    the call stack that reached the fallback.
    """
    try:
        frames = traceback.StackSummary.extract(
            traceback.walk_stack(None), limit=limit + 6, lookup_lines=False
        )
    except Exception:
        return []
    out = []
    for frame in frames:
        # Skip this module's own plumbing — the caller is what identifies the boundary.
        if os.path.abspath(frame.filename) == _THIS_FILE:
            continue
        out.append(f"{os.path.basename(frame.filename)}:{frame.lineno} in {frame.name}")
        if len(out) >= limit:
            break
    return out


def _should_emit_unbound_warning(count: int) -> bool:
    """Throttle to the 1st, 10th, 100th ... occurrence for one call site."""
    while count >= 10 and count % 10 == 0:
        count //= 10
    return count == 1


def _log_unbound_fallback() -> None:
    """Write the lost-ContextVar fallback to logs/tools/tools.jsonl.

    The agent.logger import is deferred into the call: this module sits near the
    bottom of the import graph and agent.logger reaches agent.utils, which imports
    this module back (also lazily). Keeping the edge out of module scope keeps that
    loop from ever mattering at import time, on a path that is rare by definition.
    Never raises — a broken log sink must not change profile resolution.
    """
    try:
        frames = _unbound_caller_frames()
        caller = frames[0] if frames else "unknown"
        with _unbound_warn_lock:
            count = _unbound_warn_counts.get(caller, 0) + 1
            _unbound_warn_counts[caller] = count
        if not _should_emit_unbound_warning(count):
            return

        from agent.logger import log_to_component

        log_to_component(
            _UNBOUND_LOG_COMPONENT,
            _UNBOUND_LOG_PHASE,
            _UNBOUND_LOG_MESSAGE,
            {
                "resolved_profile": UNBOUND_PROFILE,
                "thread": threading.current_thread().name,
                "caller": caller,
                "stack": frames,
                "occurrences_at_caller": count,
            },
            level=logging.WARNING,
        )
    except Exception:
        pass


def _project_root() -> str:
    return os.path.dirname(os.path.dirname(__file__))


def _user_data_root() -> str:
    return os.path.join(_project_root(), "user_data")


def _profiles_root() -> str:
    return os.path.join(_user_data_root(), "profiles")


def normalize_profile_name(profile_name: str | None) -> str:
    """Return a safe profile name, falling back to default for invalid input."""
    name = str(profile_name or "").strip()
    if not name or name == "default":
        return "default"
    if _PROFILE_NAME_RE.fullmatch(name):
        return name
    return "default"


def _env_truthy(name: str) -> bool:
    return str(os.environ.get(name, "")).strip().lower() in _TRUTHY


def is_demo_mode() -> bool:
    """Return True when the process is running in isolated demo mode."""
    return _env_truthy("DEMO_MODE") or _env_truthy("CAIRNIQ_FORCE_DEMO")


def get_demo_profile_name() -> str:
    """Return the reserved profile used for demo mode."""
    requested = normalize_profile_name(os.environ.get("DEMO_PROFILE") or DEMO_PROFILE_NAME)
    if requested == "default":
        return DEMO_PROFILE_NAME
    return requested


def is_known_profile(profile_name: str | None) -> bool:
    """Return True if the profile is the implicit default or an existing safe profile."""
    name = normalize_profile_name(profile_name)
    if name == DEMO_PROFILE_NAME and not is_demo_mode():
        return False
    if name == "default":
        return True
    return os.path.isdir(os.path.join(_profiles_root(), name))


@log_exceptions()
def enable_multiuser_guard(enabled: bool = True) -> None:
    """Mark this process as serving per-request profiles (multi-user server).

    Once enabled, get_active_profile() will NOT fall back to the process-global
    ACTIVE_PROFILE env var when the request ContextVar is missing — that path is
    only reachable from a worker thread that lost its binding, and the env var is
    shared across users, so using it leaks data between profiles. Instead it
    returns the isolated UNBOUND_PROFILE sentinel and logs the offending call
    stack to the "tools" component channel.

    Called once from server.py at startup. CLI/tests leave it off, preserving the
    historical single-user env-fallback behaviour.
    """
    global _multiuser_guard
    _multiuser_guard = bool(enabled)


def is_multiuser_guard_enabled() -> bool:
    return _multiuser_guard


def get_active_profile() -> str:
    """
    Return the active profile for the current request context.

    Priority:
      1. Value set via set_active_profile() in the current request scope
      2. ACTIVE_PROFILE environment variable  (single-user only; see guard below)
      3. "default"

    When the multi-user guard is enabled (server.py) and no request-scoped
    profile is bound, the process-global ACTIVE_PROFILE env var is intentionally
    NOT used — that would surface one user's portfolio inside another's analysis.
    The call resolves to UNBOUND_PROFILE (isolated, empty) and records the calling
    stack in logs/tools/tools.jsonl so the offending worker boundary is diagnosable.
    """
    ctx_val = _profile_ctx.get()
    if is_demo_mode():
        return get_demo_profile_name()
    if ctx_val:
        profile = normalize_profile_name(ctx_val)
        return "default" if profile == DEMO_PROFILE_NAME else profile

    if _multiuser_guard:
        # No ContextVar in a multi-user process => a detached worker lost the
        # request profile. Fail safe to an empty, isolated profile rather than
        # leaking whichever profile last wrote the shared env var.
        _log_unbound_fallback()
        return UNBOUND_PROFILE

    profile = normalize_profile_name(os.environ.get("ACTIVE_PROFILE"))
    return "default" if profile == DEMO_PROFILE_NAME else profile


@log_exceptions()
def set_active_profile(profile_name: str) -> object:
    """
    Set the active profile for the current async task / thread context.
    Returns the ContextVar Token so the caller can reset it if needed.

    Usage (FastAPI middleware):
        token = set_active_profile("default")
        ...
        _profile_ctx.reset(token)
    """
    return _profile_ctx.set(normalize_profile_name(profile_name))


@log_exceptions()
def reset_profile(token) -> None:
    """Reset the profile context to its previous value (use with set_active_profile token)."""
    _profile_ctx.reset(token)


def run_under_profile(profile: str, target, *args, **kwargs):
    """Run ``target`` with ``profile`` bound for the duration of the call.

    The active profile lives in a ContextVar that does NOT propagate into a
    manually-created threading.Thread, nor into a ThreadPoolExecutor worker.
    Any code that submits portfolio-dependent work to such a worker must capture
    get_active_profile() in the request scope and re-apply it inside the worker,
    or the work resolves the wrong (or, under the multi-user guard, the empty)
    profile. This is the canonical wrapper for that; pass it as the executor
    callable, e.g. ``executor.submit(run_under_profile, prof, fn, *args)``.
    """
    token = set_active_profile(profile)
    try:
        return target(*args, **kwargs)
    finally:
        reset_profile(token)


@contextmanager
@log_exceptions()
def profile_context(profile_name: str) -> Generator[None, None, None]:
    """Context manager to temporarily activate a profile."""
    token = set_active_profile(profile_name)
    try:
        yield
    finally:
        reset_profile(token)


@log_exceptions()
def list_available_profiles() -> list[dict]:
    """
    Discover all named profiles in user_data/profiles/ plus the implicit 'default'.
    Returns a list of dicts: [{name, display_name, data_path}]
    """
    user_data_root = _user_data_root()
    profiles_root = _profiles_root()

    results = []

    if is_demo_mode():
        demo_name = get_demo_profile_name()
        demo_path = os.path.join(profiles_root, demo_name)
        os.makedirs(demo_path, exist_ok=True)
        return [{
            "name": demo_name,
            "display_name": "Demo",
            "data_path": demo_path,
            "active": True,
            "is_demo": True,
        }]

    # Always include 'default' (maps to project root)
    results.append({
        "name": "default",
        "display_name": "Default",
        "data_path": user_data_root,
        "active": get_active_profile() == "default"
    })

    # Discover subdirectories in user_data/profiles/
    if os.path.isdir(profiles_root):
        for entry in sorted(os.scandir(profiles_root), key=lambda e: e.name):
            if entry.name == DEMO_PROFILE_NAME:
                continue
            if entry.is_dir() and not entry.name.startswith(".") and normalize_profile_name(entry.name) == entry.name:
                results.append({
                    "name": entry.name,
                    "display_name": entry.name.replace("_", " ").title(),
                    "data_path": entry.path,
                    "active": get_active_profile() == entry.name
                })

    return results


def _write_demo_portfolio(portfolio_path: str) -> None:
    example_path = os.path.join(_project_root(), "demo_portfolio.example.csv")
    legacy_path = os.path.join(_user_data_root(), "demo_portfolio.csv")
    source_path = example_path if os.path.exists(example_path) else legacy_path
    if os.path.exists(source_path):
        shutil.copyfile(source_path, portfolio_path)
        return

    with open(portfolio_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Symbol", "Shares", "Purchase Price", "Current Price", "Account", "Currency", "Return Pct", "Asset Type"])
        writer.writerow(["AAPL", "30", "170.00", "195.00", "Demo Brokerage", "USD", "", "Public"])
        writer.writerow(["MSFT", "25", "380.00", "430.00", "Demo Brokerage", "USD", "", "Public"])
        writer.writerow(["NVDA", "60", "90.00", "115.00", "Demo Brokerage", "USD", "", "Public"])
        writer.writerow(["AMZN", "40", "175.00", "190.00", "Demo Brokerage", "USD", "", "Public"])
        writer.writerow(["GOOGL", "25", "165.00", "175.00", "Demo Brokerage", "USD", "", "Public"])
        writer.writerow(["SPY", "45", "500.00", "520.00", "Demo Brokerage", "USD", "", "Public"])
        writer.writerow(["VTI", "60", "240.00", "255.00", "Demo Brokerage", "USD", "", "Public"])
        writer.writerow(["CASH.TO", "800", "50.00", "50.00", "Demo Cash", "CAD", "", "Public"])
        writer.writerow(["XIC.TO", "500", "34.00", "38.00", "Demo TFSA", "CAD", "", "Public"])
        writer.writerow(["VCN.TO", "400", "45.00", "48.00", "Demo TFSA", "CAD", "", "Public"])
        writer.writerow(["PENSION-DEMO", "59000", "1.00", "1.00", "Demo Pension", "CAD", "5.5", "Private"])


def _write_demo_history(history_path: str) -> None:
    """Seed enough demo history for the benchmark chart to render immediately."""
    fieldnames = [
        "date",
        "total_value_cad",
        "total_value_usd",
        "invested_cad",
        "invested_usd",
        "percent_return",
    ]
    end_date = date.today()
    fx_rate = 1.44
    values_cad = [
        230800, 231150, 230600, 232250, 233100, 232650, 234000, 235200,
        236350, 235900, 237500, 238150, 239250, 238700, 237850, 239600,
        240450, 241200, 240850, 242300, 243100, 242650, 244000, 244750,
        243900, 245250, 246000, 245600, 246550, 247100, 246700, 247050,
        247450, 246900, 246962,
    ]

    with open(history_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx, value_cad in enumerate(values_cad):
            days_ago = len(values_cad) - idx - 1
            invested_cad = 229500 + (idx * 160)
            value_usd = value_cad / fx_rate
            invested_usd = invested_cad / fx_rate
            percent_return = ((value_cad - invested_cad) / invested_cad) * 100
            writer.writerow({
                "date": (end_date - timedelta(days=days_ago)).isoformat(),
                "total_value_cad": f"{value_cad:.2f}",
                "total_value_usd": f"{value_usd:.2f}",
                "invested_cad": f"{invested_cad:.2f}",
                "invested_usd": f"{invested_usd:.2f}",
                "percent_return": f"{percent_return:.4f}",
            })


def _demo_memory_template() -> dict:
    return {
        "user_profile": {
            "name": "Demo User",
            "age": "38",
            "risk_tolerance": "Moderate",
            "retirement_age": "60",
            "annual_income": "$150,000",
            "investment_goals": ["Retirement readiness", "Diversified long-term growth"],
            "accounts": ["Demo Brokerage", "Demo Pension"],
            "base_currency": "USD",
            "last_updated": None,
        },
        "key_facts": [
            "This is a disposable demo profile.",
            "The demo portfolio is sample data and not tied to a real person.",
        ],
        "conversation_summaries": [],
        "past_recommendations": [],
        "active_theses": [],
        "lessons_learned": [],
    }


@log_exceptions()
def ensure_demo_profile(reset: bool | None = None) -> str:
    """
    Create or refresh the isolated demo profile.

    Demo state lives under user_data/profiles/<demo_profile>/ so normal
    user_data files remain untouched. When reset=True, mutable demo identity
    and session files are returned to known sample data.
    """
    demo_name = get_demo_profile_name()
    profile_dir = os.path.join(_profiles_root(), demo_name)
    os.makedirs(profile_dir, exist_ok=True)

    if reset is None:
        reset = _env_truthy("DEMO_RESET")

    reset_files = [
        "chat_history.json",
        "knowledge_graph.json",
        "checkpoints.sqlite",
        "checkpoints.sqlite-shm",
        "checkpoints.sqlite-wal",
        "portfolio_history.csv",
        "demo_portfolio_history.csv",
        "trade_journal.json",
        "feedback.json",
    ]
    if reset:
        for filename in reset_files:
            path = os.path.join(profile_dir, filename)
            if os.path.exists(path):
                os.remove(path)

    portfolio_path = os.path.join(profile_dir, "my_portfolio.csv")
    if reset or not os.path.exists(portfolio_path):
        _write_demo_portfolio(portfolio_path)

    memory_path = os.path.join(profile_dir, "user_memory.json")
    if reset or not os.path.exists(memory_path):
        write_json_atomic(memory_path, _demo_memory_template())

    chat_path = os.path.join(profile_dir, "chat_history.json")
    if reset or not os.path.exists(chat_path):
        write_json_atomic(chat_path, {"sessions": []})

    history_path = os.path.join(profile_dir, "demo_portfolio_history.csv")
    if reset or not os.path.exists(history_path):
        _write_demo_history(history_path)

    return profile_dir


@log_exceptions()
def get_data_path(filename: str) -> str:
    """
    Get the absolute path for a data file specific to the active profile.
    Automatically creates the profile directory if it does not exist.
    """
    profile_name = normalize_profile_name(get_active_profile())

    user_data_root = _user_data_root()

    if profile_name == "default":
        # Consolidated user data lives in user_data/
        profile_dir = user_data_root
    else:
        profile_dir = os.path.join(user_data_root, "profiles", profile_name)

    # Create the directory if it doesn't exist
    os.makedirs(profile_dir, exist_ok=True)

    base_path = os.path.abspath(profile_dir)
    target_path = os.path.abspath(os.path.join(profile_dir, filename))
    if os.path.commonpath([base_path, target_path]) != base_path:
        raise ValueError(f"Data path escapes active profile directory: {filename}")

    return target_path
