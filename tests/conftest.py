import os
import re
import shutil

# langchain_core.language_models.base does a bare `from transformers import
# GPT2TokenizerFast` inside a try/except ImportError, purely for a legacy token-counting
# fallback this project never calls. When transformers happens to be installed that
# costs ~3.2s of cold import on EVERY pytest invocation — the single biggest tax on
# running one test file. transformers is not in requirements.txt and nothing declares it
# as a dependency (`pip show transformers` lists no Required-by), so CI never pays this;
# only dev venvs that picked it up transitively do. Poison the import so langchain_core
# takes its own ImportError branch. Same trick dspy_setup.py already uses on
# IPython.display, and equally safe: no repo module imports transformers.
import sys as _sys

import net_guard  # tests/ is on sys.path — it has no __init__.py, so pytest prepends it
import pytest
from dotenv import load_dotenv

if "transformers" not in _sys.modules:
    # A None entry in sys.modules makes `import transformers` raise ImportError,
    # which is exactly the branch langchain_core already handles.
    _sys.modules["transformers"] = None

# Force the secrets store into test mode for the ENTIRE session — including pytest's
# collection phase. secrets_store._in_test_mode() normally keys off PYTEST_CURRENT_TEST
# and the per-test pytest_ profile, but neither is set yet during collection. Some test
# modules import agent.utils at the top level, which runs load_secrets_into_env() at
# import time and would otherwise probe the real OS keychain before the guard activates.
# conftest is imported before any test module, so setting this here closes that gap.
# setdefault() preserves an explicit override if one is already exported.
os.environ.setdefault("CAIRNIQ_DISABLE_KEYRING", "1")

# Automatically load the environment variables from user_data/.env at collection time
project_root = os.path.dirname(os.path.dirname(__file__))
env_path = os.path.join(project_root, "user_data", ".env")

if os.path.exists(env_path):
    _original_env = {k: v for k, v in os.environ.items() if v}
    load_dotenv(env_path, override=True)
    for k, v in _original_env.items():
        if not os.environ.get(k):
            os.environ[k] = v

# Always provide a fallback to prevent agent.utils from crashing at import time
if not os.environ.get("AIDLC_MODEL_ID"):
    os.environ["AIDLC_MODEL_ID"] = "test-model-id"
if not os.environ.get("LLM_PROVIDER"):
    os.environ["LLM_PROVIDER"] = "test-provider"

# Tests default to auth DISABLED — CI has no user_data/.env, so auth_required() is
# off there. A developer's real user_data/.env may set CAIRNIQ_AUTH_REQUIRED=1, and
# server.py runs load_dotenv(override=True) at import — which would re-set the flag
# even after a plain pop here. So import the app now (after the AIDLC fallbacks it
# needs at import) to run that load, THEN clear the flag for the session. Tests that
# exercise enforcement re-enable it per-test via monkeypatch.setenv.
try:
    import server  # noqa: F401,E402 — force server.py's import-time load_dotenv to run
except Exception:
    pass
# Force OFF with an explicit "0" rather than popping: tools.questrade/tools.alpaca
# run load_dotenv(env_path) WITHOUT override at import (pulled in when the
# disable_external_brokers fixture stubs them), and a no-override load RE-ADDS a
# *missing* var from the dev's .env — so a pop would let auth=1 sneak back. An
# existing "0" is left untouched by override=False loads. Enforcement tests set "1".
os.environ["CAIRNIQ_AUTH_REQUIRED"] = "0"

# Install the offline guard once, before any test module imports. See tests/net_guard.py
# for why: the suite was spending half its wall clock on live Yahoo Finance calls that
# nobody asked for, and inheriting Yahoo's uptime as a test dependency.
net_guard.install()


@pytest.fixture(autouse=True)
def offline_by_default(request):
    """Block real sockets unless the test is marked @pytest.mark.allow_network."""
    allowed = request.node.get_closest_marker("allow_network") is not None
    net_guard.set_enabled(not allowed)
    net_guard.set_current_test(request.node.nodeid)
    try:
        yield
    finally:
        net_guard.set_enabled(True)


@pytest.fixture(autouse=True)
def isolate_llm_budget(tmp_path, monkeypatch):
    """Isolate LLM budget file during tests to prevent interference from real runs."""
    import agent.llm_budget
    temp_budget_file = tmp_path / "llm_budget.json"
    monkeypatch.setattr(agent.llm_budget, "_STATE_PATH", str(temp_budget_file))


@pytest.fixture(autouse=True)
def isolate_global_stores(tmp_path, monkeypatch):
    """Redirect the deliberately-global user_data stores into the test's tmp dir.

    These two are module-level absolute paths rather than get_data_path() calls —
    correctly so, since neither fact belongs to a profile — which also means the
    per-test profile isolation below does not reach them, and every test that
    touched a scan or an engine tick wrote the user's real file:

    - `_SCAN_LEDGER_PATH` decides which tail names get exploration slots in a live
      broad scan. Measured 2026-07-30: 7 tickers in the real ledger carried the
      suite's run date.
    - `_HEARTBEAT_PATH` is the ops view 2.5/2.6 were built to make trustworthy. A
      test writing a production count into it is a test editing the instrument that
      says whether an engine is alive.

    Redirected centrally rather than per-test (test_dynamic_universe.py already did
    the ledger locally) because the leak belongs to any test that reaches the code,
    not to the tests that are about it. Same shape as isolate_llm_budget above.
    """
    import tools.engine_heartbeat as hb
    import tools.opportunity_scanner as opp
    monkeypatch.setattr(opp, "_SCAN_LEDGER_PATH", str(tmp_path / "funnel_scan_ledger.json"))
    monkeypatch.setattr(hb, "_HEARTBEAT_PATH", str(tmp_path / "engine_heartbeat.json"))


class _RealJournalWriteBlocked(BaseException):
    """A `real_trade_journal_path` test tried to WRITE the journal it inspects.

    Deliberately a BaseException rather than an Exception. The journal's writers are
    reached through `@log_exceptions()` tools and API handlers that wrap calls in
    `except Exception` so a bad write can never break a caller — the same shape as
    engine_heartbeat._save and opportunity_scanner._update_scan_ledger. A guard
    raised as an Exception would be caught by one of those handlers, logged, and
    reported as success, which is precisely the silence that let 117 synthetic
    trades pile up unnoticed. This one cannot be swallowed on the way out.
    """


@pytest.fixture(autouse=True)
def isolate_trade_journal(request, tmp_path, monkeypatch):
    """Point the trade journal at the test's own file.

    `_journal_file()` resolves through get_data_path(), which is profile-scoped and
    right — but `profile_middleware` re-binds every TestClient request to the real
    default profile, so the API journal tests wrote the default profile's REAL
    journal. Measured 2026-07-30: **117 synthetic `TEST_TICKER` trades**, dated
    07-23 → 07-30, 15–30 per day, in `user_data/trade_journal.json`. An append is
    invisible to the test that makes it, so nothing ever failed.

    Patched at the path seam rather than snapshot/restored in protect_real_user_data,
    because that fixture is racy across xdist workers for a file many tests touch —
    when this was tried there first, the 117-entry file came back empty.

    **`real_trade_journal_path` opts a test out of the redirect, not out of the
    isolation.** Patching the path seam makes the redirect indistinguishable, from
    inside the process, from the very leak an architecture test asserts is absent:
    `_journal_file()` returning a tmp path reads identically to it returning a path
    outside user_data. A test whose subject IS the resolver therefore has to see the
    production one. To keep that opt-out from re-opening the hole this fixture
    closed, the marked branch blocks the two functions that touch the filesystem
    instead — resolution stays real, writing is impossible. Blocking the write seam
    is also race-free, which snapshot/restore is not (see protect_real_user_data).
    """
    import tools.trade_journal as tj

    if request.node.get_closest_marker("real_trade_journal_path"):
        def _refuse_write(*args, **kwargs):
            raise _RealJournalWriteBlocked(
                "real_trade_journal_path is for asserting on path resolution only, "
                "but this test called a trade-journal writer while pointed at the "
                "real user_data journal. Drop the marker, or stop writing."
            )

        monkeypatch.setattr(tj, "_save_journal", _refuse_write)
        monkeypatch.setattr(tj, "_ensure_data_dir", _refuse_write)
        return

    journal = tmp_path / "trade_journal.json"
    monkeypatch.setattr(tj, "_journal_file", lambda: str(journal))
    # _load_journal falls through to the legacy repo-level data/trade_journal.json
    # when the profile file is absent; keep the fallthrough out of tests too.
    monkeypatch.setattr(tj, "_LEGACY_JOURNAL_FILE", str(tmp_path / "legacy_trade_journal.json"))


@pytest.fixture(autouse=True)
def isolated_test_profile(request):
    """Run each test in an isolated profile context by default."""
    from tools.user_profile import get_data_path, reset_profile, set_active_profile

    safe_name = re.sub(r"[^A-Za-z0-9_]+", "_", request.node.name)[:60]
    profile_name = f"pytest_{safe_name}"
    token = set_active_profile(profile_name)
    profile_dir = os.path.dirname(get_data_path("__cleanup__"))

    try:
        yield
    finally:
        reset_profile(token)
        if os.path.basename(profile_dir).startswith("pytest_") and os.path.isdir(profile_dir):
            shutil.rmtree(profile_dir, ignore_errors=True)


@pytest.fixture(autouse=True)
def protect_real_user_data():
    """Guard the user's real config files from test mutations.

    Tests that POST to /api/settings/save exercise the save handler, which writes
    non-secret config (e.g. BASE_CURRENCY) to os.getcwd()/user_data/.env AND, via
    update_profile under the request's middleware-resolved default profile, to
    user_data/user_memory.json — the user's REAL files (the isolated_test_profile
    fixture is bypassed because profile_middleware re-resolves the profile per
    request). Snapshot these files and restore them after each test so a suite run
    can never silently clobber the user's settings (this is exactly how the user's
    currency kept resetting to USD).

    KNOWN LIMITS, measured 2026-07-30 — snapshot/restore is the weaker of the two
    isolation patterns and this list should not grow:

    - **It is racy under `-n auto`.** Workers run in separate processes with their
      own copy of this fixture, so for a file that several tests touch, worker A can
      read it inside worker B's truncate window and then restore what it read.
      Adding trade_journal.json here did exactly that once — a 117-entry file came
      back 0 bytes. The journal is isolated at its own seam instead (see
      isolate_trade_journal); prefer redirecting the PATH over restoring the FILE.
    - **A file that did not exist beforehand is not cleaned up**, because deleting
      it would race the same way: another worker may have legitimately created it.

    Both are tolerable for these two entries only because settings tests are the
    only writers and they are few.
    """
    targets = [
        os.path.join(project_root, "user_data", ".env"),
        os.path.join(project_root, "user_data", "user_memory.json"),
    ]
    originals = {}
    for path in targets:
        if os.path.exists(path):
            with open(path, encoding="utf-8") as fh:
                originals[path] = fh.read()
    try:
        yield
    finally:
        for path, content in originals.items():
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(content)


@pytest.fixture(autouse=True)
def disable_external_brokers(monkeypatch):
    """Keep unit tests from reaching live brokerage APIs."""

    # These model the DEFAULT production state — an unlinked broker — not a healthy
    # configured one. The previous stubs returned a clean {"holdings": [], "errors": []}
    # that no unconfigured profile has ever actually produced, which is why the whole
    # sentinel/sync-error path went unexercised by 1900+ tests.
    class StubQuestradeAPI:
        def __init__(self, *args, **kwargs):
            self.enabled = False
            self.clients = []

        def get_all_holdings(self):
            return {"holdings": [], "errors": [],
                    "notices": ["Questrade integration is disabled in Settings."]}

    class StubAlpacaAPI:
        def __init__(self, *args, **kwargs):
            pass

        def is_configured(self):
            return False

        def get_aggregated_holdings(self):
            return {"holdings": [], "errors": [], "notices": ["Alpaca not configured"]}

    monkeypatch.setattr("tools.questrade.QuestradeAPI", StubQuestradeAPI)
    monkeypatch.setattr("tools.alpaca.AlpacaAPI", StubAlpacaAPI)
