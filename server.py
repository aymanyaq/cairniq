import asyncio
import logging
import os
import time
from datetime import datetime
from typing import Any
from urllib.parse import quote, urlparse

# Increase file descriptor limit for macOS/Linux to prevent
# [Errno 24] Too many open files. The 'resource' module is POSIX-only,
# so the import is inside the try/except - on Windows the ImportError is
# swallowed and we skip this block (Windows uses a different I/O model
# and does not have RLIMIT_NOFILE).
try:
    import resource
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    # Target 4096 or the system hard limit, whichever is smaller
    target = min(4096, hard) if hard > 0 else 4096
    if soft < target:
        resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
except Exception:
    pass

# Fix SSL certificate verification for Python 3.13 + macOS Homebrew
import ssl as _ssl

_ca_file = _ssl.get_default_verify_paths().cafile
if _ca_file and os.path.isfile(_ca_file):
    os.environ.setdefault('SSL_CERT_FILE', _ca_file)
    os.environ.setdefault('REQUESTS_CA_BUNDLE', _ca_file)

from dotenv import load_dotenv
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Save original non-empty environment variables before load_dotenv to prevent
# empty keys in the .env file from overwriting valid OS-level environment variables.
_original_env = {k: v for k, v in os.environ.items() if v}

# Load Environment Variables from user_data/
_ENV_DEMO_MODE_REQUESTED = os.environ.get("DEMO_MODE")
_ENV_FORCE_DEMO_REQUESTED = os.environ.get("CAIRNIQ_FORCE_DEMO")
env_path = os.path.join(os.getcwd(), "user_data", ".env")
load_dotenv(env_path, override=True)

# Restore any non-empty environment variables that were overwritten with empty/null values
for k, v in _original_env.items():
    if not os.environ.get(k):
        os.environ[k] = v

# Move any plaintext secrets out of .env into the OS keychain, then hydrate
# os.environ with whatever the keychain holds. After this point the rest of
# the codebase can keep reading os.environ.get("OPENAI_API_KEY") etc.
from tools.secrets_store import (  # noqa: E402
    clear_incompatible_aws_session_token,
    keyring_status,
    load_secrets_into_env,
    migrate_env_to_keyring,
)

try:
    _ks = keyring_status()
    _migration = migrate_env_to_keyring(env_path)
    _n_loaded = load_secrets_into_env()
    # Strip any stale AWS_SESSION_TOKEN that's incompatible with the long-term
    # IAM user key we just loaded — see secrets_store.clear_incompatible_aws_session_token.
    _aws_token_cleanup = clear_incompatible_aws_session_token()

    if not _ks["available"]:
        # Headless Linux / Docker / no-backend Mac — the app still works from .env
        # via the fallback path, but flag it so users know secrets aren't encrypted.
        from agent.logger import log_to_component
        log_to_component(
            "server",
            "secrets",
            f"keychain unavailable ({_ks['reason']}); using plaintext .env fallback",
            level=logging.WARNING,
        )
    elif _migration.get("migrated") or _n_loaded:
        # Only emit a startup line when something actually happened, to keep logs quiet.
        _summary = []
        if _migration.get("migrated"):
            _summary.append(f"migrated {len(_migration['migrated'])} from .env")
        if _n_loaded:
            _summary.append(f"hydrated {_n_loaded} from keychain")
        if _aws_token_cleanup["cleared"]:
            _summary.append("stripped stale AWS_SESSION_TOKEN")
        from agent.logger import log_to_component
        log_to_component("server", "secrets", f"{', '.join(_summary)}")
except Exception as _secrets_err:  # noqa: BLE001 — never block server startup on this
    from agent.logger import log_to_component
    log_to_component("server", "secrets", f"non-fatal init issue: {_secrets_err}", level=logging.ERROR)

if str(_ENV_DEMO_MODE_REQUESTED or "").strip().lower() in {"1", "true", "yes", "y", "on"}:
    os.environ["DEMO_MODE"] = "true"
if str(_ENV_FORCE_DEMO_REQUESTED or "").strip().lower() in {"1", "true", "yes", "y", "on"}:
    os.environ["CAIRNIQ_FORCE_DEMO"] = "true"
    os.environ["DEMO_MODE"] = "true"

from contextlib import asynccontextmanager

from agent.graph import build_graph
from agent.logger import bind_log_context, log_to_component, reset_log_context
from agent.version import __version__
from api.dependencies import get_connection_manager, set_agent
from api.routers import (
    alerts,
    auth,
    chat,
    dashboard,
    feedback,
    journal,
    memory,
    news,
    pages,
    portfolio,
    reports,
    settings,
)
from tools.auth import auth_required, extract_bearer, verify_token
from tools.user_profile import (
    enable_multiuser_guard,
    ensure_demo_profile,
    get_demo_profile_name,
    is_demo_mode,
    is_known_profile,
    normalize_profile_name,
    reset_profile,
    set_active_profile,
)

# This process serves a per-request profile (resolved by profile_middleware), so
# the process-global ACTIVE_PROFILE env var is unsafe as a fallback for any
# worker thread that loses its ContextVar — it would surface one household
# member's portfolio inside another's analysis. Refuse that fallback; lost
# bindings resolve to the isolated empty profile instead. Demo mode is exempt:
# it is single-profile by construction and relies on the env var.
if not is_demo_mode():
    enable_multiuser_guard()

if is_demo_mode():
    os.environ["DEMO_MODE"] = "true"
    os.environ["ACTIVE_PROFILE"] = get_demo_profile_name()
    os.environ["QUESTRADE_ENABLED"] = "false"
    os.environ["ALPACA_PAPER_MODE"] = "true"
    ensure_demo_profile()

def check_requirements():
    import importlib.metadata
    import re

    req_file = os.path.join(os.getcwd(), "requirements.txt")
    if not os.path.exists(req_file):
        return

    try:
        with open(req_file) as f:
            lines = f.readlines()

        mismatched = []
        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            match = re.match(r"^([a-zA-Z0-9_\-]+)", line)
            if match:
                pkg_name = match.group(1)
                try:
                    importlib.metadata.version(pkg_name)
                except importlib.metadata.PackageNotFoundError:
                    mismatched.append(pkg_name)

        if mismatched:
            print(
                f"\n\n======================================================================\n"
                f"🚨 WARNING: The following dependencies in requirements.txt are missing:\n"
                f"   {', '.join(mismatched)}\n\n"
                f"   Please run './install.sh' to update your environment!\n"
                f"======================================================================\n"
            )
            log_to_component(
                "server",
                "Startup",
                f"Missing dependencies in virtual environment: {', '.join(mismatched)}",
                level=logging.WARNING
            )
    except Exception as e:
        log_to_component("server", "Startup", f"Failed to verify requirements: {e}", level=logging.WARNING)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Check for missing dependencies on startup
    check_requirements()

    # Register LiteLLM callback to track DSPy LLM costs
    try:
        import litellm
        from litellm.integrations.custom_logger import CustomLogger

        from agent.cost_tracker import accumulate_cost as _acc_cost

        class _CostCallback(CustomLogger):
            def log_success_event(self, kwargs, response_obj, start_time, end_time):
                try:
                    usage = getattr(response_obj, "usage", None)
                    if usage is None and isinstance(response_obj, dict):
                        usage = response_obj.get("usage")
                    if usage:
                        inp = getattr(usage, "prompt_tokens", 0) or (usage.get("prompt_tokens", 0) if isinstance(usage, dict) else 0)
                        out = getattr(usage, "completion_tokens", 0) or (usage.get("completion_tokens", 0) if isinstance(usage, dict) else 0)
                        cache_read = 0
                        cache_detail = getattr(usage, "prompt_tokens_details", None)
                        if cache_detail:
                            cache_read = getattr(cache_detail, "cached_tokens", 0) or 0
                        elif isinstance(usage, dict):
                            cd = usage.get("prompt_tokens_details") or {}
                            cache_read = cd.get("cached_tokens", 0) or 0
                        model = kwargs.get("model", "") or ""
                        if inp or out:
                            _acc_cost(inp, out, model, cache_read_tokens=cache_read)
                except Exception as e:
                    log_to_component("server", "Error", f"Error in callback parsing: {e}", level=logging.ERROR)

        litellm.callbacks.append(_CostCallback())
    except Exception as e:
        log_to_component("server", "Startup", f"Error registering callback: {e}", level=logging.WARNING)

    log_to_component("server", "Startup", "Initializing LangGraph agent...")
    try:
        new_agent = build_graph(use_memory=True)
        set_agent(new_agent)
        log_to_component("server", "Startup", "Agent initialized successfully.")
    except Exception as e:
        log_to_component("server", "Startup", f"FAILED to initialize agent: {e}", level=logging.ERROR)

    # Wire the knowledge-graph save hook to the WebSocket broadcast so the Context
    # Graph view refreshes live when the graph changes (portfolio sync, agent
    # "connect-the-dots"). Without this the page only reflects the graph as of load.
    try:
        import asyncio as _asyncio

        from api.dependencies import broadcast_graph_update, set_main_loop
        from tools.graph_memory import graph_memory
        set_main_loop(_asyncio.get_running_loop())
        graph_memory.on_save_callback = broadcast_graph_update
        log_to_component("server", "Startup", "Knowledge-graph live sync wired.")
    except Exception as e:
        log_to_component("server", "Startup", f"Graph live-sync wiring skipped: {e}", level=logging.WARNING)

    # Clean old cache files at startup
    try:
        from tools.daily_cache import cleanup_old
        removed = cleanup_old(7)
        if removed:
            log_to_component("server", "Startup", f"Cleaned {removed} old cache files")
    except Exception as e:
        log_to_component("server", "Startup", f"Error cleaning old cache files: {e}", level=logging.WARNING)

    # Seed user_data/funnel_config.json from the example on first run (cross-platform;
    # covers manual venv installs that don't run install.ps1). Fail-safe — the scanner
    # uses built-in defaults if anything goes wrong.
    try:
        from tools.opportunity_scanner import seed_funnel_config_if_missing
        if seed_funnel_config_if_missing():
            log_to_component("server", "Startup", "Seeded user_data/funnel_config.json from example")
    except Exception as e:
        log_to_component("server", "Startup", f"Error seeding funnel config: {e}", level=logging.WARNING)

    # Fetch live USD_TO_CAD exchange rate asynchronously to not block startup
    async def fetch_exchange_rate():
        try:
            import yfinance as yf

            from tools.daily_cache import set_cached
            def _fetch():
                ticker = yf.Ticker("USDCAD=X")
                hist = ticker.history(period="5d", timeout=40)
                if not hist.empty:
                    return float(hist["Close"].iloc[-1])
                return None

            rate = await asyncio.to_thread(_fetch)
            if rate:
                os.environ["USD_TO_CAD"] = str(rate)
                # Startup runs with no request profile bound; the daily cache is
                # profile-namespaced, so home this profile-independent rate in
                # 'default' rather than the empty '_unbound' profile (guard).
                from tools.user_profile import run_under_profile
                run_under_profile("default", set_cached, "usd_cad_rate", rate)
                log_to_component("server", "Startup", f"Injected live USD/CAD rate: {rate:.4f}")
        except Exception as e:
            log_to_component("server", "Startup", f"Failed to fetch live USD/CAD rate: {e}", level=logging.WARNING)

    asyncio.create_task(fetch_exchange_rate())

    # Pre-warm the dashboard caches so the first visit after a restart isn't the
    # one that pays for them (31.7s cold vs 12ms warm, measured). Fire-and-forget
    # for the same reason as the rate fetch above — it must not delay binding the
    # port. Sequenced after it so the warm's quote conversions find the live rate.
    async def warm_dashboard_caches():
        try:
            from tools.cache_warm import warm_all_profiles, warm_enabled
            if not warm_enabled():
                return
            results = await asyncio.to_thread(warm_all_profiles)
            warmed = sum(1 for r in results.values() if r.get("summary") or r.get("radar"))
            log_to_component("server", "Startup", f"Dashboard caches warmed for {warmed} profile(s)")
        except Exception as e:
            log_to_component("server", "Startup", f"Dashboard cache warm skipped: {e}", level=logging.WARNING)

    asyncio.create_task(warm_dashboard_caches())

    # Start the in-process scheduler for recurring background tasks
    # (exchange rate refresh, portfolio snapshots, cache cleanup, etc.)
    try:
        from tools.scheduler import scheduler
        await scheduler.start()
        log_to_component("server", "Startup", "Background scheduler started")
    except Exception as e:
        log_to_component("server", "Startup", f"Scheduler failed to start: {e}", level=logging.WARNING)

    yield

    # Teardown: gracefully stop the scheduler
    try:
        from tools.scheduler import scheduler
        await scheduler.stop()
        log_to_component("server", "Shutdown", "Background scheduler stopped")
    except Exception as e:
        log_to_component("server", "Shutdown", f"Scheduler stop error: {e}", level=logging.WARNING)

app = FastAPI(title="CairnIQ API", lifespan=lifespan)


def _configured_allowed_origins() -> list[str]:
    """Return browser origins allowed to send credentialed requests."""
    raw = os.environ.get("CAIRNIQ_ALLOWED_ORIGINS") or os.environ.get("CORS_ALLOW_ORIGINS")
    if raw:
        origins = [origin.strip().rstrip("/") for origin in raw.split(",") if origin.strip()]
    else:
        origins = [
            "http://localhost:8000",
            "http://127.0.0.1:8000",
        ]
        # Extra origins (e.g. a LAN hostname such as http://myhost:8000) come from
        # env so machine-specific names stay out of the (public) repo. Comma-sep;
        # set CAIRNIQ_EXTRA_ORIGINS in user_data/.env (gitignored).
        extra = os.environ.get("CAIRNIQ_EXTRA_ORIGINS", "")
        origins += [o.strip().rstrip("/") for o in extra.split(",") if o.strip()]
    return [origin for origin in origins if origin != "*"]


ALLOWED_ORIGINS = _configured_allowed_origins()
_ALLOWED_ORIGIN_SET = set(ALLOWED_ORIGINS)
_MUTATING_METHODS = {"POST", "PUT", "PATCH", "DELETE"}


def _origin_from_url(value: str) -> str | None:
    parsed = urlparse(value)
    if not parsed.scheme or not parsed.netloc:
        return None
    return f"{parsed.scheme}://{parsed.netloc}".rstrip("/")


def _is_allowed_origin(origin: str | None) -> bool:
    if not origin:
        return True
    return origin.rstrip("/") in _ALLOWED_ORIGIN_SET


def _resolve_request_profile(cookie_profile: str | None, env_profile: str | None) -> str:
    if is_demo_mode():
        return get_demo_profile_name()

    for raw_profile in (cookie_profile, env_profile):
        raw_name = str(raw_profile or "").strip()
        if not raw_name:
            continue
        profile = normalize_profile_name(raw_profile)
        if profile == "default" and raw_name != "default":
            continue
        if is_known_profile(profile):
            return profile
    return "default"

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Compress HTML/JS/JSON on the way out. The pages are 76-96KB of markup and
# no_cache_html_middleware (below) deliberately forbids caching them, so the full
# payload is re-sent on every navigation — over a VPN that is the whole cost of a
# page load. Text compresses 4-6x.
#
# Streaming responses opt out by setting `Content-Encoding: identity` themselves
# (see the chat endpoint): Starlette's gzip responder does not flush per chunk, so
# compressing an incremental stream would batch it. Responses under minimum_size
# are passed through untouched.
app.add_middleware(GZipMiddleware, minimum_size=1000)

@app.middleware("http")
async def no_cache_html_middleware(request: Request, call_next):
    """Tell browsers never to cache HTML page responses.

    Without this, Edge (and Chrome on return visits) serve stale HTML from
    the disk cache, which means users run old embedded JavaScript even after
    the server has been updated.  API/JSON responses and static assets are
    unaffected — only text/html responses get the no-cache directive.
    """
    response = await call_next(request)
    ct = response.headers.get("content-type", "")
    if "text/html" in ct:
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response

@app.middleware("http")
async def local_origin_middleware(request: Request, call_next):
    """Reject credentialed browser writes from untrusted origins."""
    if request.method.upper() in _MUTATING_METHODS:
        origin = request.headers.get("origin")
        referer_origin = _origin_from_url(request.headers.get("referer", ""))
        if not _is_allowed_origin(origin) or (referer_origin and not _is_allowed_origin(referer_origin)):
            return JSONResponse({"error": "Origin not allowed"}, status_code=403)
    return await call_next(request)

# Paths reachable without a valid token even when auth enforcement is on.
# The /api/auth/* endpoints self-handle authentication; static assets and the
# OpenAPI surface stay open so the iOS client can fetch the API contract.
# /login is the browser login page — must be reachable to authenticate.
_PUBLIC_PATH_PREFIXES = (
    "/api/auth",
    "/api/health",
    "/login",
    "/static",
    "/openapi.json",
    "/docs",
    "/redoc",
    "/favicon",
)


def _is_public_path(path: str) -> bool:
    return any(path == prefix or path.startswith(prefix) for prefix in _PUBLIC_PATH_PREFIXES)


@app.middleware("http")
async def profile_middleware(request: Request, call_next):
    if is_demo_mode():
        profile_to_use = get_demo_profile_name()
        prof_token = set_active_profile(profile_to_use)
        try:
            return await call_next(request)
        finally:
            reset_profile(prof_token)

    # 1. Prefer an authenticated identity: a valid token (Bearer header for the
    #    iOS app, or the httponly cairniq_token cookie for the web UI) binds the
    #    profile from its claim and ignores the unauthenticated profile cookie.
    jwt_token = extract_bearer(
        request.headers.get("authorization"), request.cookies.get("cairniq_token")
    )
    claims = verify_token(jwt_token) if jwt_token else None

    if claims is not None:
        profile_to_use = normalize_profile_name(claims.get("profile"))
        request.state.user = claims
    else:
        request.state.user = None
        # 2. Enforcement is opt-in (CAIRNIQ_AUTH_REQUIRED). When on, anything
        #    outside the public allow-list needs a valid token.
        if (
            auth_required()
            and request.method.upper() != "OPTIONS"
            and not _is_public_path(request.url.path)
        ):
            # Browser page requests get a redirect to /login so the user sees
            # the login form instead of a raw JSON 401. API/fetch calls still
            # get the JSON error so client-side JS can handle it.
            accept = request.headers.get("accept", "")
            is_browser_page = (
                "text/html" in accept
                and not request.url.path.startswith("/api/")
            )
            if is_browser_page:
                from starlette.responses import RedirectResponse
                next_url = quote(str(request.url.path), safe="/")
                return RedirectResponse(f"/login?next={next_url}", status_code=302)
            return JSONResponse({"error": "Authentication required."}, status_code=401)
        # 3. Legacy single-user behaviour: cookie > ACTIVE_PROFILE env > default.
        cookie_profile = request.cookies.get("profile")
        env_profile = os.environ.get("ACTIVE_PROFILE")
        profile_to_use = _resolve_request_profile(cookie_profile, env_profile)

    prof_token = set_active_profile(profile_to_use)
    try:
        return await call_next(request)
    finally:
        reset_profile(prof_token)

# Version surfaced by the public health probe. Never hardcode it here — the one
# source is agent/version.py, which is test-pinned to the newest release tag.
APP_VERSION = __version__


# When this process began serving. Compared against the newest source on disk to
# detect a half-deploy — see `_code_staleness`.
_PROCESS_STARTED = time.time()

# server.py sits at the project root.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Paths whose mtime means "the deployed code changed". The git ref is the precise
# signal (`receive.denyCurrentBranch=updateInstead` moves it on every LAN push),
# and the template and source dirs are checked too so a hand-edited file on the
# box is caught as well.
_DEPLOY_WATCH_PATHS = ("templates", "server.py", "api", "agent", "tools")


def _code_staleness() -> dict[str, Any]:
    """Is this process serving code older than what is on disk?

    THE HALF-DEPLOY DETECTOR (7.1). A LAN push under
    `receive.denyCurrentBranch=updateInstead` rewrites the working tree and does
    NOT restart the service. Jinja re-reads templates per request while Python is
    frozen at exec, so the new template renders against the old context builder —
    which is how /context served `UndefinedError: 'risk_constraints' is undefined`
    for roughly five hours while every liveness probe reported ok.

    Cheap on purpose: this endpoint is hit every ~2 minutes by the watchdog and by
    the iOS client, so it stats a fixed shortlist of directories one level deep
    rather than walking the tree.

    `code_stale` is None, never False, when the answer is unknown (nothing
    stattable). An instrument that cannot see must say so — reporting False here
    would be this codebase's signature failure, a clean reading from a probe that
    never looked.
    """
    newest = 0.0
    newest_path = ""
    for rel in _DEPLOY_WATCH_PATHS:
        target = os.path.join(BASE_DIR, rel)
        try:
            candidates = [target]
            if os.path.isdir(target):
                with os.scandir(target) as entries:
                    candidates += [e.path for e in entries if e.name.endswith((".py", ".html"))]
            for path in candidates:
                mtime = os.path.getmtime(path)
                if mtime > newest:
                    newest, newest_path = mtime, path
        except OSError:
            continue

    if not newest:
        return {"code_stale": None, "code_stale_detail": "no source path could be stat'd"}

    # A small grace: the process writes .pyc files and touches paths as it boots,
    # so a file a few seconds newer than start is normal rather than a deploy.
    stale = newest > (_PROCESS_STARTED + 30)
    detail = (
        f"newest source {os.path.relpath(newest_path, BASE_DIR)} at "
        f"{datetime.fromtimestamp(newest).isoformat(timespec='seconds')}, "
        f"process started {datetime.fromtimestamp(_PROCESS_STARTED).isoformat(timespec='seconds')}"
    )
    return {"code_stale": stale, "code_stale_detail": detail}


@app.get("/api/health")
async def health_check():
    """Public liveness probe.

    The iOS client pings this to validate its (runtime-editable) server URL
    before showing the login screen, and reads ``auth_required`` to decide
    whether a login is needed. Intentionally unauthenticated and cheap — it
    must never touch the agent, brokers, or the LLM.

    It answers 200 for any reachable process, including one running stale code:
    `code_stale` is a field and not a status, because the surface genuinely IS up
    and a non-2xx here would make the watchdog treat a working server as down.
    The watchdog reports that field; restarting on it is the operator's call.
    """
    return {
        "status": "ok",
        "app": "CairnIQ",
        "version": APP_VERSION,
        "auth_required": auth_required(),
        "uptime_s": round(time.time() - _PROCESS_STARTED, 1),
        **_code_staleness(),
    }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    # HTTP middleware doesn't run for WebSockets, so enforce the token here when
    # auth is required. Browsers can't set headers on a WS handshake, so a
    # ?token= query param is also accepted alongside the cookie.
    if auth_required():
        ws_token = extract_bearer(
            websocket.headers.get("authorization"),
            websocket.cookies.get("cairniq_token"),
        ) or websocket.query_params.get("token")
        if verify_token(ws_token or "") is None:
            await websocket.close(code=1008)
            return
    manager = get_connection_manager()
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        log_to_component("server", "WebSocket", f"WebSocket error: {e}", level=logging.WARNING)
        manager.disconnect(websocket)

class LogMessage(BaseModel):
    level: str
    phase: str
    message: str
    data: dict[str, Any] | None = None

@app.post("/api/logs/frontend")
async def log_from_frontend(payload_obj: LogMessage, request: Request):
    log_context_token = None
    try:
        payload = payload_obj.model_dump()
        data = payload.get("data") or {}

        log_context_token = bind_log_context(
            user_ip=request.client.host if request.client else "unknown",
            user_agent=request.headers.get("user-agent", "unknown")
        )
        level_map = {
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL
        }
        level = level_map.get(payload.get("level", "INFO"), logging.INFO)

        log_to_component(
            component="frontend",
            phase=payload.get("phase", "UI"),
            message=payload.get("message", "Browser event"),
            data=data,
            level=level
        )
        return {"status": "ok"}
    except Exception as e:
        log_to_component("server", "Error", f"Error logging frontend event: {e}", level=logging.ERROR)
        return {"status": "error", "message": "Failed to log frontend event"}
    finally:
        if log_context_token is not None:
            reset_log_context(log_context_token)

# Custom StaticFiles to disable caching during development
class NoCacheStaticFiles(StaticFiles):
    def is_not_modified(self, response_headers, request_headers) -> bool:
        return False

    async def get_response(self, path: str, scope):
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response

# Mount static files (disable caching in dev mode for hot-reload)
_DEV_MODE = os.environ.get("LOG_LEVEL", "INFO").upper() == "DEBUG" or os.environ.get("DEV_MODE", "").lower() in ("1", "true")
_StaticFilesClass = NoCacheStaticFiles if _DEV_MODE else StaticFiles
app.mount("/static", _StaticFilesClass(directory="static"), name="static")

# Include Routers
app.include_router(auth.router)
app.include_router(pages.router)
app.include_router(chat.router)
app.include_router(memory.router)
app.include_router(portfolio.router)
app.include_router(dashboard.router)
app.include_router(settings.router)
app.include_router(news.router)
app.include_router(alerts.router)
app.include_router(journal.router)
app.include_router(feedback.router)
app.include_router(reports.router)


def network_exposure_warning(host: str, auth_on: bool) -> str | None:
    """Warn when binding to a non-loopback interface without auth enabled.

    Binding CAIRNIQ_HOST to 0.0.0.0 (or a specific LAN/VPN address) makes the
    app reachable by other devices — the intended setup for the iOS client over
    the UniFi VPN. But without CAIRNIQ_AUTH_REQUIRED=1 that exposes every
    profile's data with no login. Returns None for the safe local-only case.
    """
    loopback = {"127.0.0.1", "localhost", "::1", "[::1]"}
    if (host or "").strip() in loopback:
        return None
    if auth_on:
        return None
    exposed = host or "0.0.0.0"  # noqa: S104  # nosec B104 — message text, not an actual bind
    return (
        f"CAIRNIQ_HOST={exposed} exposes the server on your network, but "
        "CAIRNIQ_AUTH_REQUIRED is off — anyone who can reach this host can read every "
        "profile's data without logging in. Enable auth (CAIRNIQ_AUTH_REQUIRED=1 plus "
        "accounts via scripts/cairniq_user.py) or bind CAIRNIQ_HOST=127.0.0.1 for local-only."
    )


if __name__ == "__main__":
    import uvicorn

    class FilterFrontendLogs(logging.Filter):
        def filter(self, record):
            return "/api/logs/frontend" not in record.getMessage()

    logging.getLogger("uvicorn.access").addFilter(FilterFrontendLogs())

    # Hot-reload is OFF by default. CairnIQ is a personal, local-first app: a
    # normal run wants a single process, not uvicorn's reload supervisor, which
    #   (a) spawns a second worker that re-hydrates secrets from the keychain
    #       (the duplicate "keyring bulk load" at startup), and
    #   (b) re-execs server.py via multiprocessing spawn — which breaks if the
    #       interpreter changes underneath a running process (e.g. a Homebrew
    #       Python point upgrade -> "No module named 'pkgutil'").
    # Developers can opt back in with CAIRNIQ_RELOAD=1.
    dev_reload = os.environ.get("CAIRNIQ_RELOAD", "0") == "1"
    bind_host = os.environ.get("CAIRNIQ_HOST", "127.0.0.1")
    _exposure = network_exposure_warning(bind_host, auth_required())
    if _exposure:
        print(f"\n⚠️  SECURITY: {_exposure}\n")
        logging.getLogger("uvicorn.error").warning(_exposure)
    run_kwargs: dict[str, Any] = {
        "host": bind_host,
        "port": int(os.environ.get("PORT", "8000")),
    }
    if dev_reload:
        run_kwargs["reload"] = True
        # Watch only application source so test edits / .venv / data / logs churn
        # don't bounce the server. Templates and static are served fresh per
        # request (browser refresh — no Python reload needed).
        run_kwargs["reload_dirs"] = ["agent", "api", "tools", "lib"]

    # Single-instance guard. If another CairnIQ is already serving this port,
    # exit cleanly (status 0) instead of crash-looping. The launchd plist uses
    # KeepAlive={SuccessfulExit: false} + ThrottleInterval, so a clean exit here
    # keeps a duplicate DOWN rather than respawning it every few seconds — which
    # would otherwise re-run the heavy startup (ticker downloads, agent init,
    # any LLM warmup) on every cycle. A genuine crash still exits non-zero and
    # is restarted as intended.
    import socket as _socket
    import sys as _sys

    # "0.0.0.0" here is only a comparison (to probe via loopback), not a bind.
    _probe_host = "127.0.0.1" if bind_host in ("0.0.0.0", "") else bind_host  # noqa: S104  # nosec B104
    with _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM) as _probe:
        _probe.settimeout(1.0)
        if _probe.connect_ex((_probe_host, run_kwargs["port"])) == 0:
            _msg = (
                f"Another CairnIQ instance is already listening on "
                f"{_probe_host}:{run_kwargs['port']}; exiting cleanly (single-instance guard)."
            )
            print(f"\n⚠️  {_msg}\n")
            logging.getLogger("uvicorn.error").warning(_msg)
            _sys.exit(0)

    uvicorn.run("server:app", **run_kwargs)
