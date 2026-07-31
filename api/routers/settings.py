import logging
import os

import portalocker
import requests
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from agent.logger import log_to_component
from tools.user_profile import (
    get_active_profile,
    get_demo_profile_name,
    is_demo_mode,
    list_available_profiles,
)

router = APIRouter()

class SaveSettingsRequest(BaseModel):
    settings: dict[str, str]

class ProfileSwitchRequest(BaseModel):
    profile: str


def _detect_alpaca_mode(api_key: str, secret_key: str) -> str | None:
    """Probe Alpaca paper and live endpoints to determine which environment
    these keys belong to. Returns "true" for paper, "false" for live, or
    None if neither responds with 200 (invalid keys or network failure)."""
    if not api_key or not secret_key:
        return None
    headers = {"APCA-API-KEY-ID": api_key, "APCA-API-SECRET-KEY": secret_key}
    endpoints = [
        ("true", "https://paper-api.alpaca.markets/v2/account"),
        ("false", "https://api.alpaca.markets/v2/account"),
    ]
    for mode, url in endpoints:
        try:
            r = requests.get(url, headers=headers, timeout=4)
            if r.status_code == 200:
                return mode
        except requests.RequestException:
            continue
    return None


@router.post("/api/settings/save")
async def save_settings(request: SaveSettingsRequest):
    """Save terminal settings / API keys to .env and environment."""
    if is_demo_mode():
        return JSONResponse(
            {"error": "Settings are locked in demo mode so your real configuration stays unchanged."},
            status_code=403,
        )

    from dotenv import set_key
    env_path = os.path.join(os.getcwd(), "user_data", ".env")

    from tools.broker_credentials import (
        get_broker_secret,
        is_broker_secret,
        is_broker_setting,
        set_broker_secret,
        set_broker_setting,
    )
    from tools.secrets_store import get_secret, is_secret_key, set_secret

    # Snapshot the config that pins the active LLM/embed backend so we can hot-swap
    # provider-dependent globals (DSPy's LM, the Tool-RAG embedding index) after the
    # save if it changed — the chat path already re-reads os.environ per call, but
    # those two are initialized once and would otherwise keep the old provider until
    # a restart.
    _provider_snapshot = {
        k: os.environ.get(k)
        for k in ("LLM_PROVIDER", "AIDLC_MODEL_ID", "AIDLC_SONNET_MODEL_ID", "AIDLC_EMBED_MODEL_ID")
    }

    try:
        lock_path = env_path + ".lock"
        with portalocker.Lock(lock_path, timeout=10):
            for key, value in request.settings.items():
                # Always trim settings before persisting them. Stray leading/
                # trailing whitespace (e.g. a pasted " gpt-5.4-mini" or a key with
                # a trailing newline) otherwise reaches Azure/providers verbatim
                # and surfaces as DeploymentNotFound / auth failures that are
                # painful to diagnose. No legitimate setting needs surrounding
                # whitespace, so trim unconditionally at the write boundary.
                if isinstance(key, str):
                    key = key.strip()
                if isinstance(value, str):
                    value = value.strip()

                # Skip masked values (if user didn't change them)
                if "*" in value and key != "AWS_REGION":
                    continue

                if is_broker_secret(key):
                    # Broker secrets are scoped to the ACTIVE profile (the
                    # default profile keeps the legacy global keychain entry; a
                    # named profile gets its own isolated one). Don't wipe a
                    # stored key when the field comes through blank.
                    if not value and get_broker_secret(key):
                        continue
                    set_broker_secret(key, value)
                elif is_broker_setting(key):
                    # Non-secret broker config (paper-mode, enabled, owner),
                    # also per-profile.
                    set_broker_setting(key, value)
                elif is_secret_key(key):
                    # Defensive: if the field is empty AND we already have a
                    # value stashed for this key, treat as a no-op rather than
                    # deleting it. Otherwise an empty Settings form (e.g. when
                    # the server briefly fails to hydrate os.environ from the
                    # keychain on startup) would silently nuke the user's keys
                    # the next time they save an unrelated setting. Genuine
                    # deletion can still be done via the OS keychain UI.
                    if not value and get_secret(key):
                        continue

                    # Route secrets to the OS keychain. set_secret() also mirrors
                    # the value into os.environ so the running process sees it.
                    # If the keychain backend is unavailable (rare on Mac/Win,
                    # common on headless Linux), fall back to writing .env so
                    # the user can still configure the app.
                    if not set_secret(key, value):
                        set_key(env_path, key, value)
                        os.environ[key] = value
                else:
                    set_key(env_path, key, value)
                    os.environ[key] = value

            # Auto-detect Alpaca paper vs live mode when either key was just set.
            alpaca_key_changed = (
                "ALPACA_API_KEY" in request.settings and "*" not in request.settings["ALPACA_API_KEY"]
            ) or (
                "ALPACA_SECRET_KEY" in request.settings and "*" not in request.settings["ALPACA_SECRET_KEY"]
            )
            if alpaca_key_changed:
                detected = _detect_alpaca_mode(
                    get_broker_secret("ALPACA_API_KEY"),
                    get_broker_secret("ALPACA_SECRET_KEY"),
                )
                if detected is not None:
                    set_broker_setting("ALPACA_PAPER_MODE", detected)

        scheduler_updates = {}
        for key, value in list(request.settings.items()):
            if isinstance(key, str) and key.startswith("SCHEDULER_JOB_"):
                job_name = key[len("SCHEDULER_JOB_"):]
                val_bool = str(value).strip().lower() in ("true", "1", "yes", "on")
                scheduler_updates[job_name] = val_bool
            elif isinstance(key, str) and key.startswith("SCHEDULER_COOLDOWN_"):
                job_name = key[len("SCHEDULER_COOLDOWN_"):]
                try:
                    scheduler_updates[f"{job_name}_cooldown_seconds"] = float(value)
                except (ValueError, TypeError):
                    pass

        if scheduler_updates:
            from tools.scheduler import update_scheduler_settings
            update_scheduler_settings(scheduler_updates)

        if request.settings.get("BASE_CURRENCY"):
            from tools.memory import normalize_base_currency, update_profile
            update_profile({"base_currency": normalize_base_currency(request.settings["BASE_CURRENCY"])})

        # Hot-swap provider-dependent globals so an LLM_PROVIDER / model change takes
        # effect at runtime (no restart). os.environ was already updated in the loop
        # above, so get_llm() picks up the new provider on its own; DSPy and the
        # Tool-RAG index are the set-once exceptions. Best-effort: a swap failure must
        # never fail the save (the values are persisted regardless).
        now = {k: os.environ.get(k) for k in _provider_snapshot}
        llm_changed = any(
            now[k] != _provider_snapshot[k]
            for k in ("LLM_PROVIDER", "AIDLC_MODEL_ID", "AIDLC_SONNET_MODEL_ID")
        )
        embed_changed = now["LLM_PROVIDER"] != _provider_snapshot["LLM_PROVIDER"] or \
            now["AIDLC_EMBED_MODEL_ID"] != _provider_snapshot["AIDLC_EMBED_MODEL_ID"]
        if llm_changed:
            try:
                from agent.dspy_setup import reconfigure_dspy
                reconfigure_dspy(error_callback=lambda m: log_to_component("server", "Settings", m, level=logging.WARNING))
            except Exception as e:
                log_to_component("server", "Settings", f"DSPy hot-swap skipped: {e}", level=logging.WARNING)
        if embed_changed:
            try:
                from agent.tool_retriever import ToolRetriever
                ToolRetriever.reset()
            except Exception as e:
                log_to_component("server", "Settings", f"Tool-RAG reset skipped: {e}", level=logging.WARNING)

        return JSONResponse({"success": True})
    except Exception as e:
        log_to_component("server", "Settings", f"Error updating settings: {e}", level=logging.ERROR)
        return JSONResponse({"error": "Failed to update settings"}, status_code=500)

@router.get("/api/session/cost")
async def get_session_cost_api(thread_id: str | None = None):
    # Each field is resolved defensively and type-coerced so a single bad/mocked
    # dependency can never make the whole endpoint fail (and zero out the cost).
    _SUPPORTED = {"USD", "CAD", "EUR", "GBP", "AUD", "JPY"}

    cost_cad = 0.0
    tokens = 0
    try:
        if thread_id:
            from api.routers.chat import _thread_costs, _thread_tokens
            v = _thread_costs.get(thread_id, 0.0)
            cost_cad = float(v) if isinstance(v, (int, float)) else 0.0
            t = _thread_tokens.get(thread_id, 0)
            tokens = int(t) if isinstance(t, (int, float)) else 0
    except Exception:
        cost_cad, tokens = 0.0, 0

    any_unpriced = False
    breakdown: dict = {}
    try:
        from agent.cost_tracker import get_session_breakdown, get_session_stats
        s = get_session_stats()
        any_unpriced = bool(s.get("any_unpriced", False)) if isinstance(s, dict) else False
        b = get_session_breakdown()
        if isinstance(b, dict):
            # Coerce every value to a JSON-safe number so a stray/mocked value
            # can't break serialization (and thus the whole endpoint).
            for slot, vals in b.items():
                if isinstance(vals, dict):
                    breakdown[str(slot)] = {
                        str(k): (v if isinstance(v, (int, float, bool)) else 0)
                        for k, v in vals.items()
                    }
    except Exception:
        any_unpriced, breakdown = False, {}

    # Display the cost in the user's BASE currency so it's consistent with the rest
    # of the app. Cost is accumulated in CAD; convert only when the base currency is
    # a recognized non-CAD code and a numeric rate is available (else keep CAD).
    currency = "CAD"
    cost = cost_cad
    try:
        from tools.memory import get_profile_base_currency
        resolved = get_profile_base_currency()
        if isinstance(resolved, str) and resolved.strip().upper() in _SUPPORTED:
            resolved = resolved.strip().upper()
            if resolved != "CAD" and cost_cad:
                from tools.portfolio_csv import get_exchange_rate
                rate = get_exchange_rate("CAD", resolved)
                if isinstance(rate, (int, float)) and rate > 0:
                    cost, currency = cost_cad * rate, resolved
            else:
                currency = resolved
    except Exception:
        currency, cost = "CAD", cost_cad

    # Tokens are the source of truth — always accurate. `cost` is an estimate in the
    # base currency that only counts priced slots; cost_cad kept for back-compat.
    return JSONResponse({
        "cost_cad": cost_cad,
        "cost": cost,
        "currency": currency,
        "tokens": tokens,
        "any_unpriced": any_unpriced,
        "breakdown": breakdown,
    })

def _build_profile_switch_response(profile_name: str) -> JSONResponse:
    # Defense in depth: regardless of how this helper is called, the cookie value
    # must come from a fixed allow-list. We re-derive ``cookie_value`` from the
    # allow-list itself (not from the input string) so CodeQL can see that the
    # value reaching ``set_cookie`` originates only inside the allow-list.
    allowed: set[str] = {p["name"] for p in list_available_profiles()}
    allowed.add("default")
    cookie_value = "default"
    for candidate in allowed:
        if candidate == profile_name:
            cookie_value = candidate
            break
    response = JSONResponse({"status": "switched", "active_profile": cookie_value})
    response.set_cookie(
        key="profile",
        value=cookie_value,
        httponly=True,
        samesite="lax",
        max_age=86400 * 30,
    )
    return response

def _build_clear_profile_response(active_profile: str) -> JSONResponse:
    response = JSONResponse({"status": "cleared", "active_profile": active_profile})
    response.delete_cookie("profile")
    return response

@router.get("/api/profiles")
async def list_profiles():
    """
    List all available profiles and which one is currently active.
    Returns: {profiles: [...], active: "name"}
    Frontend / login page calls this to populate a profile selector.
    """
    profiles = list_available_profiles()
    active = get_active_profile()
    # Re-stamp 'active' flag with the live per-request value
    for p in profiles:
        p["active"] = (p["name"] == active)
    return JSONResponse({"profiles": profiles, "active": active})

@router.post("/api/profile/switch")
async def switch_profile(req: ProfileSwitchRequest):
    """
    Switch the active profile for subsequent requests.
    Sets a 'profile' cookie that the ProfileMiddleware picks up on every
    future request — no server restart required.

    A login page should call this endpoint after authenticating the user.
    Body: {"profile": "profile_name"}
    """
    if is_demo_mode():
        return JSONResponse(
            {
                "error": "Profile switching is disabled in demo mode.",
                "active_profile": get_demo_profile_name(),
            },
            status_code=403,
        )

    # Sanitise input — no path traversal
    name = req.profile.strip().replace("/", "").replace("..", "") or "default"

    # Validate: must be an existing profile
    available_names = [p["name"] for p in list_available_profiles()]
    if name not in available_names:
        return JSONResponse(
            {"error": f"Profile '{name}' not found. Available: {available_names}"},
            status_code=404
        )

    return _build_profile_switch_response(name)

@router.delete("/api/profile/switch")
async def clear_profile_cookie():
    """Clear the profile cookie, reverting to ACTIVE_PROFILE env var or 'default'."""
    if is_demo_mode():
        return _build_clear_profile_response(get_demo_profile_name())
    return _build_clear_profile_response(os.environ.get("ACTIVE_PROFILE", "default"))
