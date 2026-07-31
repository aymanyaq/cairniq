"""
Session usage tracker (provider-agnostic).

Tracks token usage per *slot* (primary / fast / embed) — always accurate — and
turns it into a cost ONLY when a price is configured for that slot. There is no
per-model price table to maintain: the model behind each slot changes freely
(Claude, DeepSeek, grok, gpt-5.x, Kimi, …), so we price the slot, not the model.

Tokens are the source of truth. Cost is an optional overlay: set a price per slot
and you get a $ estimate; leave it unset and that slot is reported as "unpriced"
(tokens only) instead of being silently mispriced.

Configure per-slot prices via env (USD per 1M tokens, "input/output[/cache]"):
    AIDLC_PRICE_PRIMARY="5/25"        # the model you put in the primary slot
    AIDLC_PRICE_FAST="0.15/0.60"      # the fast slot (formerly "Sonnet")
    AIDLC_PRICE_EMBED="0.02/0"
Slot membership is derived from AIDLC_MODEL_ID / AIDLC_SONNET_MODEL_ID /
AIDLC_EMBED_MODEL_ID (and their _<PROVIDER> variants).

Totals reset when the application restarts (no persistence).
"""

import os
import threading

# Fallback FX if the live USD->CAD rate hasn't been injected into os.environ yet.
USD_TO_CAD = 1.44

SLOTS = ("primary", "fast", "embed", "other")


def _new_bucket() -> dict:
    return {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read_tokens": 0,
        "cost_usd": 0.0,
        "cost_cad": 0.0,
        "priced": False,   # True once any priced usage lands in this slot
    }


def _new_grounding_bucket() -> dict:
    return {"requests": 0, "queries": 0, "cost_usd": 0.0, "cost_cad": 0.0, "priced": False}


_lock = threading.Lock()
_session: dict[str, dict] = {s: _new_bucket() for s in SLOTS}

# Google Search grounding is billed per REQUEST, not in tokens, so it cannot live
# in a token slot. The tokens of a grounded response are already counted through
# the ordinary usage path; this is the separate search charge that path cannot see.
_grounding: dict = _new_grounding_bucket()

# Guards the DSPy history cursor only. Deliberately NOT _lock: track_dspy_calls
# calls accumulate_cost, which takes _lock, and threading.Lock is not reentrant.
_dspy_cursor_lock = threading.Lock()


def _norm(s: str | None) -> str:
    return (s or "").strip().lower()


def _cfg_model(*names: str) -> str:
    """Resolve a configured slot model id (active-provider scoped, then unscoped)."""
    prov = (os.environ.get("LLM_PROVIDER") or "").strip().upper()
    for n in names:
        scoped = os.environ.get(f"{n}_{prov}") if prov else None
        val = scoped or os.environ.get(n)
        if val and val.strip():
            return _norm(val)
    return ""


def _slot_for_model(model_id: str) -> str:
    """Map a model id / deployment name to its slot via the configured slot models."""
    m = _norm(model_id)
    if not m:
        return "other"
    # Embeddings: the sentinel "titan-embed" and names like "text-embedding-3-small"
    # both contain "embed"; also honor an explicitly configured embed model.
    embed = _cfg_model("AIDLC_EMBED_MODEL_ID")
    if "embed" in m or (embed and (embed in m or m in embed)):
        return "embed"
    fast = _cfg_model("AIDLC_SONNET_MODEL_ID", "AIDLC_FAST_MODEL_ID")
    if fast and (fast in m or m in fast):
        return "fast"
    primary = _cfg_model("AIDLC_MODEL_ID")
    if primary and (primary in m or m in primary):
        return "primary"
    return "other"


def _price_for_slot(slot: str):
    """Return (input, output, cache) USD-per-1M for a slot, or None if unpriced.

    Configured via AIDLC_PRICE_<SLOT>="input/output[/cache]". Cache defaults to
    the input rate when omitted.
    """
    raw = (os.environ.get(f"AIDLC_PRICE_{slot.upper()}") or "").strip()
    if not raw:
        return None
    parts = [p.strip() for p in raw.replace(",", "/").split("/") if p.strip() != ""]
    try:
        inp = float(parts[0])
        out = float(parts[1]) if len(parts) > 1 else inp
        cache = float(parts[2]) if len(parts) > 2 else inp
        return (inp, out, cache)
    except (ValueError, IndexError):
        return None


def accumulate_cost(input_tokens: int, output_tokens: int, model_id: str = "",
                    cache_read_tokens: int = 0) -> float:
    """Add token usage to the running session total, bucketed by slot.

    Cost is computed only if the slot has a configured price; otherwise tokens are
    still tracked and the slot stays "unpriced". Returns total session cost in CAD.
    """
    slot = _slot_for_model(model_id)
    price = _price_for_slot(slot)

    cost_usd = 0.0
    if price is not None:
        inp_rate, out_rate, cache_rate = price
        non_cached = max(0, input_tokens - cache_read_tokens)
        cost_usd = (
            (non_cached / 1_000_000) * inp_rate
            + (cache_read_tokens / 1_000_000) * cache_rate
            + (output_tokens / 1_000_000) * out_rate
        )

    fx = float(os.environ.get("USD_TO_CAD", str(USD_TO_CAD)))
    cost_cad = cost_usd * fx

    # Persistent, restart-safe budget meter (circuit breaker). Embeddings can be
    # high-volume during RAG, so they add to spend but NOT to the calls/hour
    # counter, which targets expensive generations / runaway loops.
    try:
        from agent import llm_budget
        llm_budget.record(cost_cad=cost_cad, calls=0 if slot == "embed" else 1)
    except Exception:
        pass

    with _lock:
        b = _session[slot]
        b["input_tokens"] += input_tokens
        b["output_tokens"] += output_tokens
        b["cache_read_tokens"] += cache_read_tokens
        b["cost_usd"] += cost_usd
        b["cost_cad"] += cost_cad
        if price is not None:
            b["priced"] = True
        return _aggregate_locked()["cost_cad"]


def accumulate_grounding(requests: int = 1, queries: int = 0) -> None:
    """Record Gemini/Vertex Google-Search grounding, which is billed per request.

    Grounding runs server-side: the model executes the search itself rather than
    emitting a tool_call we dispatch, so nothing in the token path can see the
    search charge. The *tokens* of a grounded response are already counted by
    accumulate_cost through the ordinary usage_metadata route — double counting
    them here would be wrong, so this tracks only the separate per-request fee.

    Priced by AIDLC_PRICE_GROUNDING, USD per 1,000 grounded requests. Unset means
    requests are still counted (and visible in the breakdown) but add no cost,
    matching how an unpriced token slot behaves.
    """
    if requests <= 0:
        return

    raw = (os.environ.get("AIDLC_PRICE_GROUNDING") or "").strip()
    cost_usd = 0.0
    priced = False
    if raw:
        try:
            cost_usd = (requests / 1000.0) * float(raw)
            priced = True
        except ValueError:
            pass

    fx = float(os.environ.get("USD_TO_CAD", str(USD_TO_CAD)))
    cost_cad = cost_usd * fx

    if cost_cad:
        try:
            from agent import llm_budget
            llm_budget.record(cost_cad=cost_cad, calls=0)
        except Exception:
            pass

    with _lock:
        _grounding["requests"] += requests
        _grounding["queries"] += max(0, queries)
        _grounding["cost_usd"] += cost_usd
        _grounding["cost_cad"] += cost_cad
        if priced:
            _grounding["priced"] = True


def _aggregate_locked() -> dict:
    agg = {"input_tokens": 0, "output_tokens": 0, "cache_read_tokens": 0,
           "cost_usd": 0.0, "cost_cad": 0.0}
    for b in _session.values():
        for k in agg:
            agg[k] += b[k]
    # Grounding contributes cost but no tokens.
    agg["cost_usd"] += _grounding["cost_usd"]
    agg["cost_cad"] += _grounding["cost_cad"]
    return agg


def get_session_cost() -> float:
    """Total session cost in CAD (priced slots only)."""
    with _lock:
        return _aggregate_locked()["cost_cad"]


def get_session_tokens() -> int:
    """Total input+output tokens this session (always accurate)."""
    with _lock:
        agg = _aggregate_locked()
        return agg["input_tokens"] + agg["output_tokens"]


def get_session_stats() -> dict:
    """Aggregate session stats (tokens + cost) with an `any_unpriced` flag."""
    with _lock:
        agg = _aggregate_locked()
        agg["any_unpriced"] = any(
            (b["input_tokens"] or b["output_tokens"]) and not b["priced"]
            for b in _session.values()
        ) or bool(_grounding["requests"] and not _grounding["priced"])
        agg["grounding_requests"] = _grounding["requests"]
        agg["grounding_queries"] = _grounding["queries"]
        return agg


def get_session_breakdown() -> dict:
    """Per-slot tokens + cost, only for slots that saw usage.

    `grounding` appears only once a grounded request has actually happened, and
    carries requests/queries rather than tokens.
    """
    with _lock:
        out = {
            slot: dict(b)
            for slot, b in _session.items()
            if b["input_tokens"] or b["output_tokens"]
        }
        if _grounding["requests"]:
            out["grounding"] = dict(_grounding)
        return out


def reset_session_cost():
    """Reset all session totals to zero."""
    with _lock:
        for s in SLOTS:
            _session[s] = _new_bucket()
        _grounding.clear()
        _grounding.update(_new_grounding_bucket())


# --- Helpers for tracking non-safe_invoke calls ---

def track_dspy_calls(lm, model_id: str = "") -> int:
    """Flush new entries from a DSPy LM's history into the accumulator.

    Called from the LM subclasses in agent/dspy_setup.py after every completed
    call. DSPy does not route through safe_invoke, so this is the only path by
    which its usage reaches the meter.

    The cursor advance is guarded because scans and nodes issue DSPy calls from
    worker threads against the one shared global LM; without it, two concurrent
    flushes read the same marker and double-count the overlap.

    Note this reads `lm.history`, so it tracks nothing when DSPy is configured
    with `disable_history` (off by default).
    """
    with _dspy_cursor_lock:
        marker = getattr(lm, '_cost_tracker_cursor', 0)
        entries = list(lm.history[marker:])
        lm._cost_tracker_cursor = len(lm.history)

    tracked = 0
    for entry in entries:
        usage = entry.get("usage", {}) or {}
        inp = usage.get("prompt_tokens", 0) or usage.get("input_tokens", 0)
        out = usage.get("completion_tokens", 0) or usage.get("output_tokens", 0)
        if inp or out:
            m = entry.get("model", "") or model_id
            accumulate_cost(inp, out, m)
            tracked += 1
    return tracked


def track_embedding_cost(input_tokens: int):
    """Track embedding usage (input-only). Routes to the embed slot."""
    accumulate_cost(input_tokens, 0, "titan-embed")
