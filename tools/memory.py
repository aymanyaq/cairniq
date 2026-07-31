"""
Conversation Memory System
Provides persistent memory for CairnIQ across sessions.
Stores user profile, key facts, conversation summaries, and past recommendations.
"""
import copy
import difflib
import json
import math
import os
import re
from datetime import datetime, timedelta
from typing import Any

from agent.utils import safe_print
from tools.exception_logger import log_exceptions
from tools.graph_memory import graph_memory
from tools.json_store import write_json_atomic
from tools.user_profile import get_data_path

# Default empty memory structure
DEFAULT_MEMORY = {
    "user_profile": {
        "name": None,
        "age": None,
        "risk_tolerance": None,  # conservative, moderate, aggressive
        "retirement_age": None,
        "annual_income": None,
        "base_currency": None,
        "investment_goals": [],
        # Structured wealth goal (today's terms, base currency). None = unset:
        # no target is assumed, none is quoted back, and required-CAGR /
        # goal-funded probability stay unavailable until set. goal_horizon_years
        # is the GOAL's own horizon, deliberately independent of retirement_age
        # (the goal runway is typically shorter). goal_annual_contribution is the
        # planned yearly inflow — the single biggest driver of whether a goal is
        # reachable, and like the target it is NEVER assumed: a projection run
        # without it is a projection of a portfolio nobody is funding.
        # Read via get_financial_goal().
        "goal_target_low": None,
        "goal_target_high": None,
        "goal_horizon_years": None,
        "goal_annual_contribution": None,
        "accounts": [],
        "last_updated": None
    },
    # Numeric risk limits the user has explicitly set (max_position_pct,
    # max_fund_position_pct, max_sector_pct, max_risk_per_trade_pct,
    # restricted_symbols). EMPTY IS MEANINGFUL and is the correct default: a
    # user who has stated no limit has accepted unbounded risk, so nothing is
    # enforced and nothing may be cited back to them as "your limit". Read via
    # tools.ips_precheck.load_ips_constraints — never default a cap here.
    "risk_constraints": {},
    "key_facts": [],
    "conversation_summaries": [],
    "past_recommendations": [],
    "active_theses": [],
    "secular_themes": [
        {
            "theme": "AI / Semiconductors / Compute Infrastructure",
            "conviction": "high",
            "horizon": "5-10 years",
            "rationale": (
                "Multi-year capex super-cycle driven by foundation-model training, inference at scale, "
                "and AI-native enterprise software. Treat as a structural position, not a tactical trade."
            ),
            "trim_triggers": [
                "Weekly trend break (close below 40-week MA) on the position itself",
                "MAG7 / hyperscaler capex guide-down across two consecutive quarters",
                "Market regime shifts to Defensive Preservation (VIX > 28 or sustained sub-200-day)",
                "Position single-name concentration exceeds the user's stated single-name cap"
            ],
            "do_not_trim_for": [
                "Generic mean-reversion / RSI > 70 alone",
                "Rotation calls into lagging sectors without confirmed money-flow into those sectors",
                "Devil's-advocate framing without new contradicting evidence",
                "Short-term overbought signals during a confirmed bullish regime"
            ]
        }
    ]
}

SUPPORTED_BASE_CURRENCIES = {"USD", "CAD", "EUR", "GBP", "AUD", "JPY"}


def normalize_base_currency(value: Any, default: str = "USD") -> str:
    """Normalize a user-facing/base currency code."""
    code = str(value or "").strip().upper()
    if code in SUPPORTED_BASE_CURRENCIES:
        return code
    return default


# Locale → default currency, for a profile that has never stated one. Lives here
# rather than in the API layer because BOTH the page and the memory store must
# reach the same answer; see configured_base_currency.
LOCALE_DEFAULT_CURRENCY = {
    "English (Canada)": "CAD",
    "French (Canada)": "CAD",
    "English (United States)": "USD",
    "English (United Kingdom)": "GBP",
    "German (Frankfurt)": "EUR",
    "Japanese (Tokyo)": "JPY",
}


def _persisted_setting(key: str) -> str | None:
    """Read a non-secret setting from the persisted user_data/.env FIRST.

    os.environ is per-process: a Settings save mutates only the handling worker's
    env, so another worker keeps a stale snapshot. Reading the shared file is what
    makes the answer authoritative regardless of who serves the request.
    """
    try:
        from dotenv import dotenv_values
        value = dotenv_values(os.path.join(os.getcwd(), "user_data", ".env")).get(key)
        if value is not None and str(value).strip():
            return value
    except Exception:
        pass
    return os.environ.get(key)


def configured_base_currency() -> str:
    """The deployment's base currency, for a profile that has not stated one.

    THE single fallback, shared with the dashboard/settings layer. It used to be
    duplicated: the page resolved an unset currency through the persisted .env and
    a locale default (CAD for English (Canada)), while this module hardcoded USD.
    A profile with no `base_currency` therefore saw "CAD" on every screen while
    the memory store stamped its records "USD" — and once 4.5's wealth goal
    shipped, that meant a user could type a 10-year target believing it was CAD
    and have the projection score it as USD, a ~40% error on the number the whole
    plan is measured against. Two fallbacks for one unset field is a bug waiting
    for a feature to make it expensive; this is that feature.
    """
    explicit = (_persisted_setting("BASE_CURRENCY")
                or _persisted_setting("CAIRNIQ_BASE_CURRENCY"))
    code = str(explicit or "").strip().upper()
    if code in SUPPORTED_BASE_CURRENCIES:
        return code
    locale = _persisted_setting("REGIONAL_LOCALE") or "English (Canada)"
    return LOCALE_DEFAULT_CURRENCY.get(locale, "CAD")


@log_exceptions()
def get_profile_base_currency(profile: dict[str, Any] | None = None) -> str:
    """Resolve the user's base currency: profile first, then the deployment default."""
    if profile is None:
        try:
            profile = load_memory().get("user_profile", {})
        except Exception:
            profile = {}

    stated = (profile or {}).get("base_currency")
    if stated:
        return normalize_base_currency(stated)
    return configured_base_currency()


def _coerce_int(value: Any) -> int | None:
    """Best-effort integer parsing for profile fields stored as strings."""
    try:
        if value is None or value == "":
            return None
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return None


def _format_profile_income(income: Any, base_currency: str) -> str:
    """Format annual income with the profile's base currency."""
    if income is None or income == "":
        return "Data Unavailable"

    try:
        income_numeric = re.sub(r"[^0-9.]", "", str(income))
        return f"${float(income_numeric):,.0f} {base_currency}"
    except (ValueError, TypeError):
        return f"{income} {base_currency}"


def _join_profile_list(value: Any) -> str:
    if isinstance(value, list):
        cleaned = [str(item).strip() for item in value if str(item).strip()]
        return ", ".join(cleaned) if cleaned else "Data Unavailable"
    if value:
        return str(value)
    return "Data Unavailable"


def _format_profile_fields_block(profile: dict[str, Any]) -> list[str]:
    """Render a compact canonical profile block for LLM prompt binding."""
    profile = profile or {}
    base_currency = get_profile_base_currency(profile)
    age = _coerce_int(profile.get("age"))
    retirement_age = _coerce_int(profile.get("retirement_age"))
    years_to_retirement = (
        retirement_age - age
        if age is not None and retirement_age is not None and retirement_age >= age
        else None
    )
    meaningful_fields = [
        "name",
        "age",
        "risk_tolerance",
        "retirement_age",
        "annual_income",
        "base_currency",
        "investment_goals",
        "accounts",
    ]
    profile_status = "available" if any(profile.get(key) for key in meaningful_fields) else "missing"

    def value_or_unavailable(value: Any) -> str:
        return str(value) if value is not None and value != "" else "Data Unavailable"

    return [
        "PROFILE_FIELDS:",
        f"profile_status: {profile_status}",
        f"name: {value_or_unavailable(profile.get('name'))}",
        f"age: {value_or_unavailable(profile.get('age'))}",
        f"risk_tolerance: {value_or_unavailable(profile.get('risk_tolerance'))}",
        f"retirement_age: {value_or_unavailable(profile.get('retirement_age'))}",
        f"years_to_retirement: {value_or_unavailable(years_to_retirement)}",
        f"annual_income: {_format_profile_income(profile.get('annual_income'), base_currency)}",
        f"base_currency: {base_currency}",
        f"investment_goals: {_join_profile_list(profile.get('investment_goals'))}",
        f"accounts: {_join_profile_list(profile.get('accounts'))}",
        "END_PROFILE_FIELDS",
        "",
    ]


@log_exceptions()
def load_memory() -> dict[str, Any]:
    """Load memory from disk. Returns default memory if file doesn't exist."""
    try:
        memory_file = get_data_path("user_memory.json")
        if os.path.exists(memory_file):
            with open(memory_file) as f:
                memory = json.load(f)
                # Ensure all keys exist (for backwards compatibility)
                for key in DEFAULT_MEMORY:
                    if key not in memory:
                        memory[key] = copy.deepcopy(DEFAULT_MEMORY[key])
                if isinstance(memory.get("user_profile"), dict):
                    for key, value in DEFAULT_MEMORY["user_profile"].items():
                        memory["user_profile"].setdefault(key, copy.deepcopy(value))
                return memory
    except Exception as e:
        safe_print(f"⚠️ Error loading memory: {e}")

    return copy.deepcopy(DEFAULT_MEMORY)


@log_exceptions()
def save_memory(memory: dict[str, Any]) -> bool:
    """Save memory to disk. Returns True on success."""
    try:
        memory_file = get_data_path("user_memory.json")
        write_json_atomic(memory_file, memory)
        return True
    except Exception as e:
        safe_print(f"⚠️ Error saving memory: {e}")
        return False


def _normalize_thesis_symbol(symbol: Any) -> str:
    """Base ticker for held-matching: '$keel.to' -> 'KEEL'.

    Strips the '$' the thesis extractor sometimes stores and the listing suffix, so a
    thesis on 'KEEL' matches a held 'KEEL.TO'. Same base-symbol convention
    add_recommendation already uses to reconcile a sell against a thesis.
    """
    return str(symbol or "").strip().upper().lstrip("$").split(".")[0]


def _held_base_symbols() -> set[str] | None:
    """Base symbols the user actually holds, or None if the portfolio can't be read.

    None means UNKNOWN, never "not held" — the caller must not issue a directive that
    depends on held status when it cannot be verified. Reads the same source of truth
    as the verify_portfolio_holdings tool so the thesis block and the "absent = Not
    Held" hard rule can never disagree. Cheap: get_portfolio_summary is 5-min cached
    and Today's Priority warms it via check_portfolio_allocation.
    """
    try:
        from tools.portfolio_csv import get_portfolio_decision_context

        context = get_portfolio_decision_context()
        if not isinstance(context, dict) or context.get("error"):
            return None
        owned = context.get("owned_symbols")
        if not isinstance(owned, list):
            return None
        return {_normalize_thesis_symbol(s) for s in owned if s}
    except Exception:
        return None


def _thesis_position_state(thesis: dict[str, Any], held: set[str] | None) -> str:
    """'held' | 'watching' | 'unknown' for a thesis, from verified holdings."""
    if held is None:
        return "unknown"
    base = _normalize_thesis_symbol(thesis.get("symbol"))
    if not base:
        return "unknown"
    return "held" if base in held else "watching"


def _enrich_thesis_with_price_context(
    thesis: dict[str, Any],
    held: set[str] | None = None,
) -> dict[str, Any]:
    """Add live price context and health flags to a thesis.

    `held` is the verified holdings set from _held_base_symbols(); pass it in so a
    block of theses shares one portfolio read. Omitting it resolves holdings lazily —
    only once there is a live price and a recorded level to judge, so the no-data
    paths stay network-free.

    Flag vocabulary depends on whether the position is actually HELD. A BUY thesis on
    a name the user does NOT own yet is a monitored *entry plan*, not an open
    position: its price hitting the stop means the setup broke before entry, and its
    price hitting the target means the move happened without them. Neither is an exit.
    Firing exit flags on those (the old behaviour, which had no portfolio awareness at
    all) made Today's Priority Lane 0 recommend selling a name that was never bought,
    then "resolve" the clash with the absent-equals-Not-Held rule by deleting the
    watchlist thesis the user was actively waiting to execute.
    """
    from datetime import datetime

    symbol = thesis.get('symbol', '')
    if not symbol:
        return thesis

    # 1. Parse the decision levels, keeping stop_loss / target_price / entry DISTINCT.
    # The UI/API store stop_loss and target_price as bare numbers (e.g. "600"), so
    # parse those fields directly. Entry prices, by contrast, come ONLY from the
    # free-text conditions where an explicit "$" marks a price (an unprefixed number
    # there could be a date or a percentage). Conflating the stop into the entry bag
    # — the old behaviour — made a thesis whose only figure was a stop_loss report a
    # bogus "ENTRY MISSED, N% above entry zone" using the STOP as the entry anchor.
    def _num(v):
        try:
            return float(str(v).replace('$', '').replace(',', '').strip())
        except (ValueError, TypeError, AttributeError):
            return None

    stop_price = _num(thesis.get('stop_loss'))
    target_price = _num(thesis.get('target_price'))
    entry_prices = [float(m) for m in re.findall(r'\$(\d+(?:\.\d+)?)', thesis.get('conditions', '') or '')]

    # 2. Age is network-free — compute it first and build the staleness flag up front,
    # so a failed/empty live-price lookup below still surfaces "STALE". The old code
    # returned with NO flags at all on a price-fetch miss, silently hiding a
    # months-old thesis — the same silent-no-op class as the stop-loss parse bug above.
    created = thesis.get('created_at', '')
    try:
        age_days = (datetime.now() - datetime.fromisoformat(created)).days if created else None
    except Exception:
        age_days = None
    stale_flags = []
    if age_days and age_days > 30:
        stale_flags.append(f"STALE — Thesis is {age_days} days old with no update")

    # 3. Get current price (fast, cached). On any failure, fall through keeping the
    # age-based flags rather than discarding them.
    data = None
    current_price = None
    try:
        from tools.market_data import get_stock_data
        data = get_stock_data(symbol)
        if isinstance(data, dict):
            current_price_str = data.get('current_price') or data.get('price') or ""
            if current_price_str and current_price_str != "N/A":
                current_price = float(str(current_price_str).replace('$', '').replace(',', ''))
    except Exception:
        data, current_price = None, None

    # Nothing to compare against — no live price, or no recorded levels at all. Still
    # surface the age-based staleness flag rather than returning bare.
    if not current_price or (not entry_prices and stop_price is None and target_price is None):
        thesis['_age_days'] = age_days
        thesis['_health_flags'] = stale_flags
        return thesis

    # Held status decides the whole flag vocabulary below, so resolve it only now —
    # past the early returns, where there is actually a decision to make.
    if held is None:
        held = _held_base_symbols()
    state = _thesis_position_state(thesis, held)
    thesis['_position_state'] = state

    action = thesis.get('action', 'BUY').upper()
    is_long = action in ('BUY', 'ADD', 'HOLD')

    # Held status could not be verified — emit price/age context but NO directive that
    # depends on it. Never tell the user to exit a position we cannot confirm they own.
    if state == "unknown":
        thesis['_live_price'] = current_price
        thesis['_drift_pct'] = 0.0
        thesis['_age_days'] = age_days
        thesis['_health_flags'] = [
            "HELD STATUS UNVERIFIED — could not read the portfolio, so no entry or exit "
            "directive is issued for this thesis. Re-check when holdings data returns."
        ] + stale_flags
        return thesis

    # 4. Dynamic entry-drift threshold based on volatility (beta). Higher-beta names
    # get a wider "entry missed" band; low-beta ETFs get a tight one.
    try:
        beta_str = data.get('beta', '1.0') if isinstance(data, dict) else '1.0'
        beta = float(beta_str) if beta_str not in ("N/A", "", None) else 1.0
    except Exception:
        beta = 1.0
    missed_threshold = 15.0 if beta > 1.2 else (5.0 if beta < 0.9 else 8.0)

    # 5. Build flags from the DISTINCT levels — vocabulary split by held status.
    flags = []
    drift_pct = 0.0

    # Entry drift — ONLY from a real recorded entry price in conditions, never the stop.
    reasonable_entries = [p for p in entry_prices if p < current_price * 1.5]
    highest_entry = max(reasonable_entries) if (is_long and reasonable_entries) else None
    if highest_entry is not None:
        drift_pct = ((current_price - highest_entry) / highest_entry) * 100

    if is_long and state == "watching":
        # NOT HELD — this is an entry plan being monitored for execution. Exit flags are
        # meaningless here; the only decisions available are enter, stand down, or re-base.
        if stop_price and current_price <= stop_price:
            flags.append(
                f"SETUP BROKEN PRE-ENTRY — Price ${current_price:.2f} is at/below the ${stop_price:.2f} stop "
                f"and the position was never opened. Nothing to sell. Re-base the entry/stop or drop the thesis — do NOT buy into it here."
            )
        elif highest_entry is not None and drift_pct > missed_threshold:
            flags.append(
                f"ENTRY MISSED — Price ${current_price:.2f} is {drift_pct:.1f}% above the ${highest_entry:.2f} entry zone "
                f"(Threshold: {missed_threshold:.0f}%) and was never entered. Stand down or re-base the entry — do NOT chase."
            )
        elif highest_entry is not None and drift_pct > 0:
            flags.append(
                f"APPROACHING ENTRY — Price ${current_price:.2f} is {drift_pct:.1f}% above the ${highest_entry:.2f} entry zone. "
                f"Not actionable yet; watch for the zone."
            )
        elif highest_entry is not None:
            # drift_pct <= 0: price is AT or INSIDE the buy zone. This is the moment the
            # user pinned the thesis to catch, and it previously produced no flag at all.
            flags.append(
                f"ENTRY TRIGGERED — Price ${current_price:.2f} is at/below the ${highest_entry:.2f} entry zone "
                f"({abs(drift_pct):.1f}% inside it) and the position is NOT yet open. This is the entry the thesis was waiting for: "
                f"execute it or explicitly stand down — do not default to 'monitor'."
            )
        if target_price and current_price >= target_price:
            flags.append(
                f"TARGET REACHED PRE-ENTRY — Price ${current_price:.2f} is at/above the ${target_price:.2f} target but the position "
                f"was never opened; the move happened without you. Nothing to take profit on. Re-base or drop the thesis."
            )
    elif is_long:
        # HELD — an open position with a live exit plan.
        if highest_entry is not None and drift_pct > missed_threshold:
            flags.append(f"ENTRY MISSED — Price ${current_price:.2f} is {drift_pct:.1f}% above the ${highest_entry:.2f} entry zone (Threshold: {missed_threshold:.0f}%)")
        elif highest_entry is not None and drift_pct > 0:
            flags.append(f"HOVERING ABOVE ENTRY — Price ${current_price:.2f} is {drift_pct:.1f}% above the ${highest_entry:.2f} entry zone. Check if entry was triggered recently.")

        # Target reached — the thesis's own objective is met; needs a take-profit/re-base
        # decision, not a default 'maintain'.
        if target_price and current_price >= target_price:
            flags.append(f"TARGET REACHED — Price ${current_price:.2f} is at/above the ${target_price:.2f} target. Decide: take profit or re-base the thesis — do not default to 'maintain'.")

        # Stop breached — a long thesis invalidated by its own stop.
        if stop_price and current_price <= stop_price:
            flags.append(f"STOP BREACHED — Price ${current_price:.2f} is at/below the ${stop_price:.2f} stop. The thesis is invalidated — close it.")

    flags.extend(stale_flags)

    thesis['_live_price'] = current_price
    thesis['_drift_pct'] = drift_pct
    thesis['_age_days'] = age_days
    thesis['_health_flags'] = flags

    return thesis


# Actions that express a bullish keep-or-accumulate stance. For these, a
# recommendation "hits" when the name beats SPY (alpha > 0). SELL/TRIM express a
# reduce stance, which hits when the name lags SPY (alpha < 0). HOLD belongs on
# the long side: scoring it with sell semantics (the prior bug) recorded every
# correct "keep holding a market-beater" call as a miss and poisoned the
# calibration track record injected into every prompt.
_LONG_BIAS_ACTIONS = frozenset({"BUY", "ADD", "HOLD"})

# Past calls older than this (days) are surfaced to the model with a "likely stale,
# re-derive" marker rather than as a bare line it can re-affirm. Kept short because
# the ledger's live use is tactical single-stock calls, where a multi-day-old entry
# price and thesis are already suspect against current data.
_RECOMMENDATION_STALE_DAYS = 5


def _is_long_bias_action(action: str) -> bool:
    """True for BUY/ADD/HOLD (hit == alpha > 0); False for SELL/TRIM/other (hit == alpha < 0)."""
    return (action or "").strip().upper() in _LONG_BIAS_ACTIONS


@log_exceptions()
def get_user_context() -> str:
    """
    Format memory as context string for injection into system prompts.
    Returns empty string if no meaningful memory exists.
    """
    memory = load_memory()
    context_parts = []

    # --- TIME AWARENESS ---
    current_date = datetime.now().strftime("%Y-%m-%d")
    context_parts.append(f"Current Date: {current_date}")
    context_parts.append("")

    # Canonical profile fields come first so quick-action prompts can bind to
    # the actual user profile before reading broader lessons and market context.
    profile = memory.get("user_profile", {})
    context_parts.extend(_format_profile_fields_block(profile))

    # 0. INJECT GLOBAL GOVERNANCE RULES (Highest Priority)
    context_parts.append("⚠️ CRITICAL INSTRUCTIONS & GOVERNANCE RULES:")
    context_parts.append("  ❌/✅ REVISED RULE 7 (ANTI-HALLUCINATION & SOURCE VERIFICATION): ANY AI-generated research — INCLUDING YOUR OWN OUTPUTS — is TIER 3 (Hypothesis, not Evidence). Before citing or generating ANY specific number, ask: Can I verify this directly in my tool outputs? If NO -> state 'I DON'T KNOW' or 'Data Unavailable'. DO NOT invent or estimate plausible numbers.")
    context_parts.append("  ❌/✅ NON-US STOCK DISCLOSURE: Quotes/fundamentals cover US, Canadian (TSX), Australian (ASX), and European (LSE, XETRA, Euronext) named tickers. Market-wide MOVERS scanning exists for US (scan_intraday_movers) and Canada/TSX (scan_tsx_movers) — use those tools, not news search, for 'what's moving' questions. For ASX/European discovery or exchanges not listed, disclose that data may be limited and default to Grade C confidence.")
    context_parts.append("")

    # 0.5. INJECT CALIBRATION (ADVISOR TRACK RECORD)
    past_recs = memory.get("past_recommendations", [])
    if past_recs:
        stats = {"HIGH": {"total": 0, "hits": 0}, "MEDIUM": {"total": 0, "hits": 0}, "LOW": {"total": 0, "hits": 0}}
        for rec in past_recs:
            # Exclude calls reversed/closed by a later opposite-bias call — counting a
            # superseded call in the self-grade the model calibrates against is exactly
            # the loop we are trying to break.
            if rec.get("superseded"):
                continue
            conf = rec.get("confidence_grade", "MEDIUM")
            scores = rec.get("scores", {})
            action = rec.get("action", "")
            long_bias = _is_long_bias_action(action)

            for key in ["2w", "1m", "3m"]:
                if key in scores:
                    alpha = scores[key]["alpha"]
                    is_hit = (alpha > 0) if long_bias else (alpha < 0)
                    if conf in stats:
                        stats[conf]["total"] += 1
                        if is_hit:
                            stats[conf]["hits"] += 1

        cal_parts = []
        for conf in ["HIGH", "MEDIUM", "LOW"]:
            total = stats[conf]["total"]
            if total > 0:
                hit_rate = (stats[conf]["hits"] / total) * 100.0
                cal_parts.append(f"{conf}-confidence calls hit {hit_rate:.1f}% (n={total})")
        if cal_parts:
            context_parts.append("  -- ADVISOR CALIBRATION TRACK RECORD --")
            context_parts.append(f"  Your measured historical accuracy against SPY: {'; '.join(cal_parts)}.")
            context_parts.append(
                "  This is a SELF-GRADE on a small sample, not external truth. Use it as a humility "
                "check on your own edge — it is NOT evidence for re-affirming a prior pick. If accuracy "
                "is low or n is small, re-derive each new call from live data rather than leaning on past conviction."
            )
            context_parts.append("")

    lessons = memory.get("lessons_learned", [])
    if lessons:
        context_parts.append("  -- USER LESSONS --")
        for lesson in lessons:
            context_parts.append(f"  ❌/✅ {lesson}")
    context_parts.append("") # spacer

    # Secular conviction themes — structural positions the user wants treated
    # differently than tactical/cyclical exposure. Trim recommendations on these
    # require a higher bar than generic rebalance math.
    secular_themes = memory.get("secular_themes", [])
    if secular_themes:
        context_parts.append("  -- STRUCTURAL CONVICTION (SECULAR THEMES) --")
        context_parts.append(
            "  These are multi-year secular positions, not tactical trades. "
            "Apply the per-theme trim_triggers / do_not_trim_for rules strictly: "
            "DO NOT recommend trimming a position tagged to one of these themes "
            "unless at least one trim_trigger is currently met. RSI/overbought/"
            "rotation-into-laggards arguments alone are insufficient."
        )
        for theme in secular_themes:
            name = theme.get("theme") or "Unnamed theme"
            conviction = theme.get("conviction", "medium")
            horizon = theme.get("horizon", "")
            rationale = theme.get("rationale", "")
            context_parts.append(f"  🎯 {name}  [conviction={conviction}, horizon={horizon}]")
            if rationale:
                context_parts.append(f"     • Rationale: {rationale}")
            triggers = theme.get("trim_triggers") or []
            if triggers:
                context_parts.append("     • Trim ONLY if any of:")
                for t in triggers:
                    context_parts.append(f"        - {t}")
            blockers = theme.get("do_not_trim_for") or []
            if blockers:
                context_parts.append("     • DO NOT trim for:")
                for b in blockers:
                    context_parts.append(f"        - {b}")
        context_parts.append("")  # spacer

    # User profile
    if profile.get("name"):
        context_parts.append(f"User's name: {profile['name']}")
    if profile.get("age"):
        context_parts.append(f"User's age: {profile['age']}")
    if profile.get("risk_tolerance"):
        context_parts.append(f"Risk tolerance: {profile['risk_tolerance']}")
    if profile.get("retirement_age"):
        context_parts.append(f"Target retirement age: {profile['retirement_age']}")
    base_currency = get_profile_base_currency(profile)
    if profile.get("annual_income"):
        income = profile["annual_income"]
        # income may be stored as string ("$150,000") or int — handle both safely
        try:
            income_numeric = re.sub(r"[^0-9.]", "", str(income))
            income_display = f"${float(income_numeric):,.0f}"
        except (ValueError, TypeError):
            income_display = str(income)
        context_parts.append(f"Annual income ({base_currency}): {income_display} {base_currency}")
    if base_currency:
        context_parts.append(f"Base currency: {base_currency}")
    if profile.get("investment_goals"):
        context_parts.append(f"Investment goals: {', '.join(profile['investment_goals'])}")
    if profile.get("accounts"):
        context_parts.append(f"Investment accounts: {', '.join(profile['accounts'])}")

    # Key facts (Strict limit: 5)
    key_facts = memory.get("key_facts", [])
    if key_facts:
        context_parts.append("\nKey facts about user:")
        for fact in key_facts[-5:]:  # Reduced from 10 to 5
            context_parts.append(f"  • {fact}")

    # Recent conversation summaries (Strict limit: 2)
    summaries = memory.get("conversation_summaries", [])
    if summaries:
        context_parts.append("\nRecent conversations:")
        for summary in summaries[-2:]:  # Reduced from 3 to 2
            context_parts.append(f"  • {summary.get('date', 'Unknown')}: {summary.get('summary', '')}")

    # Prior recommendations (Strict limit: 3). This is a PROVENANCE TRAIL of past
    # calls, NOT a list of standing positions. Replaying a bare "BUY LITE" as a
    # present-tense premise is exactly how a stale call gets re-affirmed days later
    # without re-derivation — so each line now carries its age and entry price, and
    # the header explicitly instructs RE-VERIFY before restating. Calls closed out by
    # a later opposite-bias call for the same ticker (superseded) are dropped so a
    # contradicted BUY can't keep resurfacing next to the SELL that closed it.
    all_recommendations = memory.get("past_recommendations", [])
    recommendations = [r for r in all_recommendations if not r.get("superseded")]
    if recommendations:
        now = datetime.now()
        context_parts.append(
            "\nPrior recommendations — PAST calls to RE-VERIFY against live data before "
            "restating, NOT current positions or open orders:"
        )
        shown = recommendations[-3:]  # Reduced from 5 to 3
        if any(r.get("executed") is None for r in shown):
            context_parts.append(
                "  Rows marked 'user action: NOT RECORDED' are exactly that: this ledger is "
                "written from the ADVISOR's text, so it stores what was advised, never what "
                "was done. For those you cannot tell an executed call from one the user "
                "declined — never state or imply either, and never read it off the portfolio "
                "(absent could mean declined, unfilled, or since sold). If it matters, ask."
            )
        for rec in shown:
            date_str = rec.get('date', '')
            age_txt = ""
            try:
                # Staleness is measured from the last re-affirmation, not the
                # original anchor date: a stance restated today was re-derived
                # today, even though its scoring clock (date/entry price) stays
                # pinned to the first call.
                affirmed_str = rec.get('last_affirmed') or date_str
                age_days = (now - datetime.strptime(affirmed_str, "%Y-%m-%d")).days
                if age_days <= 0:
                    age_txt = " (today)"
                elif age_days >= _RECOMMENDATION_STALE_DAYS:
                    age_txt = f" ({age_days}d ago — ⚠️ likely stale, re-derive)"
                else:
                    age_txt = f" ({age_days}d ago)"
            except Exception:
                pass
            price = rec.get('price_at_advice')
            price_txt = f" @ entry ${price}" if price else ""
            executed = rec.get('executed')
            if executed is True:
                exec_txt = " — user action: EXECUTED"
            elif executed is False:
                exec_txt = " — user action: DECLINED (user did NOT act on this)"
            else:
                exec_txt = " — user action: NOT RECORDED"
            note = str(rec.get('execution_note') or "").strip()
            if note and executed is not None:
                exec_txt += f" ({note})"
            context_parts.append(
                f"  • {date_str}{age_txt}: {rec.get('action', '')} {rec.get('ticker', '')}{price_txt}{exec_txt}"
            )
    elif all_recommendations:
        # Rows exist but every one was closed out by a later opposite-bias call, so none
        # render. That is NOT an empty history — saying "no calls on record" here would
        # be its own fabrication, in the opposite direction.
        context_parts.append(
            f"\nPrior recommendations — NONE OPEN. {len(all_recommendations)} logged call(s) "
            "were each superseded by a later opposite call and are withheld from you. Calls "
            "WERE made for this user: do not claim an empty recommendation history. You "
            "cannot see the closed ones, so do not name, date, or assign an outcome or "
            "reason to any of them."
        )
    else:
        # Same reasoning as the empty-theses branch below: an omitted block is read as
        # "not provided" and gets back-filled. A brief once narrated "past rotation
        # targets (two tickers) failed strict entry rules" with this ledger empty and
        # neither ticker anywhere in the profile — the reasons were borrowed from real
        # user lessons, which is what made it read as sourced.
        context_parts.append(
            "\nPrior recommendations — NONE ON RECORD. No call has been logged for this "
            "user. This is the complete and authoritative history, not a missing block: "
            "do NOT claim that any name was previously recommended, evaluated, screened, "
            "rejected, entered, or passed over. A ticker seen in a scan, funnel, or "
            "watchlist was only SEEN — that is not an evaluation, and its absence from "
            "the portfolio does not reveal why it is absent."
        )

    if not context_parts:
        return ""

    # Integrate Curated Graph Memory
    # Extract structured portfolio context for the LLM to reason over.
    try:
        graph_memory.prune_stale()
        graph_parts = []
        holdings_compact = []   # Compact "AAPL(Tech)" list, assembled after the loop
        owned_symbols = []      # Tickers from OWNS edges (graph fallback for holdings)
        sector_by_symbol = {}   # {symbol: sector} from IN_SECTOR edges (annotation only)
        sector_exposure = []    # "Tech: 25.9%"
        themes = []             # "GLD→Gold"
        correlations = []       # "AAPL↔MSFT (0.85)"
        identities = []         # User identity edges
        interests = []          # User interest edges

        for u, v, data in graph_memory.graph.edges(data=True):
            rel = data.get("relation", "")

            if rel == "OWNS":
                # Ownership lives on Portfolio->TICKER (legacy: User->TICKER) edges.
                # This is the graph's record of WHAT is held; sector is layered on below.
                if u in ("Portfolio", "User"):
                    owned_symbols.append(v)
            elif rel == "IN_SECTOR":
                sector_by_symbol[u] = v
            elif rel == "EXPOSED_TO":
                pct = data.get("percentage", "")
                if pct:
                    sector_exposure.append(f"{v}: {pct}")
            elif rel == "CORRELATED_WITH":
                strength = data.get("strength", "")
                if strength:
                    correlations.append(f"{u}↔{v} ({strength})")
            elif u == "User":
                if rel in ["HAS_ACCOUNT", "HAS_ACCOUNT_AT", "HAS_RISK_TOLERANCE",
                            "WORKS_AT", "LIVES_IN", "PREFERS"]:
                    identities.append(f"{rel.replace('_', ' ').title()}: {v}")
                elif rel in ["INTERESTED_IN", "MONITORS", "TRACKING"]:
                    interests.append(f"{v}")

        # --- Assemble the holdings line from the SOURCE OF TRUTH ---
        # Previously this was derived from IN_SECTOR edges alone. Only sector-tagged
        # names land there (often just one), which made the injected context claim the
        # user held a single stock (e.g. "Holdings: AAPL(Tech) [1 total]") and led
        # downstream nodes to report a bogus "100% Technology" portfolio. Prefer the
        # live portfolio summary (cached — never force a recompute in this hot path),
        # fall back to graph OWNS edges, then to sector-tagged names, so we never
        # regress to a one-symbol view of a diversified book.
        holding_syms = []
        try:
            from tools.cache import get_cached
            from tools.portfolio_csv import is_demo_mode
            _pkey = "demo_portfolio_summary" if is_demo_mode() else "portfolio_summary"
            _psum = get_cached(_pkey, ttl_seconds=900)
            if isinstance(_psum, dict):
                for h in _psum.get("holdings", []):
                    s = h.get("symbol")
                    if s and s != "CASH" and not h.get("is_cash_or_pension"):
                        holding_syms.append(s)
        except Exception:
            holding_syms = []
        if not holding_syms:
            holding_syms = owned_symbols or list(sector_by_symbol.keys())

        _seen = set()
        for sym in holding_syms:
            if sym in _seen:
                continue
            _seen.add(sym)
            sec = sector_by_symbol.get(sym)
            holdings_compact.append(f"{sym}({sec[:4]})" if sec else sym)

        # Extract thematic exposure from node attributes (stored during cleanup)
        for node, data in graph_memory.graph.nodes(data=True):
            theme = data.get("theme")
            if theme and data.get("owned"):
                themes.append(f"{node}→{theme}")

        has_data = any([holdings_compact, sector_exposure, themes,
                        correlations, identities, interests])
        if has_data:
            graph_parts.append("\n=== 🧠 PORTFOLIO GRAPH ===")
            if holdings_compact:
                # Show the full book (CSV + live broker sync, e.g. Questrade/Alpaca).
                # Capped generously to stay bounded without truncating a normal
                # multi-account portfolio down to a misleadingly small subset.
                graph_parts.append(f"Holdings: {', '.join(holdings_compact[:60])} [{len(holdings_compact)} total]")
            if sector_exposure:
                graph_parts.append(f"Sectors: {', '.join(sector_exposure[:8])}")
            if themes:
                graph_parts.append(f"Themes: {', '.join(themes)}")
            if correlations:
                graph_parts.append(f"Correlated: {', '.join(correlations[:5])}")
            if identities:
                graph_parts.append(f"User: {', '.join(identities)}")
            if interests:
                graph_parts.append(f"Tracking: {', '.join(interests[:10])}")
            graph_parts.append("===========================")
            context_parts.append("\n".join(graph_parts))
    except Exception as e:
        safe_print(f"⚠️ Error formatting graph context: {e}")

    # Integrate Global Market Context
    # This ensures all agents are aware of macro regime and sector rotations automatically
    try:
        from tools.macro_strategy import analyze_macro_context
        from tools.market_mechanics import detect_sector_rotation

        macro = analyze_macro_context()
        rot = detect_sector_rotation()

        context_parts.append("\n=== 🌍 GLOBAL MARKET CONTEXT ===")
        context_parts.append(f"Regime: {macro.get('current_regime', 'Unknown')}")
        context_parts.append(f"Leaders: {', '.join(rot.get('leading_sectors', []))}")
        context_parts.append(f"Laggards: {', '.join(rot.get('lagging_sectors', []))}")
        if "Tactical: " in (macro.get('plain_english') or ''):
             context_parts.append(macro['plain_english'])
        context_parts.append("=============================\n")
    except Exception:
        pass

    # ACTIVE THESES (High Priority)
    active_theses = memory.get("active_theses", [])
    if active_theses:
        context_parts.append("\n=== 📌 ACTIVE INVESTMENT THESES (PRIORITY) ===")
        context_parts.append("NOTE: Each thesis below includes live price context. If any thesis shows")
        context_parts.append("⚠️ flags, you MUST address them proactively and recommend updating or closing")
        context_parts.append("the thesis instead of defaulting to 'maintain hold'.")
        context_parts.append("Each thesis is tagged HELD or WATCHING (verified against actual holdings).")
        context_parts.append("WATCHING = an entry plan the user is monitoring to execute, NOT an open position:")
        context_parts.append("it can be entered, re-based, or dropped — it can NEVER be sold, trimmed, or")
        context_parts.append("'taken profit' on, and being absent from the portfolio is expected, not a")
        context_parts.append("contradiction to reconcile by deleting the thesis.\n")
        # One portfolio read for the whole block (5-min cached) instead of per-thesis.
        held = _held_base_symbols()
        for thesis in active_theses:
            thesis = _enrich_thesis_with_price_context(thesis, held=held)
            symbol = thesis.get('symbol', 'UNKNOWN')
            action = thesis.get('action', 'HOLD')
            stop = thesis.get('stop_loss', 'N/A')
            catalyst = thesis.get('catalyst', 'N/A')
            expiry = thesis.get('expiry_date', 'N/A')
            cond = thesis.get('conditions', 'N/A')
            notes = thesis.get('notes', '')
            state = thesis.get('_position_state', 'unknown')
            state_label = {
                'held': 'HELD (open position)',
                'watching': 'WATCHING (entry plan — not held, nothing to sell)',
            }.get(state, 'HELD STATUS UNVERIFIED')

            context_parts.append(f"📌 {symbol} [{action}] — {state_label} — until {expiry}")

            # Inject live context
            live_price = thesis.get('_live_price')
            if live_price:
                context_parts.append(f"   • Current Price: ${live_price:.2f}")
            for flag in thesis.get('_health_flags', []):
                context_parts.append(f"   🚨 ⚠️ {flag}")

            context_parts.append(f"   • Stop Loss: {stop}")
            context_parts.append(f"   • Catalyst: {catalyst}")
            context_parts.append(f"   • Logic: {cond}")
            if notes:
                context_parts.append(f"   • Context: {notes}")

            # Reconciliation flag: a later reduce call (SELL/TRIM) for this ticker was
            # logged to the recommendation ledger, contradicting this still-open long
            # thesis. Surface it so the model closes/updates the thesis instead of
            # re-affirming a position its own advice has already exited.
            # Only meaningful for a HELD position — a sell logged against a name the user
            # never owned contradicts nothing, and rendering it drove the model to cancel
            # a live watchlist thesis. Render-time gate as well as the stamp-time gate in
            # add_recommendation, so stamps already written to memory go quiet too.
            exit_sig = thesis.get('exit_signal')
            if exit_sig and state == 'held':
                context_parts.append(
                    f"   🚨 ⚠️ CONTRADICTED — a {exit_sig.get('action')} for {symbol} was later logged "
                    f"on {exit_sig.get('date')}. Reconcile: UPDATE or CLOSE this thesis; do not default to 'maintain'."
                )
        context_parts.append("=== END ACTIVE THESES ===\n")
    else:
        # State the absence explicitly. Emitting NOTHING here reads as "this block was
        # not provided", and a model that wants a thesis will happily supply one — a
        # Today's Priority brief once put a fabricated entry plan (stop, zone and a
        # [WATCHING] tag) on the radar with active_theses empty. A negative fact is
        # grounding; silence is an invitation.
        context_parts.append("\n=== 📌 ACTIVE INVESTMENT THESES (PRIORITY) ===")
        context_parts.append(
            "NONE ON RECORD — the user has no active theses. This is the complete and "
            "authoritative list, not a missing block: do NOT reference, recall, or "
            "reconstruct a thesis, entry zone, stop, target, or [WATCHING]/[HELD] tag "
            "for ANY ticker. There are no pre-committed plans to triage."
        )
        context_parts.append("=== END ACTIVE THESES ===\n")

    return "=== USER MEMORY (Recent Context) ===\n" + "\n".join(context_parts) + "\n=== END MEMORY ===\n"


@log_exceptions()
def clean_memory(target: str = "all") -> str:
    """
    Clean the memory to start fresh or remove specific parts.
    Target options: 'all', 'facts', 'profile', 'history'.
    """
    memory = load_memory()

    if target == "all":
        memory = copy.deepcopy(DEFAULT_MEMORY)
        # Also clean Graph Memory
        try:
            from tools.user_profile import get_data_path
            graph_file = get_data_path("knowledge_graph.json")
            if os.path.exists(graph_file):
                os.remove(graph_file)
        except Exception as e:
            safe_print(f"⚠️ Failed to clear Graph Memory: {e}")

        msg = "Memory completely wiped (Flat + Graph)."
    elif target == "facts":
        memory["key_facts"] = []
        msg = "Key facts cleared."
    elif target == "profile":
        memory["user_profile"] = copy.deepcopy(DEFAULT_MEMORY["user_profile"])
        msg = "User profile reset."
    elif target == "history":
        memory["conversation_summaries"] = []
        memory["past_recommendations"] = []
        msg = "Conversation history and recommendations cleared."
    else:
        return f"Unknown target: {target}"

    save_memory(memory)
    return msg


RISK_CONSTRAINT_KEYS = (
    "max_position_pct",
    "max_fund_position_pct",
    "max_sector_pct",
    "max_risk_per_trade_pct",
)

# Where the "yes, I mean unconstrained" answer is recorded. Not a cap and never
# read as one: it names the axes the user has deliberately left open, so a blank
# the user CHOSE can be told apart from a blank nobody has ever been asked
# about. Written only through the acknowledge_unconstrained flag below.
UNCONSTRAINED_ACK_KEY = "unconstrained_ack"


@log_exceptions()
def set_risk_constraints(updates: dict[str, Any]) -> dict[str, Any]:
    """Set or clear the user's own numeric risk limits.

    Pass None (or any non-positive value) for a key to CLEAR that limit, which
    means the user accepts unbounded risk on that axis — a cleared limit is
    enforced nowhere and must never be quoted back to them. This is the only
    supported way to create a limit: nothing in the app may supply a default,
    because a cap the user did not choose still gets cited as theirs.

    `acknowledge_unconstrained` is the one meta-key: True records that the user
    accepts every axis still blank AFTER this write as unconstrained, False
    withdraws that. It is applied last so a save that sets a cap and confirms
    the rest in one click records exactly the axes that ended up blank. Because
    the acknowledgement stores the axis NAMES rather than a bare flag, later
    clearing a cap leaves that axis outside the acknowledged set — a limit the
    user deletes goes back to being an unanswered blank rather than inheriting a
    confirmation given about different axes.

    Returns the resulting risk_constraints block.
    """
    memory = load_memory()
    constraints = memory.get("risk_constraints")
    if not isinstance(constraints, dict):
        constraints = {}

    acknowledge = (updates or {}).get("acknowledge_unconstrained")

    for key, value in (updates or {}).items():
        if key == "restricted_symbols":
            if value is None:
                constraints.pop(key, None)
            elif isinstance(value, (list, tuple, set)):
                constraints[key] = sorted({str(s).upper().strip() for s in value if str(s).strip()})
            continue
        if key not in RISK_CONSTRAINT_KEYS:
            continue
        if value is None:
            constraints.pop(key, None)
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue  # keep the existing limit; see below
        # Only an explicit None clears. A malformed or non-positive figure is
        # REJECTED, leaving any existing limit standing — a typo must never
        # silently delete a protection the user deliberately set, and storing 0
        # would read as "risk nothing", which is not what anyone means.
        if number > 0:
            constraints[key] = number

    if acknowledge is True:
        blank = [key for key in RISK_CONSTRAINT_KEYS if key not in constraints]
        constraints[UNCONSTRAINED_ACK_KEY] = {
            "acknowledged_at": datetime.now().isoformat(timespec="seconds"),
            "axes": blank,
        }
    elif acknowledge is False:
        constraints.pop(UNCONSTRAINED_ACK_KEY, None)

    memory["risk_constraints"] = constraints
    save_memory(memory)
    return constraints


# ---------------------------------------------------------------------------
# Target allocation — the sleeve mix the user is actually trying to hold
# ---------------------------------------------------------------------------
# AUTHORED CONSTANT (2.7). How far the stated weights may sum from 100% before
# the store refuses them. Not measured — it exists to catch a typo while leaving
# room for the rounding a human does by hand.
TARGET_ALLOCATION_SUM_TOLERANCE_PCT = 0.5

TARGET_ALLOCATION_KEY = "target_allocation"


@log_exceptions()
def get_target_allocation() -> dict[str, float] | None:
    """The user's stated target sleeve mix, or None if they have never set one.

    None is MEANINGFUL and is never filled in. There is no house default mix:
    a target allocation is a statement about someone's plan, and one this app
    invented would be quoted back to them as their own — then used to generate
    BUY and SELL instructions against a plan nobody chose.
    """
    stored = load_memory().get(TARGET_ALLOCATION_KEY)
    if not isinstance(stored, dict):
        return None
    weights = stored.get("weights")
    if not isinstance(weights, dict) or not weights:
        return None
    return {str(k).upper().strip(): float(v) for k, v in weights.items()}


@log_exceptions()
def get_target_allocation_record() -> dict[str, Any] | None:
    """The stored block with its metadata (`weights`, `set_at`, `note`)."""
    stored = load_memory().get(TARGET_ALLOCATION_KEY)
    return stored if isinstance(stored, dict) and stored.get("weights") else None


@log_exceptions()
def set_target_allocation(weights: dict[str, Any] | None,
                          note: str = "") -> dict[str, Any]:
    """Store the target sleeve mix. `None` or `{}` clears it.

    **The weights are NOT normalized, and that is the load-bearing decision.**
    The explicit-argument path in `check_rebalance_drift` rescales whatever it is
    handed to sum to 1, which is right for a caller passing a mix inline. It is
    wrong for a stored plan: a user who types sleeves summing to 90% has either
    forgotten one or is holding 10% cash on purpose, and silently rescaling turns
    the second into "spread that cash across everything" — which then emits real
    BUY instructions for money they meant to keep aside.

    So a mix that does not sum to ~100% is REFUSED with its own total quoted
    back, and the fix is the user's to make: add the missing sleeve, or name cash
    as one. Returns ``{"ok": bool, ...}``; never raises, never partially writes.
    """
    memory = load_memory()

    if not weights:
        memory.pop(TARGET_ALLOCATION_KEY, None)
        save_memory(memory)
        return {"ok": True, "cleared": True, "target_allocation": None}

    cleaned: dict[str, float] = {}
    rejected: list[str] = []
    for key, value in weights.items():
        symbol = str(key or "").upper().strip()
        if not symbol:
            continue
        try:
            pct = float(value)
        except (TypeError, ValueError):
            rejected.append(symbol)
            continue
        # A zero or negative sleeve is not a target, it is an omission. Refused
        # rather than stored, so it cannot later read as a deliberate 0% target.
        if pct > 0:
            cleaned[symbol] = round(pct, 4)
        else:
            rejected.append(symbol)

    if not cleaned:
        return {"ok": False,
                "error": "no usable positive weights",
                "rejected": sorted(rejected)}

    # A rejected sleeve is REPORTED, never dropped on the way to a successful
    # write. `CASH 0` silently vanishing while the remaining sleeves happened to
    # total 100% would store a plan the user did not type — and the check that
    # would have caught it, the sum, passes precisely because the dropped line
    # contributed nothing. Same silent-drop failure the line parser refuses.
    if rejected:
        return {
            "ok": False,
            "error": (
                f"these sleeves have no usable positive weight: {', '.join(sorted(rejected))}. "
                "A 0% sleeve is an omission rather than a target — remove the line, or give "
                "it the weight you mean."
            ),
            "rejected": sorted(rejected),
        }

    total = round(sum(cleaned.values()), 4)
    if abs(total - 100.0) > TARGET_ALLOCATION_SUM_TOLERANCE_PCT:
        return {
            "ok": False,
            "error": (
                f"weights sum to {total:g}%, not 100%. They are stored as written "
                "rather than rescaled: if the remainder is cash, add it as its own "
                "sleeve (e.g. CASH) so the drift check knows not to invest it."
            ),
            "total_pct": total,
            "weights": cleaned,
        }

    record = {
        "weights": cleaned,
        "total_pct": total,
        "set_at": datetime.now().isoformat(timespec="seconds"),
        "note": str(note or "").strip(),
    }
    memory[TARGET_ALLOCATION_KEY] = record
    save_memory(memory)
    return {"ok": True, "target_allocation": record}


# --- 4.7a — tax jurisdiction, stated per ACCOUNT ----------------------------
#
# The tax treatment of a shelter is a property of the COUNTRY, and until this
# store existed the only evidence in the codebase was the account NAME. That
# heuristic is load-bearing and has already been wrong twice in ways that
# inverted the answer ("ISA" matching *Visa*, "REGISTERED" matching
# *Non-Registered*), and it cannot see the cases where the name is simply silent
# — "Brokerage", "Pension", "Joint" name a class without naming a country.
#
# Jurisdiction lives on the ACCOUNT and not on the profile because one household
# can hold accounts in two countries; `REGIONAL_LOCALE` is a display locale and
# keying tax off it is the error this store exists to make impossible.
#
# Unset stays unset. There is no default and no inheritance from the locale: an
# account nobody has answered for keeps failing closed, which is the behaviour
# that shipped before this store and is the correct one.

ACCOUNT_JURISDICTIONS_KEY = "account_jurisdictions"

# The third state, and the reason this is not a plain string->string map. An
# account can be ANSWERED with "I cannot name a country for this one" — a real
# answer that must not read as an unasked question. Same distinction 2.9's
# `unconstrained_ack` draws for risk limits: a finished profile and an untouched
# one are otherwise identical from downstream.
JURISDICTION_UNKNOWN = "UNKNOWN"

_JURISDICTION_CODE_RE = re.compile(r"^[A-Z]{2}$")


def normalize_account_key(account_name: Any) -> str:
    """The key an account name is stored and looked up under.

    Upper-cased, trimmed, and internal whitespace collapsed — because the same
    account arrives as "RRSP  Spousal" from one source and "rrsp spousal" from
    another, and a stored jurisdiction that fails to match its own account is a
    dark store that looks filled in.
    """
    return " ".join(str(account_name or "").upper().split())


@log_exceptions()
def get_account_jurisdictions() -> dict[str, str]:
    """Stated jurisdiction per account: ``{NORMALIZED NAME: "CA" | "UNKNOWN"}``.

    An account absent from this map has NOT been answered — callers must treat
    that as unknown and fail closed, never as a licence to guess. Never raises;
    an unreadable store reads as "nothing stated", which is the safe direction.
    """
    stored = load_memory().get(ACCOUNT_JURISDICTIONS_KEY)
    if not isinstance(stored, dict):
        return {}
    accounts = stored.get("accounts")
    if not isinstance(accounts, dict):
        return {}
    resolved: dict[str, str] = {}
    for name, code in accounts.items():
        key = normalize_account_key(name)
        value = str(code or "").upper().strip()
        if key and (value == JURISDICTION_UNKNOWN or _JURISDICTION_CODE_RE.match(value)):
            resolved[key] = value
    return resolved


@log_exceptions()
def get_account_jurisdictions_record() -> dict[str, Any] | None:
    """The stored block with its metadata (`accounts`, `set_at`, `note`)."""
    stored = load_memory().get(ACCOUNT_JURISDICTIONS_KEY)
    return stored if isinstance(stored, dict) and stored.get("accounts") else None


@log_exceptions()
def set_account_jurisdictions(accounts: dict[str, Any] | None,
                              note: str = "") -> dict[str, Any]:
    """Store the per-account jurisdictions. ``None`` or ``{}`` clears the block.

    Values are two-letter country codes, or ``UNKNOWN`` to record that the
    question was asked and has no answer. A blank value REMOVES that account
    from the map, returning it to unanswered.

    **A code this app does not have rules for is stored, not refused.** The
    engines are the right place to say "no policy module covers DE" — refusing
    the entry instead would make an unsupported jurisdiction indistinguishable
    from a typo, and would quietly push the user toward naming a country we
    happen to cover rather than the one their account is in. What IS refused is
    something that cannot be a country code at all, reported per entry rather
    than dropped, following the target-allocation precedent: a silently ignored
    line stores a plan the user did not type.

    Returns ``{"ok": bool, ...}``; never raises, never partially writes.
    """
    memory = load_memory()

    if not accounts:
        memory.pop(ACCOUNT_JURISDICTIONS_KEY, None)
        save_memory(memory)
        return {"ok": True, "cleared": True, "account_jurisdictions": None}

    cleaned: dict[str, str] = {}
    rejected: list[str] = []
    for name, code in accounts.items():
        key = normalize_account_key(name)
        if not key:
            continue
        value = str(code or "").upper().strip()
        if not value:
            # An explicit blank clears that account rather than storing "".
            continue
        if value == JURISDICTION_UNKNOWN or _JURISDICTION_CODE_RE.match(value):
            cleaned[key] = value
        else:
            rejected.append(f"{key} → {value}")

    if rejected:
        return {
            "ok": False,
            "error": (
                "these are not country codes: " + ", ".join(sorted(rejected))
                + ". Use a two-letter code (CA, US, GB, AU) or UNKNOWN to record "
                "that the account has no jurisdiction you can name."
            ),
            "rejected": sorted(rejected),
        }

    if not cleaned:
        memory.pop(ACCOUNT_JURISDICTIONS_KEY, None)
        save_memory(memory)
        return {"ok": True, "cleared": True, "account_jurisdictions": None}

    record = {
        "accounts": cleaned,
        "set_at": datetime.now().isoformat(timespec="seconds"),
        "note": str(note or "").strip(),
    }
    memory[ACCOUNT_JURISDICTIONS_KEY] = record
    save_memory(memory)
    return {"ok": True, "account_jurisdictions": record}


# Logical goal key -> user_profile field. The wealth goal lives ON the profile
# (one home for "what the user wants") — these are just typed accessors over the
# three fields, so there is no second store to drift.
_FINANCIAL_GOAL_FIELDS = {
    "target_low": "goal_target_low",
    "target_high": "goal_target_high",
    "horizon_years": "goal_horizon_years",
    "annual_contribution": "goal_annual_contribution",
}


def get_financial_goal() -> dict[str, Any] | None:
    """Assemble the user's structured wealth goal from the profile, or None.

    Backed by user_profile fields (goal_target_low / goal_target_high /
    goal_horizon_years / goal_annual_contribution), in TODAY's terms and the
    profile's base currency. Returns None when nothing is set — callers must
    treat that as "unavailable" (no required-CAGR, no goal-funded probability)
    and never substitute a default. Individual keys may be None even when the
    goal exists, and each must be checked on its own: a target with no
    contribution is a real, partially-specified state, not a reason to invent
    an inflow.
    horizon_years is the goal's OWN horizon, NOT retirement_age - age.
    """
    profile = load_memory().get("user_profile") or {}
    values = {logical: profile.get(field) for logical, field in _FINANCIAL_GOAL_FIELDS.items()}
    if not any(v is not None for v in values.values()):
        return None
    values["currency"] = get_profile_base_currency(profile)
    return values


@log_exceptions()
def set_financial_goal(updates: dict[str, Any]) -> dict[str, Any] | None:
    """Set or clear the wealth goal (stored on the user profile, today's terms).

    Keys: target_low, target_high, horizon_years. Pass None (or a non-positive
    number) to CLEAR that field; a malformed value is rejected and leaves the
    existing figure standing, so a typo never silently erases a goal. Nothing here
    supplies a target the user did not choose. Returns the resulting goal, or None
    if nothing is set.
    """
    memory = load_memory()
    profile = memory.setdefault("user_profile", {})
    for logical, field in _FINANCIAL_GOAL_FIELDS.items():
        if logical not in (updates or {}):
            continue
        value = updates[logical]
        if value is None:
            profile[field] = None
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue  # keep the existing figure; a typo must not erase a goal
        if number > 0:
            profile[field] = int(number) if field == "goal_horizon_years" else number
    profile["last_updated"] = datetime.now().isoformat()
    save_memory(memory)
    return get_financial_goal()


@log_exceptions()
def update_profile(updates: dict[str, Any]) -> None:
    """Update user profile with new information."""
    memory = load_memory()
    for key, value in updates.items():
        if key in memory["user_profile"] and value is not None:
            if key == "base_currency":
                value = normalize_base_currency(value)
            memory["user_profile"][key] = value
    memory["user_profile"]["last_updated"] = datetime.now().isoformat()
    save_memory(memory)

    # --- PROACTIVE MIRRORING (Box -> Graph) ---
    # Sync critical attributes to Graph for Deep Reasoning
    if "risk_tolerance" in updates and updates["risk_tolerance"]:
        val = str(updates["risk_tolerance"]).title() # e.g. "Aggressive"
        graph_memory.add_relationship("User", val, "HAS_RISK_TOLERANCE")
        safe_print(f"🕸️ Mirrored Risk Tolerance '{val}' to Graph")


def _find_near_duplicate(text: str, existing: list[str], threshold: float = 0.6) -> str | None:
    """Return an existing entry that's a near-duplicate of `text`, if any.

    Catches reworded repeats (not just exact string matches) so a capped,
    prompt-injected list (facts, lessons) doesn't silently fill up with
    variations of the same point.
    """
    norm = text.strip().lower()
    for item in existing:
        if difflib.SequenceMatcher(None, norm, item.strip().lower()).ratio() >= threshold:
            return item
    return None


@log_exceptions()
def add_fact(fact: str) -> None:
    """Add a new key fact about the user (avoids exact and near-duplicates)."""
    if not fact or len(fact) < 5:
        return

    memory = load_memory()
    existing = memory["key_facts"]
    if fact in existing:
        return
    similar = _find_near_duplicate(fact, existing)
    if similar:
        safe_print(f"💾 Skipped near-duplicate fact (similar to: {similar[:80]})")
        return

    existing.append(fact)
    # Keep only last 50 facts
    memory["key_facts"] = existing[-50:]
    save_memory(memory)


@log_exceptions()
def add_conversation_summary(summary: str, thread_id: str | None = None) -> None:
    """Add or update a conversation's summary.

    Post-processing re-summarizes the whole transcript-so-far at the end of every
    turn (so the summary keeps improving as a long session grows), not just once.
    When thread_id is given, this replaces that thread's existing entry in place
    instead of appending a new one each turn — otherwise a single long session
    floods the 20-entry cap with near-duplicates of itself and evicts every other
    session's summary.
    """
    if not summary:
        return

    memory = load_memory()
    summaries = memory.setdefault("conversation_summaries", [])
    entry = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "summary": summary,
        "thread_id": thread_id,
    }
    if thread_id:
        for i, existing in enumerate(summaries):
            if existing.get("thread_id") == thread_id:
                summaries[i] = entry
                break
        else:
            summaries.append(entry)
    else:
        summaries.append(entry)
    # Keep only last 20 summaries
    memory["conversation_summaries"] = summaries[-20:]
    save_memory(memory)


# Every lesson is injected into EVERY prompt, so the store stays bounded.
# 10 → 15 with FIFO truncation on the user's call 2026-07-27.
LESSON_CAP = 15

# add_lesson outcomes. A caller has to be able to tell "already knew that" from
# "this cost you the oldest rule", because only one of them needs reporting.
LESSON_ADDED = "added"
LESSON_DUPLICATE = "duplicate"
LESSON_EVICTED = "evicted"
LESSON_EMPTY = "empty"


def lessons_pending_eviction() -> list[str]:
    """The rules the NEXT ``add_lesson`` would retire — empty when there is room.

    Read this BEFORE adding. Once ``add_lesson`` has run the retired text is gone
    from the store and nothing can reconstruct it, so this is the only way a
    caller can name what the write cost. Returns a list because a cap that is
    lowered leaves stores above it, and one add then trims several.
    """
    existing = load_memory().get("lessons_learned", []) or []
    overflow = len(existing) + 1 - LESSON_CAP
    return list(existing[:overflow]) if overflow > 0 else []


@log_exceptions()
def add_lesson(lesson: str) -> str:
    """Record a corrective lesson from the user (avoids exact and near-duplicates).

    AT CAP THIS RETIRES THE OLDEST RULE to make room and returns
    ``LESSON_EVICTED`` (user's call 2026-07-27, replacing 1.7's refusal).

    What 1.7 actually got wrong was not eviction, it was SILENCE: the old
    implementation dropped a rule the user wrote months ago out of every prompt
    and announced it with a ``safe_print`` on a server nobody reads. So eviction
    is back, but it is loud — the distinct return code exists so every caller can
    name the casualty, ``lessons_pending_eviction`` lets them capture the text
    before it goes, and /context warns which rule is next BEFORE the user adds.
    Keep that chain intact: a silent eviction here is the invisible blank 2.8
    exists to end.
    """
    body = str(lesson or "").strip()
    if not body:
        return LESSON_EMPTY

    memory = load_memory()
    if "lessons_learned" not in memory: memory["lessons_learned"] = []
    existing = memory["lessons_learned"]

    if body in existing:
        return LESSON_DUPLICATE
    similar = _find_near_duplicate(body, existing)
    if similar:
        safe_print(f"📖 Skipped near-duplicate lesson (similar to: {similar[:80]})")
        return LESSON_DUPLICATE

    existing.append(body)
    retired, existing = existing[:-LESSON_CAP], existing[-LESSON_CAP:]
    memory["lessons_learned"] = existing
    save_memory(memory)
    safe_print(f"📖 Learned Lesson: {body}")
    if retired:
        safe_print(
            f"📖 At cap ({LESSON_CAP}) — retired to make room: "
            + "; ".join(r[:80] for r in retired)
        )
        return LESSON_EVICTED
    return LESSON_ADDED

@log_exceptions()
def update_lesson(index: int, new_text: str) -> bool:
    """Update an existing lesson by its index."""
    memory = load_memory()
    if "lessons_learned" not in memory: return False
    if 0 <= index < len(memory["lessons_learned"]):
        memory["lessons_learned"][index] = new_text
        save_memory(memory)
        return True
    return False

@log_exceptions()
def delete_lesson(index: int) -> bool:
    """Remove a lesson by its index."""
    memory = load_memory()
    if "lessons_learned" not in memory: return False
    if 0 <= index < len(memory["lessons_learned"]):
        memory["lessons_learned"].pop(index)
        save_memory(memory)
        return True
    return False

@log_exceptions()
def delete_key_fact(index: int) -> bool:
    """Remove a key fact by its index."""
    memory = load_memory()
    if "key_facts" not in memory: return False
    if 0 <= index < len(memory["key_facts"]):
        memory["key_facts"].pop(index)
        save_memory(memory)
        return True
    return False


@log_exceptions()
def add_recommendation(
    ticker: str,
    action: str,
    reason: str,
    price_at_advice: float | None = None,
    confidence_grade: str | None = None,
    horizon: str | None = None
) -> None:
    """Record a recommendation given to the user.

    Deduped same-day: restating the same ticker+action later in the same
    conversation (a multi-turn thread often reiterates "my BUY on AAPL stands")
    updates the existing entry in place instead of appending a duplicate —
    otherwise one call gets scored N times and skews the hit-rate/calibration
    numbers toward whatever ticker was discussed at length.
    """
    memory = load_memory()
    if "past_recommendations" not in memory:
        memory["past_recommendations"] = []

    today = datetime.now().strftime("%Y-%m-%d")
    ticker_u = ticker.upper()
    action_u = action.upper()
    confidence_u = confidence_grade.upper() if confidence_grade else None

    for existing in memory["past_recommendations"]:
        if existing.get("date") == today and existing.get("ticker") == ticker_u and existing.get("action") == action_u:
            existing["price_at_advice"] = price_at_advice
            existing["confidence_grade"] = confidence_u
            existing["horizon"] = horizon
            existing["reason"] = reason
            memory["past_recommendations"] = _trim_recommendations(memory["past_recommendations"])
            save_memory(memory)
            return

    # RESTATEMENT COLLAPSE (cross-day): re-affirming an unchanged stance (same
    # ticker+action, not superseded) on a later day updates that stance in place
    # instead of appending a new row. The original date and price stay as the
    # scoring anchor — a call's forward return is measured from when it was first
    # made (the same first-surfacing rule the funnel backtest applies), and
    # restating it daily must not reset that clock. Without this, daily portfolio
    # reviews re-log the same HOLDs (~5/day observed on the live ledger), the
    # completed-entry cap then holds only ~9 days of history, and no entry ever
    # reaches the 14-day scoring horizon — the scorecard stays structurally empty.
    for existing in memory["past_recommendations"]:
        if (existing.get("ticker") == ticker_u and existing.get("action") == action_u
                and not existing.get("superseded")):
            existing["last_affirmed"] = today
            existing["confidence_grade"] = confidence_u
            existing["horizon"] = horizon
            existing["reason"] = reason
            # Trim on this path too: the common daily turn is a restatement, and
            # the healing pass inside the trim must not wait for the next
            # genuinely-new ticker call to fold pre-collapse duplicate stances.
            memory["past_recommendations"] = _trim_recommendations(memory["past_recommendations"])
            save_memory(memory)
            return

    # RECONCILIATION: a new call of the OPPOSITE bias for this ticker closes out any
    # prior still-open call — so a SELL supersedes an earlier BUY instead of the two
    # coexisting in context and the stale BUY getting re-affirmed next turn. Long bias
    # = BUY/ADD/HOLD; reduce bias = SELL/TRIM. Only opposite-bias pairs reconcile
    # (a HOLD after a BUY is consistent and is left intact).
    new_is_long = _is_long_bias_action(action_u)
    for existing in memory["past_recommendations"]:
        if existing.get("ticker") != ticker_u or existing.get("superseded"):
            continue
        if _is_long_bias_action(existing.get("action", "")) != new_is_long:
            existing["superseded"] = True
            existing["superseded_by"] = {"date": today, "action": action_u}

    # A reduce call (SELL/TRIM) also contradicts a user-pinned LONG thesis for the
    # same name. We deliberately do NOT auto-delete user-entered theses — instead we
    # stamp an exit signal that the thesis injection surfaces, so the model reconciles
    # or closes the thesis rather than defaulting to "maintain hold". Match on the base
    # symbol so a "SELL KEEL.TO" call reconciles a "KEEL" thesis.
    # Only a HELD name can be contradicted: a WATCHING thesis is an entry plan, and a
    # sell logged against a name that was never owned is not evidence the plan is stale.
    # Unverifiable held status is treated as not-contradicted — a false CONTRADICTED
    # costs the user a live thesis, a missing one only costs a nudge.
    if not new_is_long:
        base = _normalize_thesis_symbol(ticker_u)
        held = _held_base_symbols()
        for thesis in memory.get("active_theses", []):
            t_sym = _normalize_thesis_symbol(thesis.get("symbol"))
            if not t_sym or t_sym != base:
                continue
            if not _is_long_bias_action(str(thesis.get("action", "")).upper()):
                continue
            if _thesis_position_state(thesis, held) != "held":
                continue
            thesis["exit_signal"] = {"date": today, "action": action_u, "price": price_at_advice}

    memory["past_recommendations"].append({
        "date": today,
        "ticker": ticker_u,
        "action": action_u,
        "price_at_advice": price_at_advice,
        "confidence_grade": confidence_u,
        "horizon": horizon,
        "reason": reason,
        # None = the user never said. This ledger is written by extracting the
        # ADVISOR's own text, so it can only ever know what was advised — whether the
        # user acted is a fact only the user has. Left unset until they state it via
        # set_recommendation_execution(); never inferred from holdings, because a name
        # being absent from the portfolio cannot distinguish "declined" from "not filled
        # yet" from "sold again since".
        "executed": None,
        "scores": {}  # populated by scorer
    })
    memory["past_recommendations"] = _trim_recommendations(memory["past_recommendations"])
    save_memory(memory)


@log_exceptions()
def set_recommendation_execution(
    ticker: str,
    executed: bool,
    note: str | None = None,
) -> str:
    """Record whether the USER acted on prior advice for a ticker.

    Call this ONLY when the user states it themselves ("I bought AAPL", "I passed on
    TSLA", "never got filled"). Do not infer it from the portfolio: an absent name
    could be declined, unfilled, or bought and sold again, and guessing produces
    exactly the fabricated history this field exists to prevent.

    Stamps every non-superseded row for the ticker, so a stance restated across days
    (which collapses into one row) stays consistent. Returns a human-readable summary.
    """
    ticker_u = str(ticker or "").strip().upper()
    if not ticker_u:
        return "No ticker provided — nothing recorded."

    memory = load_memory()
    rows = [
        r for r in memory.get("past_recommendations", [])
        if r.get("ticker") == ticker_u and not r.get("superseded")
    ]
    if not rows:
        # Do NOT invent a row here. A report about advice we have no record of giving
        # is a discrepancy worth surfacing, not a ledger entry worth manufacturing.
        return (
            f"No open recommendation on record for {ticker_u}, so nothing was updated. "
            f"The ledger has no call to attach this to — say so plainly rather than "
            f"implying one existed."
        )

    stamped_at = datetime.now().strftime("%Y-%m-%d")
    for row in rows:
        row["executed"] = bool(executed)
        row["execution_reported_at"] = stamped_at
        if note:
            row["execution_note"] = str(note).strip()

    save_memory(memory)
    verb = "EXECUTED" if executed else "NOT executed (user declined)"
    return f"Recorded: {ticker_u} — {verb}. Updated {len(rows)} ledger row(s)."


_REC_COMPLETED_KEEP = 50
# In-flight retention bound: the longest scoring horizon (90d) plus slack for the
# daily scorer to catch up. Entries older than this are evictable even when
# unscored, so an unpriceable ticker can't pin the ledger forever.
_REC_RETENTION_MAX_DAYS = 120


def _rec_is_in_flight(rec: dict[str, Any], now: datetime) -> bool:
    """True while a recommendation still has an unscored horizon ahead of it."""
    if rec.get("superseded"):
        return False
    scores = rec.get("scores") or {}
    if all(k in scores for k in ("2w", "1m", "3m")):
        return False
    try:
        age = (now - datetime.strptime(rec.get("date", ""), "%Y-%m-%d")).days
    except Exception:
        return False
    return age <= _REC_RETENTION_MAX_DAYS


def _collapse_restatements(recs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One-pass healing of ledgers written before the cross-day collapse in
    add_recommendation existed: consecutive rows for the same open stance
    (same ticker+action, neither superseded) are folded into the FIRST row —
    it keeps its date/price as the scoring anchor, absorbs the newest
    reason/confidence/horizon, and records the latest restatement date as
    ``last_affirmed``. Superseded rows pass through untouched (they are closed
    history, and a re-entry after a reversal is a genuinely new stance)."""
    anchors: dict[tuple, dict[str, Any]] = {}
    out: list[dict[str, Any]] = []
    for rec in recs:
        key = (rec.get("ticker"), rec.get("action"))
        if rec.get("superseded"):
            # A closed stance also ends the open anchor for this key: whatever
            # follows is a fresh call, not a restatement of the closed one.
            anchors.pop(key, None)
            out.append(rec)
            continue
        anchor = anchors.get(key)
        if anchor is None:
            anchors[key] = rec
            out.append(rec)
            continue
        anchor["last_affirmed"] = max(
            rec.get("last_affirmed") or rec.get("date") or "",
            anchor.get("last_affirmed") or anchor.get("date") or "",
        )
        for field in ("reason", "confidence_grade", "horizon"):
            if rec.get(field) is not None:
                anchor[field] = rec[field]
    return out


def _trim_recommendations(recs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Bound the ledger without evicting entries the scorer hasn't finished.

    The old flat ``[-50:]`` cap evicted purely by recency: at ~5 calls/day the
    ledger held ~9 days of history, so no entry ever survived to the 14-day
    scoring horizon and the scorecard could never accrue a single score. Keep
    every in-flight entry (still young enough to be scored); cap only the
    completed / superseded / expired remainder at the most recent
    ``_REC_COMPLETED_KEEP``.
    """
    recs = _collapse_restatements(recs)
    now = datetime.now()
    in_flight = {id(r) for r in recs if _rec_is_in_flight(r, now)}
    done_keep = {id(r) for r in [r for r in recs if id(r) not in in_flight][-_REC_COMPLETED_KEEP:]}
    return [r for r in recs if id(r) in in_flight or id(r) in done_keep]

@log_exceptions()
def add_active_thesis(thesis: dict[str, Any]) -> None:
    """Record a new active investment thesis."""
    memory = load_memory()
    if "active_theses" not in memory: memory["active_theses"] = []

    # Generate simple ID if missing
    if not thesis.get("id"):
        import uuid
        thesis["id"] = str(uuid.uuid4())[:8]

    # Add created timestamp
    thesis["created_at"] = datetime.now().isoformat()

    memory["active_theses"].append(thesis)
    save_memory(memory)
    safe_print(f"📌 Pinned Thesis: {thesis.get('symbol')} ({thesis.get('action')})")


@log_exceptions()
def delete_active_thesis(thesis_id: str) -> bool:
    """Remove an active thesis by ID."""
    memory = load_memory()
    if "active_theses" not in memory: return False

    original_count = len(memory["active_theses"])
    memory["active_theses"] = [t for t in memory["active_theses"] if t.get("id") != thesis_id]

    if len(memory["active_theses"]) < original_count:
        save_memory(memory)
        return True
    return False


@log_exceptions()
def update_active_thesis(thesis_id: str, updates: dict[str, Any]) -> bool:
    """Update an existing active thesis with new field values."""
    memory = load_memory()
    if "active_theses" not in memory: return False

    for i, thesis in enumerate(memory["active_theses"]):
        if thesis.get("id") == thesis_id:
            # Update fields
            memory["active_theses"][i].update(updates)
            memory["active_theses"][i]["updated_at"] = datetime.now().isoformat()
            # A user edit IS the reconciliation act — clear any pending exit_signal so
            # the CONTRADICTED flag doesn't keep firing after they've addressed it
            # (unless this very update re-sets one).
            if "exit_signal" not in updates:
                memory["active_theses"][i].pop("exit_signal", None)
            save_memory(memory)
            return True
    return False


@log_exceptions()
def get_active_theses() -> list[dict[str, Any]]:
    """Retrieve all active theses."""
    memory = load_memory()
    return memory.get("active_theses", [])



# Import DSPy and Signatures
from agent.dspy_setup import dspy
from agent.signatures import ActiveThesisExtraction, ContextExtraction


def _parse_llm_structured_value(value: Any, expected_type):
    """Parse JSON-ish DSPy string outputs without trusting eval."""
    if isinstance(value, expected_type):
        return value
    if value is None:
        return expected_type()

    text = str(value).strip()
    if not text or text in {"{}", "[]", "None", "null"}:
        return expected_type()

    for parser in (json.loads, __import__("ast").literal_eval):
        try:
            parsed = parser(text)
            if isinstance(parsed, expected_type):
                return parsed
        except Exception:
            continue

    return expected_type()


@log_exceptions()
def extract_thesis_from_text(text: str) -> dict[str, Any]:
    """
    Use LLM to extract a structured investment thesis from conversation text.
    """
    try:
        extractor = dspy.ChainOfThought(ActiveThesisExtraction)
        result = extractor(conversation_context=text)

        return {
            "symbol": result.symbol,
            "action": result.action,
            "quantity": getattr(result, "quantity", "N/A"),
            "catalyst": result.catalyst,
            "catalyst_date": result.catalyst_date,
            "stop_loss": result.stop_loss,
            "conditions": result.conditions,
            "expiry_date": result.expiry_date,
            "target_price": getattr(result, "target_price", "N/A"),
            "notes": result.notes
        }
    except Exception as e:
        safe_print(f"⚠️ Thesis Extraction failed: {e}")
        return {}


@log_exceptions()
def extract_context_with_llm(text: str) -> tuple[dict[str, Any], list[str], list[list[str]]]:
    """
    Use LLM to extract profile updates and important facts.

    Runs on the FAST slot. This is mechanical extraction — pull a couple of JSON
    fields out of one short message — and it fires on every non-ghost turn, so on
    the shared global LM it was being billed at deep-tier rates for work the deep
    tier does no better. `fast_dspy_context()` falls through to the global LM when
    no separate fast model is configured.
    """
    from agent.dspy_setup import fast_dspy_context

    try:
        with fast_dspy_context():
            extractor = dspy.ChainOfThought(ContextExtraction)
            result = extractor(user_message=text)

        # Parse JSON profile updates
        profile_updates = _parse_llm_structured_value(result.profile_updates, dict)

        # Parse facts list
        facts = _parse_llm_structured_value(result.new_facts, list)
        facts = [str(fact).strip() for fact in facts if str(fact).strip()]
        relationships = _parse_llm_structured_value(result.new_relationships, list)

        return profile_updates, facts, relationships
    except Exception as e:
        safe_print(f"⚠️ LLM Extraction failed: {e}")
        return {}, [], []


@log_exceptions()
def process_user_message(message: str) -> None:
    """
    Process a user message to extract and save profile info and facts.
    Uses LLM for intelligent extraction.
    """
    # Use LLM for extraction
    profile_updates, facts, relationships = extract_context_with_llm(message)

    if profile_updates:
        update_profile(profile_updates)
        safe_print(f"💾 Updated user profile: {list(profile_updates.keys())}")

    for fact in facts:
        if isinstance(fact, str) and len(fact) >= 5:
            add_fact(fact)
            safe_print(f"💾 Remembered: {fact[:120]}")

    # Update Graph Memory
    # Guard 1: Only add relationships involving 'User' or known portfolio entities.
    # Guard 2: Only allow specific relationship types (allowlist).
    # Guard 3: Temporal relations auto-expire after 30 days.
    ALLOWED_RELATIONS = {
        "INTERESTED_IN", "MONITORS", "TRACKING",
        "WORKS_AT", "LIVES_IN", "HAS_ACCOUNT", "HAS_ACCOUNT_AT",
        "HAS_RISK_TOLERANCE", "PREFERS",
    }
    # Relations that are ephemeral and should auto-expire
    TEMPORAL_RELATIONS = {"MONITORS", "INTERESTED_IN", "TRACKING"}

    if relationships:
        # Build a set of known entities (portfolio symbols + User)
        known_entities = {"User"}
        try:
            for node, data in graph_memory.graph.nodes(data=True):
                if data.get("owned") or data.get("type") == "Sector":
                    known_entities.add(node)
        except Exception:
            pass

        count = 0
        for rel in relationships:
            if isinstance(rel, list) and len(rel) == 3:
                s, r, t = rel
                r_upper = r.strip().upper()

                # Guard 2: Allowlist check
                if r_upper not in ALLOWED_RELATIONS:
                    safe_print(f"🚫 Skipped graph relation: {s} --{r}--> {t} (relation type '{r_upper}' not in allowlist)")
                    continue

                # Guard 1: Known entity check
                if s in known_entities or t in known_entities:
                    # Guard 3: Set auto-expiry for temporal relations
                    stale_days = 30 if r_upper in TEMPORAL_RELATIONS else None
                    graph_memory.add_relationship(s, t, r, stale_after_days=stale_days)
                    count += 1
                else:
                    safe_print(f"🚫 Skipped graph relation: {s} --{r}--> {t} (neither side is a known portfolio entity)")
        if count > 0:
            safe_print(f"🕸️ Added {count} relationships to Knowledge Graph")

@log_exceptions()
def get_greeting() -> str:
    """Get a personalized greeting based on memory."""
    memory = load_memory()
    name = memory.get("user_profile", {}).get("name")

    if name:
        return f"Welcome back, {name}! 👋"
    else:
        return "Welcome to CairnIQ. I'll remember our conversations to keep your research context sharper over time."


# --------------------------------------------------------------------------
# Partial-hold scoring (Roadmap 1.8) — grade a call BEFORE supersession retires it
# --------------------------------------------------------------------------
#
# A call closed by a later opposite-bias call never reaches 2w/1m/3m, so it used
# to leave the ledger unscored: measured 2026-07-26, 13 superseded recs had
# scored ZERO times and 6 of the 8 mature ACTIONABLE calls were among them. The
# fix grades the leg over the horizon it was ACTUALLY held (anchor date →
# supersession date) and files the result under its own key.
#
# `held` is NEVER pooled with 2w/1m/3m: a partial hold is not a full-horizon
# result, and mixing populations is how a statistic quietly becomes a lie (cf.
# the 4.3b zero-capital leg that moved the pooled hit rate). Every consumer must
# opt in deliberately and read `held_days` alongside the alpha.
PARTIAL_SCORE_KEY = "held"

# Below this, a "call" is restatement flap or a same-week reversal, not a
# position. Grading it adds a near-zero-information row to a small-n population,
# which is exactly the pollution the population lesson warns about — so it is
# withheld rather than scored.
_PARTIAL_MIN_HOLD_DAYS = 3

# How far past the exit date we look for a CONFIRMING bar (see below), and how
# far past the anchor date the first real print may sit before the anchor price
# stops being an honest entry.
_PARTIAL_CONFIRM_PAD_DAYS = 7
_PARTIAL_ANCHOR_TOLERANCE_DAYS = 7


def _realized_return_between(symbol: str, start: datetime, end: datetime) -> float | None:
    """Realized % return from the first real close on/after `start` to the last
    real close on/before `end`. Returns None whenever the window cannot be priced
    from actual bars — the price is never estimated, interpolated or carried.

    Freshness (5.8): a bar strictly AFTER `end` must exist in the fetched window.
    That confirming bar is what distinguishes "Friday's close is genuinely the
    last print before a Saturday exit" from "the series just stops here because
    the exit is today and the tape has not printed yet". Without it, the newest
    available bar is a STALE stand-in for the exit price, and using it would
    consume the scoring event with a wrong number — the same failure mode as
    advancing a crossing on a stale quote. Withhold instead; the daily scorer
    retries tomorrow, when the confirming bar exists.
    """
    try:
        import yfinance as yf

        fetch_end = end + timedelta(days=_PARTIAL_CONFIRM_PAD_DAYS)
        hist = yf.Ticker(symbol).history(
            start=start.date().isoformat(), end=fetch_end.date().isoformat()
        )
        if hist is None or hist.empty or "Close" not in hist:
            return None
        close = hist["Close"].dropna()
        if close.empty:
            return None

        dates = [d.date() if hasattr(d, "date") else d for d in close.index]
        start_d, end_d = start.date(), end.date()

        entry_i = next((i for i, d in enumerate(dates) if d >= start_d), None)
        if entry_i is None:
            return None
        # A first print a fortnight after the call is not the price that call was
        # made at (halted / illiquid / wrong symbol). Refuse rather than anchor on it.
        if (dates[entry_i] - start_d).days > _PARTIAL_ANCHOR_TOLERANCE_DAYS:
            return None

        exit_i = None
        for i, d in enumerate(dates):
            if d <= end_d:
                exit_i = i
            else:
                break  # index is ascending
        if exit_i is None or exit_i <= entry_i:
            return None
        if not any(d > end_d for d in dates):
            return None  # no confirming bar — see docstring

        entry = float(close.iloc[entry_i])
        exit_px = float(close.iloc[exit_i])
        if not math.isfinite(entry) or not math.isfinite(exit_px) or entry <= 0:
            return None
        pct = ((exit_px - entry) / entry) * 100.0
        # Producer-side NaN/inf guard: one non-finite number written to the
        # DURABLE ledger outlives the request that produced it.
        return pct if math.isfinite(pct) else None
    except Exception:
        return None


def _partial_hold_window(rec: dict[str, Any]) -> tuple[datetime, datetime] | None:
    """(anchor, exit) for a superseded leg, or None when the row cannot SAY when
    the call was closed.

    The exit date is read only from `superseded_by.date` — the stamp written by
    the reconciliation path in add_recommendation. It is never inferred from
    `last_affirmed` or from "now": a guessed exit date silently invents the
    holding period the score is supposed to report.
    """
    try:
        anchor = datetime.strptime(str(rec.get("date")), "%Y-%m-%d")
    except Exception:
        return None
    closed_on = (rec.get("superseded_by") or {}).get("date")
    if not closed_on:
        return None
    try:
        exit_dt = datetime.strptime(str(closed_on), "%Y-%m-%d")
    except Exception:
        return None
    if (exit_dt - anchor).days < _PARTIAL_MIN_HOLD_DAYS:
        return None
    return anchor, exit_dt


def _score_partial_hold(rec: dict[str, Any], between) -> bool:
    """Grade one superseded leg over the horizon it was actually held.

    Idempotent: a leg that already carries a `held` score is left exactly as it
    is, so the daily scorer and any back-fill can run repeatedly over the same
    ledger without rewriting history. Returns True only when a score was written.
    """
    scores = rec.get("scores")
    if isinstance(scores, dict) and PARTIAL_SCORE_KEY in scores:
        return False

    window = _partial_hold_window(rec)
    if not window:
        return False
    anchor, exit_dt = window
    ticker = rec.get("ticker")
    if not ticker:
        return False

    perf = between(ticker, anchor, exit_dt)
    spy_perf = between("SPY", anchor, exit_dt)
    # No benchmark, no alpha. A raw return without the counterfactual it is
    # judged against is not the same measurement as the rest of this ledger.
    if perf is None or spy_perf is None:
        return False

    if not isinstance(scores, dict):
        scores = {}
        rec["scores"] = scores
    scores[PARTIAL_SCORE_KEY] = {
        "perf": round(perf, 2),
        "spy_perf": round(spy_perf, 2),
        "alpha": round(perf - spy_perf, 2),
        # The label that keeps this out of full-horizon math by construction.
        "held_days": (exit_dt - anchor).days,
        "from": anchor.strftime("%Y-%m-%d"),
        "to": exit_dt.strftime("%Y-%m-%d"),
        "partial": True,
    }
    return True


def get_partial_hold_stats(past_recs: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate the partial-hold population — calls closed by a reversal before
    their stated horizon elapsed — SEPARATELY from the 2w/1m/3m population.

    Reported on its own so a consumer can include or exclude it deliberately.
    `avg_held_days` is part of the answer, not a footnote: a 45% hit rate over a
    median 6-day hold and the same rate over 3 months are different claims.

    **Every statistic here is computed over distinct CALLS, not over rows**, with
    the row count reported beside it so nothing is hidden. Measured on the live
    ledger 2026-07-27, the first day this population existed: 9 graded rows were
    4 calls. MU SELL had been restated on five consecutive days and each
    restatement is its own row, so a row-counted hit rate credited one correct
    call five times — the "N re-runs over one window is one sample" trap, landing
    inside the fix for the corpus problem it was built to solve.

    Rows are one call when they share (ticker, action, supersession event). That
    is a fact about the ledger rather than a heuristic: rows closed by the SAME
    reversal were one stance the whole time, while a re-entry after a reversal is
    closed by a DIFFERENT event and stays its own call. It is the same
    distinction `_collapse_restatements` draws, applied to the rows it
    deliberately cannot touch — it leaves superseded rows alone because they are
    closed history, so duplicates written before that collapse existed are frozen
    in the ledger permanently and only aggregation can account for them.

    The row kept for a group is the EARLIEST: the first time a call was made is
    the point the user could have acted on it, and it carries the group's longest
    held window.
    """
    groups: dict[tuple, dict[str, Any]] = {}
    graded_rows = 0
    for rec in past_recs or []:
        if not rec.get("superseded"):
            continue
        score = (rec.get("scores") or {}).get(PARTIAL_SCORE_KEY)
        if not isinstance(score, dict) or "alpha" not in score:
            continue
        graded_rows += 1
        # The supersession event is what makes restatements one call. A row with
        # no exit stamp cannot be grouped, so it stands alone under a unique key
        # rather than being merged into an unrelated group by a shared None.
        closed_on = (rec.get("superseded_by") or {}).get("date")
        key = (
            rec.get("ticker"),
            str(rec.get("action", "")).upper(),
            closed_on if closed_on else f"__ungrouped_{id(rec)}",
        )
        prior = groups.get(key)
        if prior is None or str(rec.get("date") or "") < str(prior["rec"].get("date") or ""):
            groups[key] = {"rec": rec, "score": score}

    total = hits = 0
    total_alpha = 0.0
    total_days = 0
    actionable = 0
    for group in groups.values():
        rec, score = group["rec"], group["score"]
        alpha = score["alpha"]
        long_bias = _is_long_bias_action(rec.get("action", ""))
        is_hit = (alpha > 0) if long_bias else (alpha < 0)
        total += 1
        hits += int(is_hit)
        total_alpha += alpha if long_bias else -alpha
        total_days += int(score.get("held_days") or 0)
        if str(rec.get("action", "")).upper() != "HOLD":
            actionable += 1

    if not total:
        return {
            "total": 0, "hits": 0, "hit_rate": 0.0,
            "avg_alpha": 0.0, "avg_held_days": 0.0, "actionable": 0,
            "graded_rows": graded_rows,
        }
    return {
        # Distinct calls. This is the only n here a gate may count.
        "total": total,
        "hits": hits,
        "hit_rate": round((hits / total) * 100.0, 1),
        "avg_alpha": round(total_alpha / total, 2),
        "avg_held_days": round(total_days / total, 1),
        # The subset 3.8 gates on: a graded SELL/TRIM/BUY/ADD, not a graded HOLD.
        "actionable": actionable,
        # Reported, never an n. graded_rows > total means restatements were
        # folded; a reader shown only one of these two numbers is being told the
        # sample is larger than it is.
        "graded_rows": graded_rows,
    }


def count_graded_legs(memory: dict[str, Any]) -> tuple[int, int]:
    """(full-horizon legs, partial-hold legs) currently carrying a score.

    Read off the ledger a consumer would read, not reported by the scorer about
    itself — a producer's self-report is not evidence its output landed (the
    dark-store lesson). The scheduler diffs this across a pass to say how many
    legs it actually graded.
    """
    full = partial = 0
    for rec in memory.get("past_recommendations") or []:
        scores = rec.get("scores") or {}
        if not isinstance(scores, dict):
            continue
        full += sum(1 for key in ("2w", "1m", "3m") if key in scores)
        partial += 1 if PARTIAL_SCORE_KEY in scores else 0
    return full, partial


def score_past_recommendations(memory: dict[str, Any]) -> bool:
    """Evaluate returns at 2w, 1m, and 3m horizons lazily. Returns True if memory was updated.

    Superseded legs are graded too, over the horizon actually held, under the
    separate `held` key (Roadmap 1.8 — see PARTIAL_SCORE_KEY).
    """
    from tools.funnel_backtest import _forward_return
    updated = False

    past_recs = memory.get("past_recommendations", [])
    if not past_recs:
        return False

    # Memoize per (symbol, date, horizon): SPY repeats for every rec sharing an
    # advice date, and several recs often share one — without this the daily
    # scorer refetches the same SPY window once per rec per horizon.
    _fwd_cache: dict[tuple, float | None] = {}

    def _fwd(symbol: str, start: datetime, days: int) -> float | None:
        key = (symbol, start, days)
        if key not in _fwd_cache:
            _fwd_cache[key] = _forward_return(symbol, start, days)
        return _fwd_cache[key]

    # Same memoization for the partial-hold windows: several legs closed by one
    # reversal share an exit date, and every one of them prices SPY.
    _between_cache: dict[tuple, float | None] = {}

    def _between(symbol: str, start: datetime, end: datetime) -> float | None:
        key = (symbol, start, end)
        if key not in _between_cache:
            _between_cache[key] = _realized_return_between(symbol, start, end)
        return _between_cache[key]

    now = datetime.now()

    for rec in past_recs:
        # A call reversed/closed by a later opposite-bias call is still NOT graded
        # on its full forward horizon — it was exited early, and pretending
        # otherwise would distort the hit-rate and the calibration block injected
        # into every prompt. It is graded on the horizon it was actually held,
        # under its own key, so the corpus accrues instead of recycling (1.8).
        if rec.get("superseded"):
            if _score_partial_hold(rec, _between):
                updated = True
            continue
        if "scores" not in rec:
            rec["scores"] = {}
            updated = True

        try:
            advice_date = datetime.strptime(rec["date"], "%Y-%m-%d")
        except Exception:
            continue

        days_elapsed = (now - advice_date).days

        # Horizons map: (horizon_key, days_target)
        horizons = [("2w", 14), ("1m", 30), ("3m", 90)]

        for key, days in horizons:
            if key not in rec["scores"] and days_elapsed >= days:
                ticker = rec.get("ticker")
                if not ticker:
                    continue
                # Calculate ticker return
                perf = _fwd(ticker, advice_date, days)
                # Calculate SPY benchmark return
                spy_perf = _fwd("SPY", advice_date, days)

                if perf is not None and spy_perf is not None:
                    rec["scores"][key] = {
                        "perf": round(perf, 2),
                        "spy_perf": round(spy_perf, 2),
                        "alpha": round(perf - spy_perf, 2)
                    }
                    updated = True

    return updated


@log_exceptions()
def get_advisor_scorecard() -> str:
    """
    Computes performance statistics of past recommendations and returns a formatted Markdown report.
    This runs scoring lazily to ensure fresh data.
    """
    memory = load_memory()

    # Run lazy scoring
    if score_past_recommendations(memory):
        save_memory(memory)

    past_recs = memory.get("past_recommendations", [])
    if not past_recs:
        return (
            "### 📊 Advisor Performance Scorecard\n\n"
            "No past recommendations have been logged yet. "
            "Advice is auto-logged at the end of chat sessions when specific BUY/SELL calls are made."
        )

    # Compute statistics
    stats = {
        "2w": {"total": 0, "hits": 0, "total_alpha": 0.0},
        "1m": {"total": 0, "hits": 0, "total_alpha": 0.0},
        "3m": {"total": 0, "hits": 0, "total_alpha": 0.0}
    }

    confidence_stats = {
        "HIGH": {"total": 0, "hits": 0},
        "MEDIUM": {"total": 0, "hits": 0},
        "LOW": {"total": 0, "hits": 0}
    }

    recs_table = []

    for rec in past_recs:
        ticker = rec.get("ticker")
        action = rec.get("action")
        date_str = rec.get("date")
        price = rec.get("price_at_advice")
        conf = rec.get("confidence_grade", "MEDIUM")
        scores = rec.get("scores", {})
        superseded = bool(rec.get("superseded"))

        # Build display scores
        score_displays = []
        if superseded:
            # Reversed/closed calls are excluded from the horizon table and the
            # confidence calibration below — they were exited early, so they are
            # not a 2w/1m/3m result. They are shown (and aggregated) as their own
            # partial-hold population instead.
            held = (scores or {}).get(PARTIAL_SCORE_KEY)
            if isinstance(held, dict) and "alpha" in held:
                held_hit = (
                    (held["alpha"] > 0) if _is_long_bias_action(action) else (held["alpha"] < 0)
                )
                score_displays.append(
                    f"held {held.get('held_days')}d: {held['perf']:+.1f}% "
                    f"({held['alpha']:+.1f}% vs SPY) {'🟢' if held_hit else '🔴'}"
                )
            else:
                score_displays.append("held: unscored")
        for key in ["2w", "1m", "3m"]:
            if superseded:
                continue
            if key in scores:
                alpha = scores[key]["alpha"]
                perf = scores[key]["perf"]
                long_bias = _is_long_bias_action(action)
                is_hit = (alpha > 0) if long_bias else (alpha < 0)

                stats[key]["total"] += 1
                if is_hit:
                    stats[key]["hits"] += 1

                rec_alpha = alpha if long_bias else -alpha
                stats[key]["total_alpha"] += rec_alpha

                # Update confidence breakdown
                if conf in confidence_stats:
                    confidence_stats[conf]["total"] += 1
                    if is_hit:
                        confidence_stats[conf]["hits"] += 1

                emoji = "🟢" if is_hit else "🔴"
                score_displays.append(f"{key}: {perf:+.1f}% ({alpha:+.1f}% vs SPY) {emoji}")
            else:
                score_displays.append(f"{key}: pending")

        price_display = f"${price:.2f}" if price else "N/A"
        ticker_display = f"{ticker} (closed)" if superseded else ticker
        recs_table.append(
            f"| {date_str} | {ticker_display} | {action} | {price_display} | {conf} | {', '.join(score_displays)} |"
        )

    # Format scorecard MD
    md = [
        "### 📊 Advisor Performance Scorecard",
        "Tracks the realized return of actionable advice vs the S&P 500 (SPY) benchmark.",
        ""
    ]

    # Summary Table
    md.extend([
        "#### 📈 Summary by Horizon",
        "| Horizon | Scored Calls | Hit Rate (vs SPY) | Average Alpha |",
        "|---|---|---|---|"
    ])

    for key in ["2w", "1m", "3m"]:
        total = stats[key]["total"]
        if total > 0:
            hit_rate = (stats[key]["hits"] / total) * 100.0
            avg_alpha = stats[key]["total_alpha"] / total
            md.append(f"| {key} | {total} | {hit_rate:.1f}% | {avg_alpha:+.2f}% |")
        else:
            md.append(f"| {key} | 0 | N/A | N/A |")

    md.append("")

    # Confidence breakdown
    md.extend([
        "#### 🎯 Accuracy by Stated Confidence",
        "| Confidence | Scored Calls | Hit Rate (vs SPY) |",
        "|---|---|---|"
    ])
    for conf in ["HIGH", "MEDIUM", "LOW"]:
        total = confidence_stats[conf]["total"]
        if total > 0:
            hit_rate = (confidence_stats[conf]["hits"] / total) * 100.0
            md.append(f"| {conf} | {total} | {hit_rate:.1f}% |")
        else:
            md.append(f"| {conf} | 0 | N/A |")

    # Partial holds — reported beside the horizon table, never inside it.
    partial = get_partial_hold_stats(past_recs)
    if partial["total"]:
        md.extend([
            "",
            "#### ⏹️ Closed Early (partial holds — NOT included above)",
            "| Distinct Calls | Actionable | Avg Hold | Hit Rate (vs SPY) | Average Alpha |",
            "|---|---|---|---|---|",
            f"| {partial['total']} | {partial['actionable']} | {partial['avg_held_days']:.1f}d | "
            f"{partial['hit_rate']:.1f}% | {partial['avg_alpha']:+.2f}% |",
            "",
            "*Calls reversed by a later opposite call before their horizon elapsed, "
            "graded over the period actually held. A partial hold is not a full-horizon "
            "result — these are kept out of the tables above on purpose, and the average "
            "hold length is part of reading them.*",
        ])
        if partial.get("graded_rows", partial["total"]) > partial["total"]:
            # Say it where the number is read. A restated call is one decision,
            # and a reader who sees only the row count is being told the sample
            # is larger than it is.
            md.append(
                f"*Counted as **{partial['total']} distinct calls** from "
                f"{partial['graded_rows']} graded ledger rows — a call restated on "
                f"several days and closed by one reversal is one decision, not several.*"
            )

    md.extend([
        "",
        "#### 📝 Log of Actionable Recommendations",
        "| Date | Ticker | Action | Price | Conf | Performance |",
        "|---|---|---|---|---|---|",
    ])
    md.extend(reversed(recs_table[-15:])) # Show last 15 recommendations, newest first

    return "\n".join(md)


def get_scored_recommendations_data() -> dict[str, Any]:
    """Retrieve and score past recommendations, aggregating metrics as a dictionary."""
    memory = load_memory()

    # Run lazy scoring
    if score_past_recommendations(memory):
        save_memory(memory)

    past_recs = memory.get("past_recommendations", [])

    # Compute statistics
    stats = {
        "2w": {"total": 0, "hits": 0, "total_alpha": 0.0},
        "1m": {"total": 0, "hits": 0, "total_alpha": 0.0},
        "3m": {"total": 0, "hits": 0, "total_alpha": 0.0}
    }

    confidence_stats = {
        "HIGH": {"total": 0, "hits": 0},
        "MEDIUM": {"total": 0, "hits": 0},
        "LOW": {"total": 0, "hits": 0}
    }

    for rec in past_recs:
        action = rec.get("action")
        conf = rec.get("confidence_grade", "MEDIUM")
        scores = rec.get("scores", {})
        superseded = bool(rec.get("superseded"))

        if superseded:
            continue

        for key in ["2w", "1m", "3m"]:
            if key in scores:
                alpha = scores[key]["alpha"]
                long_bias = _is_long_bias_action(action)
                is_hit = (alpha > 0) if long_bias else (alpha < 0)

                stats[key]["total"] += 1
                if is_hit:
                    stats[key]["hits"] += 1

                rec_alpha = alpha if long_bias else -alpha
                stats[key]["total_alpha"] += rec_alpha

                if conf in confidence_stats:
                    confidence_stats[conf]["total"] += 1
                    if is_hit:
                        confidence_stats[conf]["hits"] += 1

    # Format the stats for returning
    formatted_stats = {}
    for key in ["2w", "1m", "3m"]:
        total = stats[key]["total"]
        if total > 0:
            hit_rate = (stats[key]["hits"] / total) * 100.0
            avg_alpha = stats[key]["total_alpha"] / total
            formatted_stats[key] = {
                "total": total,
                "hits": stats[key]["hits"],
                "hit_rate": round(hit_rate, 1),
                "avg_alpha": round(avg_alpha, 2)
            }
        else:
            formatted_stats[key] = {
                "total": 0,
                "hits": 0,
                "hit_rate": 0.0,
                "avg_alpha": 0.0
            }

    formatted_confidence = {}
    for conf in ["HIGH", "MEDIUM", "LOW"]:
        total = confidence_stats[conf]["total"]
        if total > 0:
            hit_rate = (confidence_stats[conf]["hits"] / total) * 100.0
            formatted_confidence[conf] = {
                "total": total,
                "hits": confidence_stats[conf]["hits"],
                "hit_rate": round(hit_rate, 1)
            }
        else:
            formatted_confidence[conf] = {
                "total": 0,
                "hits": 0,
                "hit_rate": 0.0
            }

    return {
        "stats": formatted_stats,
        "confidence_stats": formatted_confidence,
        # Deliberately its own block, never folded into `stats`: these legs were
        # closed by a reversal before their horizon elapsed (1.8).
        "partial_stats": get_partial_hold_stats(past_recs),
        "recommendations": past_recs
    }


# Import re at top level


if __name__ == "__main__":
    # Test the memory system
    print("=== Testing Memory System ===\n")

    # Test extraction
    test_msg = "I'm 45 years old, make $150,000 a year, and want to retire at 60. I have a TFSA and RRSP. I'm bullish on AI and worried about inflation."
    print(f"Test message: {test_msg}\n")

    profile_updates, facts, relationships = extract_context_with_llm(test_msg)
    print(f"Extracted profile updates: {profile_updates}\n")
    print(f"Extracted facts: {facts}\n")
    print(f"Extracted relationships: {relationships}\n")

    # Test context generation
    print("=== Memory Context ===")
    print(get_user_context())
