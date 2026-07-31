"""
Opportunity Scanner Tool — Funnel V2 (Accumulation-First, Top-Down)
==================================================================
Inverts the old bottom-up value screen: detect where capital is *beginning*
to move, resolve the driving theme/sector, then locate the best-positioned
vehicle and gate the entry on cycle stage. Spec: docs/technical/OPPORTUNITY_FUNNEL_V2_SPEC.md

  Stage 0: Dynamic Universe Assembly (movers ∪ rotating-sector members ∪
           theme producers ∪ guru ∪ RS leaders → deduped, hard-capped ≤250)
  Stage 1: Event & Flow Radar (rank themes/sectors by inflow momentum,
           macro alignment, active catalysts; per-candidate relative strength)
  Stage 2: Theme → Sector + Cycle Stage (driver narrative; early/mid/late;
           keep candidates in top themes only → narrow to ~60)
  Stage 3: Stock Selection Within Sector (relative strength + forward
           fundamentals → additive base conviction → narrow to ~15-20)
  Stage 4: Flow Confirm + Entry-Quality Gate (dark-pool/whale/insider/setup
           → capped confirmation bonus; entry_stage multiplier)
  Stage 5: Risk & Portfolio-Fit Overlay (concentration/short/event risk as
           surfaced flags + capped modifier → final ranked top_picks)

Accumulation/proxy flow signals are confirmation-only (capped), not leading
pillars. Performance target: full "All sectors" broad scan under the 150s
deadline (pipeline is deadline-aware and aborts cooperatively at boundaries).
"""
import json
import os
import re
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from concurrent.futures import TimeoutError as FuturesTimeoutError
from datetime import UTC, date, datetime, timedelta
from typing import Any

import numpy as np
import yfinance as yf

from agent.utils import is_cancelled, safe_print
from tools.cache import cached
from tools.exception_logger import log_exceptions
from tools.json_store import write_json_atomic
from tools.user_profile import get_active_profile, run_under_profile

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------
_FUNNEL_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "user_data", "funnel_config.json")
_FUNNEL_CONFIG_EXAMPLE_PATH = os.path.join(os.path.dirname(__file__), "..", "funnel_config.example.json")
_SIGNAL_LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "user_data", "funnel_signal_log")
_SCAN_LEDGER_PATH = os.path.join(os.path.dirname(__file__), "..", "user_data", "funnel_scan_ledger.json")
_universe_cache = {}


def _load_scan_ledger() -> dict[str, str]:
    """{symbol: last-deep-scanned ISO date} — powers the exploration slots."""
    try:
        with open(_SCAN_LEDGER_PATH) as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _exploration_candidates(scored: list, fast_screen_top_n: int, slots: int,
                            ledger: dict[str, str]) -> list:
    """
    Pick ``slots`` names from below the fast-screen cut, least-recently
    deep-scanned first. Never-scanned names (absent from the ledger — empty
    string sorts before any ISO date) come first; fast-score desc breaks ties
    within the same date bucket, symbol as a deterministic final tiebreak.
    """
    pool = list(scored[fast_screen_top_n:])
    pool.sort(key=lambda x: (ledger.get(x[0].upper(), ""), -x[1], x[0]))
    return pool[:slots]


def _update_scan_ledger(symbols: list[str]) -> None:
    """Stamp today's date on every deep-scanned symbol; prune entries >180d old."""
    try:
        from datetime import date, timedelta
        ledger = _load_scan_ledger()
        today = date.today()
        for sym in symbols:
            if sym:
                ledger[sym.upper()] = today.isoformat()
        cutoff = (today - timedelta(days=180)).isoformat()
        ledger = {s: d for s, d in ledger.items() if d >= cutoff}
        write_json_atomic(_SCAN_LEDGER_PATH, ledger, indent=None)
    except Exception as e:
        safe_print(f"⚠️ Scan-ledger update skipped (non-fatal): {e}")
_SECTOR_ALIASES = {
    "TECH": "Technology", "SEMIS": "Semiconductors", "SEMICONDUCTORS": "Semiconductors",
    "FINANCE": "Finance", "FINANCIALS": "Finance", "FINTECH": "Finance", "FINANCIAL SERVICES": "Finance",
    "HEALTHCARE": "Healthcare", "BIOTECH": "Healthcare",
    "ENERGY": "Energy", "GREEN ENERGY": "Energy",
    "CONSUMER": "Consumer Discretionary", "CONSUMER DISCRETIONARY": "Consumer Discretionary", "CONSUMER CYCLICAL": "Consumer Discretionary",
    "CONSUMER STAPLES": "Consumer Staples", "CONSUMER DEFENSIVE": "Consumer Staples",
    "INDUSTRIALS": "Industrials", "UTILITIES": "Utilities",
    "REAL ESTATE": "Real Estate", "MATERIALS": "Materials", "BASIC MATERIALS": "Materials",
    "COMMUNICATION": "Communication Services", "COMMUNICATIONS": "Communication Services", "COMMUNICATION SERVICES": "Communication Services", "COMM SERVICES": "Communication Services",
    "CYBERSECURITY": "Technology", "AI": "Technology",
}
_SCAN_TIMEOUT = 90          # Legacy/value scan timeout (seconds) – outer safety net
_V2_SCAN_TIMEOUT = 150      # Funnel V2 broad scans do more staged work
                            # NOTE: agent/nodes/market_analyst.py::_TOOL_BATCH_TIMEOUT
                            # must stay ABOVE this, or the caller abandons the batch
                            # before this budget can ever produce its result.
_PHASE5_GATE_BUDGET_S = 45  # Total wall-clock for the three Phase-5 gates combined
_PHASE5_MIN_GATE_S = 6      # Below this, skip the gates rather than half-run them
_PHASE5_RESERVE_S = 4       # Held back for rescoring + summary after the gates

# In-flight scan registry: {(profile, normalized_sector): Future}. `screen_stocks`
# and `scan_opportunities` are the SAME function behind two tool names, so a
# planner that calls both — which it does, since both are registered and both
# match "scan the market" — used to run two full identical pipelines side by
# side in one process, doubling the yfinance/network load they then blame for
# being slow (observed 2026-07-28: two 143-ticker universes, two 30s earnings
# warms, GDELT 429s). The second caller now waits on the first's result.
_INFLIGHT_SCANS: dict[tuple[str, str], Any] = {}
_INFLIGHT_LOCK = threading.Lock()
_BATCH_DOWNLOAD_PERIOD = "6mo"
_FAST_SCREEN_TOP_N = 60     # Candidates that advance to the deep-dive phase
_EXPLORATION_SLOTS = 10     # Extra deep-dive slots for least-recently-scanned names
                            # (fights universe convergence: without them the same
                            # deterministic top-60 wins the fast screen every day)
_FINAL_TOP_N = 15           # Returned to the user from "All" mode
_MIN_SCORE_THRESHOLD = 40   # Minimum score to qualify as a result
_FALLBACK_MIN_SCORE = 25    # If nothing clears the threshold, relax to this
# Funnel V2 / M1 — Dynamic Universe Assembly
_UNIVERSE_CAP = 250         # Hard cap on broad-scan candidate pool
_DOWNLOAD_CHUNK_SIZE = 75   # Max tickers per yf.download batch (prevents timeout regression)
_MOVER_LATE_THRESHOLD_PCT = 15.0   # Daily gain % above which a mover is treated as "already extended"
# Funnel V2 / M3 — Additive scoring + entry-stage governor
_M3_PILLAR_WEIGHTS = {"theme": 30.0, "relstr": 25.0, "forward": 25.0, "quality": 20.0}
_M3_FLOW_BONUS = {"one": 5.0, "two_plus": 10.0}
_M3_ENTRY_MULTIPLIERS = {
    "accumulation_base": 1.10,
    "early_breakout": 1.00,
    "mid_trend": 0.85,
    "extended": 0.40,
}
_M3_RISK_CAP = 15.0
_M3_EXTENDED_OVER_SMA50_PCT = 35.0
_DEFAULT_FUNNEL_CONFIG: dict[str, Any] = {
    "theme_weights": {"momentum": 0.45, "macro": 0.25, "catalyst": 0.20, "breadth": 0.10},
    "pillars": {"theme": 30.0, "relstr": 25.0, "forward": 25.0, "quality": 20.0},
    "flow_bonus": {"two_plus": 10.0, "one": 5.0},
    "entry_multipliers": dict(_M3_ENTRY_MULTIPLIERS),
    "tiers": {"exceptional": 100.0, "high": 80.0, "qualified": 60.0, "watchlist": 40.0},
    "risk_cap": _M3_RISK_CAP,
    "concentration": {"threshold_pct": 25.0, "penalty_per_excess_pct": 1.0},
    "top_k_themes": 4,
    "fast_screen_top_n": _FAST_SCREEN_TOP_N,
    "exploration_slots": _EXPLORATION_SLOTS,
    "final_top_n": _FINAL_TOP_N,
    "universe_cap": _UNIVERSE_CAP,
    "download_chunk": _DOWNLOAD_CHUNK_SIZE,
    "early_mover_max_5d_gain_pct": 20.0,
    "extended_over_sma50_pct": _M3_EXTENDED_OVER_SMA50_PCT,
    "timeouts": {
        "scan_seconds": _SCAN_TIMEOUT,
        "v2_scan_seconds": _V2_SCAN_TIMEOUT,
        "batch_download_seconds": 12,
        "chunked_download_seconds": 55,
    },
    "batch_download": {
        "max_retries": 2,
        "max_workers": 3,
        "gap_fill_max_tickers": 5,
    },
    "validation": {
        "enable_signal_logging": True,
        "min_matured_samples": 20,
        "default_lookback_days": 120,
        "default_horizon_days": 63,
        "promotion_min_hit_rate_pct": 50,
    },
}
_funnel_config_missing_warned = False
_last_batch_download_diagnostics: dict[str, Any] = {}
_SCORING_FUNDAMENTAL_FIELDS = [
    "revenue_growth",
    "earnings_growth",
    "profit_margin",
    "forward_pe",
    "trailing_pe",
    "peg_ratio",
    "analyst_target",
]


def _safe_float(value: Any) -> float | None:
    """Convert numeric API fields to float while preserving unavailable values."""
    if _is_missing_data_point(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _is_missing_data_point(value: Any) -> bool:
    """Return True for values that should be treated as unavailable in diagnostics."""
    if value is None:
        return True
    if isinstance(value, (float, np.floating)) and np.isnan(value):
        return True
    if isinstance(value, str) and not value.strip():
        return True
    return False


def _deep_merge_dict(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Merge nested config dictionaries without mutating the defaults."""
    merged = dict(base)
    for key, value in (overlay or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_funnel_config() -> dict[str, Any]:
    """
    Load Funnel V2 config from user_data/funnel_config.json.
    Missing/malformed config falls back to safe defaults and emits one loud warning.
    """
    global _funnel_config_missing_warned
    try:
        with open(_FUNNEL_CONFIG_PATH) as f:
            user_cfg = json.load(f)
        if not isinstance(user_cfg, dict):
            raise ValueError("config root must be an object")
        return _deep_merge_dict(_DEFAULT_FUNNEL_CONFIG, user_cfg)
    except FileNotFoundError:
        if not _funnel_config_missing_warned:
            safe_print(
                f"⚠️ Funnel config missing at {_FUNNEL_CONFIG_PATH}; using built-in defaults. "
                "Copy funnel_config.example.json into user_data/funnel_config.json to customize."
            )
            _funnel_config_missing_warned = True
    except Exception as e:
        safe_print(f"⚠️ Funnel config invalid ({e}); using built-in defaults.")
    return _deep_merge_dict(_DEFAULT_FUNNEL_CONFIG, {})


def seed_funnel_config_if_missing() -> bool:
    """Create user_data/funnel_config.json from the checked-in example on first run.

    Idempotent and fail-safe: the scanner runs from built-in defaults regardless,
    so this only gives users an editable starting file (and silences the missing-
    config warning). Called from install (install.ps1) and once at server startup
    so every platform — including manual venv installs — gets the file seeded.
    Returns True iff a new file was written.
    """
    import shutil
    try:
        if os.path.exists(_FUNNEL_CONFIG_PATH):
            return False
        if not os.path.exists(_FUNNEL_CONFIG_EXAMPLE_PATH):
            return False
        os.makedirs(os.path.dirname(_FUNNEL_CONFIG_PATH), exist_ok=True)
        shutil.copyfile(_FUNNEL_CONFIG_EXAMPLE_PATH, _FUNNEL_CONFIG_PATH)
        return True
    except Exception:
        return False


def seed_stock_universe_if_missing() -> bool:
    """Create user_data/stock_universe.json from stock_universe.example.json on first run.

    Idempotent and fail-safe: the scanner runs from built-in defaults regardless,
    so this only gives users an editable starting file.
    Returns True iff a new file was written.
    """
    import shutil
    try:
        if os.path.exists(_UNIVERSE_PATH):
            return False
        if not os.path.exists(_STOCK_UNIVERSE_EXAMPLE_PATH):
            return False
        os.makedirs(os.path.dirname(_UNIVERSE_PATH), exist_ok=True)
        shutil.copyfile(_STOCK_UNIVERSE_EXAMPLE_PATH, _UNIVERSE_PATH)
        return True
    except Exception:
        return False


def _cfg_section(name: str) -> dict[str, Any]:
    section = _load_funnel_config().get(name, {})
    return section if isinstance(section, dict) else {}


def _cfg_number(path: str, default: float) -> float:
    cfg: Any = _load_funnel_config()
    for part in path.split("."):
        if not isinstance(cfg, dict) or part not in cfg:
            return float(default)
        cfg = cfg[part]
    try:
        return float(cfg)
    except (TypeError, ValueError):
        return float(default)


def _missing_scoring_fields(fundamentals: dict[str, Any]) -> list[str]:
    """List fundamental fields required by the scoring gate that are missing."""
    return [
        field for field in _SCORING_FUNDAMENTAL_FIELDS
        if _is_missing_data_point(fundamentals.get(field))
    ]


def _assess_foundation_quality(fundamentals: dict[str, Any]) -> dict[str, Any]:
    """
    Evidence check for whether a large pullback is backed by durable fundamentals.

    This is intentionally not a universal hard gate: banks, insurers, REITs, and
    early-stage biotech do not fit the same margin/cash-flow template as software.
    The scanner uses it to avoid presenting a normal high-beta drawdown as a
    structurally de-risked bargain.
    """
    str(fundamentals.get("symbol", "")).upper()
    sector = str(fundamentals.get("sector_yf", ""))
    industry = str(fundamentals.get("industry", ""))

    # Detect if ticker is a financial institution using yfinance sector/industry data.
    # Financial companies (banks, lenders, insurers) have structurally different
    # balance sheets and cash flows — high leverage and negative operating cash flow
    # are normal for borrow-to-lend business models.
    _FINANCIAL_INDUSTRIES = {
        "Banks", "Banks—Diversified", "Banks—Regional",
        "Credit Services", "Capital Markets", "Consumer Finance",
        "Insurance", "Insurance—Diversified", "Insurance—Life",
        "Insurance—Property & Casualty", "Insurance—Specialty",
        "Financial Data & Stock Exchanges", "Asset Management",
        "Mortgage Finance", "Shell Companies",
        "Financial Conglomerates", "Savings & Cooperative Banking",
    }
    is_financial = (
        sector == "Financial Services" or
        industry in _FINANCIAL_INDUSTRIES
    )

    gross_margin = _safe_float(fundamentals.get("gross_margin"))
    profit_margin = _safe_float(fundamentals.get("profit_margin"))
    free_cashflow = _safe_float(fundamentals.get("free_cashflow"))
    operating_cashflow = _safe_float(fundamentals.get("operating_cashflow"))
    total_debt = _safe_float(fundamentals.get("total_debt"))
    total_cash = _safe_float(fundamentals.get("total_cash"))
    ebitda = _safe_float(fundamentals.get("ebitda"))

    checks = []
    missing = []

    # 1. CASH FLOW / PROFITABILITY CHECK
    if is_financial:
        # Financial institutions/lenders routinely show negative/volatile operating cash flows
        # due to loan origination flows, which are reported as GAAP cash outflows.
        # Net profitability/margins are the correct proxy for financial health.
        passed_cf = True
        val_cf = "financial_regime"
        if free_cashflow is not None and free_cashflow > 0:
            val_cf = free_cashflow
        elif operating_cashflow is not None and operating_cashflow > 0:
            val_cf = operating_cashflow

        checks.append({
            "name": "cash_flow_positive",
            "passed": passed_cf,
            "value": val_cf,
            "note": "Negative operating/free cash flow is structurally normal for lending business models; profitability prioritised."
        })
    elif free_cashflow is not None or operating_cashflow is not None:
        cashflow_value = free_cashflow if free_cashflow is not None else operating_cashflow
        checks.append({
            "name": "cash_flow_positive",
            "passed": cashflow_value is not None and cashflow_value > 0,
            "value": cashflow_value,
        })
    else:
        missing.append("cash_flow")

    # 2. MARGIN QUALITY CHECK
    if gross_margin is not None:
        checks.append({
            "name": "gross_margin_quality",
            "passed": gross_margin >= 0.40,
            "value": gross_margin,
        })
    elif profit_margin is not None:
        checks.append({
            "name": "profit_margin_quality",
            "passed": profit_margin >= 0.10,
            "value": profit_margin,
        })
    else:
        missing.append("margin_quality")

    # 3. BALANCE SHEET / LEVERAGE CHECK
    if is_financial:
        # Financial services/lenders fund operations via debt and securitizations (high leverage).
        # Corporate metrics like net debt to EBITDA do not apply. Cet1/Capital Adequacy is the standard.
        checks.append({
            "name": "balance_sheet_strength",
            "passed": True,
            "value": "financial_leverage_normal",
            "note": "High structural leverage is standard for banks and credit service providers."
        })
    elif total_debt is not None and total_cash is not None:
        if total_cash >= total_debt:
            checks.append({
                "name": "balance_sheet_strength",
                "passed": True,
                "value": "net_cash",
            })
        elif ebitda and ebitda > 0:
            net_debt_to_ebitda = (total_debt - total_cash) / ebitda
            checks.append({
                "name": "balance_sheet_strength",
                "passed": net_debt_to_ebitda <= 1.5,
                "value": round(net_debt_to_ebitda, 2),
            })
        else:
            missing.append("ebitda_for_leverage")
    else:
        missing.append("debt_cash")

    passed_count = sum(1 for check in checks if check["passed"])
    if len(checks) >= 3 and passed_count == len(checks):
        grade = "Strong"
    elif len(checks) >= 2 and passed_count >= 1:
        grade = "Mixed"
    else:
        grade = "Unproven"

    return {
        "grade": grade,
        "checks": checks,
        "missing_metrics": missing,
        "note": (
            "Strong only when cash flow, margin quality, and balance-sheet evidence all clear. "
            "Missing metrics mean the pullback should be treated as volatility, not proof of a durable discount."
        ),
    }


# ---------------------------------------------------------------------------
# 0. UNIVERSE DEFINITIONS
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# FUNNEL V2 / M1 — DYNAMIC UNIVERSE HELPERS
# ---------------------------------------------------------------------------

# Maps rotation sector names (market_mechanics / sector_rotation) → FMP screener sector param.
# Keys are matched case-insensitively and by prefix (e.g. "Financial" ↔ "Financials").
_FMP_SECTOR_MAP: dict[str, str] = {
    "Technology": "Technology",
    "Financials": "Financial Services",
    "Financial": "Financial Services",
    "Healthcare": "Healthcare",
    "Energy": "Energy",
    "Consumer Discretionary": "Consumer Cyclical",
    "Consumer Cyclical": "Consumer Cyclical",
    "Consumer Staples": "Consumer Defensive",
    "Consumer Defensive": "Consumer Defensive",
    "Industrials": "Industrials",
    "Materials": "Basic Materials",
    "Basic Materials": "Basic Materials",
    "Real Estate": "Real Estate",
    "Utilities": "Utilities",
    "Communication Services": "Communication Services",
    "Communication": "Communication Services",
    "Semiconductors": "Technology",
}

# Semiconductor / memory seed — capital-intensive cyclicals that are NOT in the
# built-in fallback universe and would otherwise be silently excluded when FMP and
# geo-themes are unavailable. This is the static guarantee that names like MU enter
# a broad scan even with no FMP key and no active memory event.
_SEMI_SEED: list[str] = [
    "MU", "WDC", "STX",                       # memory (DRAM/NAND)
    "NVDA", "AMD", "AVGO", "QCOM", "MRVL",     # design
    "TSM", "INTC", "GFS",                      # foundry/IDM
    "AMAT", "LRCX", "KLAC", "ASML", "TER",     # equipment
]

# Symbols known to be delisted/acquired — skipped at universe assembly so they never
# reach batch download or the gap-filler. Seeded with confirmed 2024 energy M&A names
# that still linger in tools/geopolitical_scanner.py's producer maps (PXD→XOM,
# SWN→EXE, TELL→acquired). The set grows at runtime as the gap-filler discovers more.
_known_delisted: set[str] = {"PXD", "SWN", "TELL"}


def _resolve_fmp_sector(sector_name: str) -> str | None:
    """Map a rotation sector name to an FMP screener sector, tolerant of name variants."""
    if not sector_name:
        return None
    if sector_name in _FMP_SECTOR_MAP:
        return _FMP_SECTOR_MAP[sector_name]
    su = sector_name.strip().lower()
    for key, val in _FMP_SECTOR_MAP.items():
        kl = key.lower()
        if kl in su or su in kl:
            return val
    return None


# Circuit breaker: the FMP company-screener is a PAID endpoint. On plans without it,
# every call returns HTTP 402/401/403. Once we see that, skip the screener for the
# rest of the process (it won't start working mid-session) — avoids one wasted
# round-trip per hot sector on every scan. Reset only on a fresh process.
_fmp_screener_available: bool = True


def _fmp_sector_constituents(sector_name: str, limit: int = 30) -> list[str]:
    """
    Fetch top constituents of a sector via the FMP company-screener.
    Uses fmp_api._fmp_get (key-rotation + structured params).
    Returns [] on any failure so callers can fall back gracefully.
    Disables itself process-wide if the endpoint is not on the FMP plan (402/401/403).
    """
    global _fmp_screener_available
    if not _fmp_screener_available:
        return []
    fmp_sector = _resolve_fmp_sector(sector_name)
    if not fmp_sector:
        return []
    try:
        from tools.fmp_api import _fmp_get as fmp_get
        data, err = fmp_get(
            "company-screener",
            params={
                "sector": fmp_sector,
                "marketCapMoreThan": 1_000_000_000,
                "country": "US",
                "isActivelyTrading": "true",
                "limit": limit,
            },
            timeout=8,
        )
        if err:
            # Plan/auth errors mean the endpoint will never work on this key → disable.
            if any(code in str(err) for code in ("402", "401", "403")):
                _fmp_screener_available = False
                safe_print(f"ℹ️ FMP company-screener unavailable on this plan ({err}) — "
                           "disabling it for this session; using static/seed universe.")
            return []
        if not isinstance(data, list):
            return []
        return [item["symbol"] for item in data if isinstance(item, dict) and item.get("symbol")]
    except Exception:
        return []


# TradingView Sector Map -> GICS Canonical Sector
_TV_SECTOR_MAP: dict[str, list[str]] = {
    "Technology": ["Electronic Technology", "Technology Services"],
    "Energy": ["Energy Minerals", "Industrial Services"],
    "Financials": ["Finance"],
    "Healthcare": ["Health Technology", "Health Services"],
    "Real Estate": ["Real Estate"],
    "Consumer Staples": ["Consumer Non-Durables", "Process Industries"],
    "Consumer Discretionary": ["Consumer Durables", "Consumer Services", "Retail Trade"],
    "Communication Services": ["Communications"],
    "Industrials": ["Producer Manufacturing", "Transportation", "Commercial Services"],
    "Utilities": ["Utilities"],
    "Materials": ["Non-Energy Minerals"],
}


def _sector_matches_tv(tv_sec: str, canon: str) -> bool:
    """Check if TradingView sector string aligns with canonical GICS sector."""
    if not tv_sec or not canon:
        return True
    tv_lower = tv_sec.lower()
    allowed = _TV_SECTOR_MAP.get(canon, [])
    if not allowed:
        return True
    for a in allowed:
        if a.lower() in tv_lower or tv_lower in a.lower():
            return True
    return False


def _tv_sector_constituents(sector_name: str, limit: int = 30) -> list[str]:
    """
    Fetch live liquid sector constituents from TradingView Screener API (keyless).
    Covers both US (NYSE/NASDAQ) and Canada (TSX). Daily-cached per sector.
    """
    import tools.daily_cache as daily_cache
    canon = _canonical_sector(sector_name) or sector_name
    cache_key = f"tv_sector_constituents:{canon}"
    hit = daily_cache.get_cached(cache_key)
    if isinstance(hit, list) and hit:
        return hit[:limit]

    tickers: list[str] = []
    import requests

    for region in ("america", "canada"):
        url = f"https://scanner.tradingview.com/{region}/scan"
        payload = {
            "filter": [
                {"left": "volume", "operation": "greater", "right": 100000},
                {"left": "close", "operation": "greater", "right": 1.0},
                {"left": "market_cap_basic", "operation": "greater", "right": 150000000},
            ],
            "columns": ["name", "close", "change", "volume", "sector"],
            "sort": {"sortBy": "volume", "sortOrder": "desc"},
            "range": [0, 50],
        }
        try:
            # Use requests (certifi-backed) rather than urllib: the framework
            # Python this ships under has no OpenSSL cafile
            # (ssl.get_default_verify_paths().cafile is None), so a raw
            # urllib.urlopen fails every TLS verify with CERTIFICATE_VERIFY_FAILED.
            # requests bundles certifi and verifies correctly on the same runtime.
            resp = requests.post(
                url,
                json=payload,
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=5,
            )
            resp.raise_for_status()
            data = resp.json()
            for item in data.get("data", []):
                d = item.get("d", [])
                if not d or len(d) < 5:
                    continue
                sym = str(d[0]).strip().upper()
                sec = str(d[4] or "")
                if canon and not _sector_matches_tv(sec, canon):
                    continue
                if region == "canada" and not (sym.endswith(".TO") or sym.endswith(".V")):
                    sym = f"{sym}.TO"
                if sym and sym not in tickers:
                    tickers.append(sym)
        except Exception as e:
            safe_print(f"ℹ️ TradingView screener unavailable for {region}/{sector_name}: {e}")

    if tickers:
        daily_cache.set_cached(cache_key, tickers)
    return tickers[:limit]


# Canonical sector → Yahoo Finance sector key (yfinance Sector API).
_YAHOO_SECTOR_KEYS: dict[str, str] = {
    "Technology": "technology",
    "Financials": "financial-services",
    "Healthcare": "healthcare",
    "Energy": "energy",
    "Consumer Discretionary": "consumer-cyclical",
    "Consumer Staples": "consumer-defensive",
    "Industrials": "industrials",
    "Utilities": "utilities",
    "Real Estate": "real-estate",
    "Materials": "basic-materials",
    "Communication Services": "communication-services",
}


def _yf_sector_constituents(sector_name: str, limit: int = 30) -> list[str]:
    """
    Live sector constituents via Yahoo Finance (yfinance Sector.top_companies,
    ~50 names per sector, free). This is the fallback for the FMP
    company-screener, which is a paid endpoint not on the current plan — without
    it the universe degrades to ~43 hardcoded names across 5 sectors and the
    scan surfaces the same stocks every day.

    Daily-cached per sector; the yfinance call runs under a hard timeout because
    yfinance has no global timeout (a hung call would stall universe assembly).
    Returns [] on any failure so callers fall through to the static lists.
    """
    canon = _canonical_sector(sector_name)
    yahoo_key = _YAHOO_SECTOR_KEYS.get(canon or "")
    if not yahoo_key:
        return []

    import tools.daily_cache as daily_cache
    cache_key = f"yf_sector_constituents:{yahoo_key}"
    hit = daily_cache.get_cached(cache_key)
    if isinstance(hit, list) and hit:
        return hit[:limit]

    def _fetch() -> list[str]:
        import yfinance as yf
        df = yf.Sector(yahoo_key).top_companies
        if df is None or df.empty:
            return []
        return [str(s).upper() for s in df.index.tolist() if s]

    try:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=1) as pool:
            symbols = pool.submit(_fetch).result(timeout=10)
        if symbols:
            daily_cache.set_cached(cache_key, symbols)
        return symbols[:limit]
    except Exception as e:
        safe_print(f"ℹ️ Yahoo sector constituents unavailable for {sector_name} ({type(e).__name__}) — static fallback.")
        return []


@cached(key_func=lambda: "funnel_geo_check", ttl=3600)
@log_exceptions()
def _cached_geo_check() -> dict:
    """
    Daily-cached wrapper around quick_geopolitical_check (which hits GDELT + web).
    The funnel calls the geo check from two places per broad scan — universe
    assembly (M1) and catalyst ranking (M2) — so caching collapses those to a
    single network round-trip.
    """
    from tools.geopolitical_scanner import quick_geopolitical_check
    return quick_geopolitical_check()


def _get_active_theme_tickers() -> dict[str, list[str]]:
    """
    Resolve currently-active geo/policy themes to their producer tickers.
    Uses the cached geo check to find active commodities/themes, then resolves
    each to tickers via COMMODITY_TICKER_MAP. Returns {theme_key: [tickers]}.
    """
    theme_tickers: dict[str, list[str]] = {}
    try:
        from tools.geopolitical_scanner import (
            COMMODITY_TICKER_MAP,
            _get_tickers_for_commodity,
        )
        check = _cached_geo_check()
        if not check.get("alert"):
            return {}
        for commodity in check.get("commodities_at_risk", []):
            key = commodity.lower().replace(" ", "_")
            if key in COMMODITY_TICKER_MAP:
                tickers = _get_tickers_for_commodity(key)
                if tickers:
                    theme_tickers[key] = tickers
    except Exception:
        pass
    return theme_tickers


def _inflowing_sectors(rotation_data: dict) -> list[str]:
    """
    Extract the set of sectors money is rotating INTO, robust to both rotation
    implementations:
      - market_mechanics.detect_sector_rotation() → full_rotation_map[{sector, trend}]
        (trend strings like "Leading 🟢", "Improving 🔵")
      - sector_rotation.detect_sector_rotation()   → sector_performance[{sector, momentum_score}]
    """
    sectors: list[str] = []
    seen: set[str] = set()

    def _push(name: str) -> None:
        if name and name not in seen:
            seen.add(name)
            sectors.append(name)

    # Shape A: market_mechanics (the implementation _scan_impl actually uses)
    for item in rotation_data.get("full_rotation_map", []) or []:
        if not isinstance(item, dict):
            continue
        trend = str(item.get("trend", ""))
        if "Leading" in trend or "Improving" in trend:
            _push(item.get("sector", ""))

    # Shape B: sector_rotation (momentum_score)
    for item in rotation_data.get("sector_performance", []) or []:
        if isinstance(item, dict) and isinstance(item.get("momentum_score"), (int, float)) and item["momentum_score"] > 0:
            _push(item.get("sector", ""))

    # Fallback: explicit leading_sectors list (names or dicts)
    if not sectors:
        for item in rotation_data.get("leading_sectors", []) or []:
            if isinstance(item, str):
                _push(item)
            elif isinstance(item, dict):
                _push(item.get("sector", ""))

    return sectors


def _assemble_dynamic_universe(
    rotation_data: dict,
    intraday_movers: dict | None = None,
    guru_tickers: list[str] | None = None,
) -> tuple[list[str], dict[str, list[str]]]:
    """
    Funnel V2 Stage 0 — Dynamic Universe Assembly (M1).

    Builds a deduped, hard-capped ≤_UNIVERSE_CAP candidate pool from live
    market state rather than a static file.  Multi-source names are ranked
    first so the theme/flow signal survives the downstream fast-screen cut.

    Args:
        rotation_data   — output of detect_sector_rotation() (either implementation)
        intraday_movers — pre-fetched scan_intraday_movers() output, or None to fetch
        guru_tickers    — caller-provided guru/media overlay tickers (named Mad
                          Money scans pre-fetch these); None means no guru
                          overlay — the institutional source is now 13F diffs

    Returns:
        candidates   — ordered list[str] of ticker symbols (≤ _UNIVERSE_CAP)
        provenance   — {symbol: [source, ...]} for every candidate
    """
    from agent.logger import log_to_component

    provenance: dict[str, list[str]] = {}
    source_status: dict[str, int] = {}

    def _add(symbol: str, source: str) -> None:
        sym = symbol.upper().strip()
        if not sym:
            return
        if sym not in provenance:
            provenance[sym] = []
        if source not in provenance[sym]:
            provenance[sym].append(source)

    # ── 1. Constituents of inflowing sectors: TradingView → FMP → Yahoo → static ─
    hot_sectors = _inflowing_sectors(rotation_data)
    tv_count = 0
    fmp_count = 0
    yf_count = 0
    for sec_name in hot_sectors:
        tv_tickers = _tv_sector_constituents(sec_name, limit=30)
        if tv_tickers:
            for t in tv_tickers:
                _add(t, f"sector_tv:{sec_name}")
            tv_count += len(tv_tickers)
            continue
        tickers = _fmp_sector_constituents(sec_name, limit=30)
        if tickers:
            for t in tickers:
                _add(t, f"sector:{sec_name}")
            fmp_count += len(tickers)
            continue
        # FMP screener unavailable (paid endpoint) — live Yahoo constituents.
        yf_tickers = _yf_sector_constituents(sec_name, limit=30)
        if yf_tickers:
            for t in yf_tickers:
                _add(t, f"sector_yf:{sec_name}")
            yf_count += len(yf_tickers)
            continue
        # Last resort — static universe.
        for t in _get_sector_tickers(sec_name):
            _add(t, f"sector_static:{sec_name}")

    source_status["tradingview"] = tv_count
    source_status["fmp_screener"] = fmp_count
    source_status["yahoo_sector"] = yf_count
    if hot_sectors and tv_count == 0 and fmp_count == 0 and yf_count == 0:
        safe_print("⚠️ Live sector constituents unavailable — using static fallback for all hot sectors")
        log_to_component("tools", "OpportunityFunnel", "universe-source-empty", {
            "source": "sector_constituents",
            "hot_sectors": hot_sectors,
            "note": "Static universe fallback applied per sector",
        })

    # ── 2. Intraday movers ────────────────────────────────────────────────
    if intraday_movers is None:
        try:
            from tools.market_scanner import scan_intraday_movers
            intraday_movers = scan_intraday_movers()
        except Exception:
            intraday_movers = {}

    mover_count = 0
    # Most-active by volume (less of a late signal than % gainers)
    for item in (intraday_movers.get("most_active") or [])[:8]:
        sym = item.get("symbol", "") if isinstance(item, dict) else ""
        if sym:
            _add(sym, "mover:active")
            mover_count += 1
    # Gainers — skip names already extended (avoids chasing parabolas at universe-seed
    # stage; FMP's raw gainers are mostly penny-pumps at +100-300%, correctly excluded).
    for item in (intraday_movers.get("top_gainers") or [])[:8]:
        if not isinstance(item, dict):
            continue
        sym = item.get("symbol", "")
        try:
            chg_pct = float(str(item.get("change", "0%")).replace("%", "").replace("+", ""))
        except (ValueError, TypeError):
            chg_pct = 0.0
        if sym and chg_pct <= _MOVER_LATE_THRESHOLD_PCT:
            _add(sym, "mover:gainer_early")
            mover_count += 1
    # Fallback shape: when scan_intraday_movers can't reach FMP it returns `active_movers`
    # (a yfinance-derived list) instead of most_active/top_gainers. Read it too so the
    # mover source isn't silently empty in that degraded mode (observed in live logs).
    for item in (intraday_movers.get("active_movers") or [])[:12]:
        if not isinstance(item, dict):
            continue
        sym = item.get("symbol", "")
        if sym and sym.upper() not in provenance:
            _add(sym, "mover:active")
            mover_count += 1

    source_status["movers"] = mover_count
    if mover_count == 0:
        log_to_component("tools", "OpportunityFunnel", "universe-source-empty", {
            "source": "movers",
            "note": "Intraday movers returned no data",
        })

    # ── 3. Active geo/policy theme producers ─────────────────────────────
    theme_map = _get_active_theme_tickers()
    theme_count = 0
    for theme_name, tickers in theme_map.items():
        for t in tickers:
            _add(t, f"theme:{theme_name}")
            theme_count += 1
    source_status["themes"] = theme_count

    # ── 4. Institutional accumulation overlay — SEC 13F diffs (Roadmap 5.1) ─
    # New buys + meaningful adds from tracked managers' latest quarter.
    # Replaces the scraped media/guru feed as the universe producer: 13F is
    # keyless, institutional, and accumulation-first — the funnel's own thesis.
    # A caller-provided guru overlay (named Mad Money scans pre-fetch it) is
    # still honored as its own source; it is no longer fetched here.
    inst_count = 0
    try:
        from tools.sec_edgar import get_13f_universe
        for t in get_13f_universe() or []:
            _add(t, "13f")
            inst_count += 1
    except Exception:
        pass
    source_status["13f"] = inst_count

    guru_count = 0
    for t in guru_tickers or []:
        _add(t, "guru")
        guru_count += 1
    source_status["guru"] = guru_count

    # ── 5. Static safety net ──────────────────────────────────────────────
    # (a) Semiconductor/memory seed — these capital-intensive cyclicals are absent
    #     from the built-in fallback universe; without this seed a name like MU could
    #     only enter via a live mover/theme. This is the static guarantee for the
    #     exact failure mode M1 exists to fix.
    for t in _SEMI_SEED:
        _add(t, "static:Semiconductors")
    # (b) Top-6 per known sector so well-established names are never silently excluded.
    for sec_name in _get_all_sector_names():
        for t in _get_sector_tickers(sec_name)[:6]:
            if t.upper() not in provenance:
                _add(t, f"static:{sec_name}")

    # ── 6. Rank by provenance breadth then cap ────────────────────────────
    # Multi-source names carry more cross-signal conviction → rank first.
    # This is what makes a themed name (e.g. MU via memory_chips + sector)
    # survive the downstream _FAST_SCREEN_TOP_N cut.
    ranked = sorted(
        (s for s in provenance if s.upper() not in _known_delisted),  # drop delisted/acquired
        key=lambda s: (-len(provenance[s]), s),   # most sources first, alpha tiebreak
    )
    candidates = ranked[:_UNIVERSE_CAP]
    final_provenance = {s: provenance[s] for s in candidates}

    # ── 7. Loud warning when pool is thin ────────────────────────────────
    total = len(candidates)
    if total < 30:
        msg = (
            f"⚠️ Thin universe ({total} candidates) — data sources degraded. "
            f"Sources: FMP={source_status.get('fmp_screener', 0)} "
            f"Movers={source_status.get('movers', 0)} "
            f"Themes={source_status.get('themes', 0)} "
            f"13F={source_status.get('13f', 0)} "
            f"Guru={source_status.get('guru', 0)}"
        )
        safe_print(msg)
        log_to_component("tools", "OpportunityFunnel", "thin-universe", {
            "count": total, "source_status": source_status,
        })

    safe_print(
        f"🌐 Dynamic universe: {total} candidates | "
        f"FMP:{source_status.get('fmp_screener', 0)} "
        f"Movers:{source_status.get('movers', 0)} "
        f"Themes:{source_status.get('themes', 0)} "
        f"13F:{source_status.get('13f', 0)} "
        f"Guru:{source_status.get('guru', 0)} "
        f"Static fallback included"
    )
    return candidates, final_provenance


def _batch_download_chunked(
    tickers: list[str],
    period: str = _BATCH_DOWNLOAD_PERIOD,
    chunk_size: int = _DOWNLOAD_CHUNK_SIZE,
) -> "pd.DataFrame":
    """
    Chunked wrapper around _batch_download.
    Splits large ticker lists into ≤chunk_size batches to avoid the yfinance
    timeout regression fixed in SCAN_OPPORTUNITIES_OPTIMIZATION.
    Results are merged into a single multi-level DataFrame matching the shape
    that _compute_technicals_batch expects.
    """
    import pandas as pd

    if len(tickers) <= chunk_size:
        return _batch_download(tickers, period=period)

    chunks = [tickers[i:i + chunk_size] for i in range(0, len(tickers), chunk_size)]
    # A trailing single-ticker chunk is dangerous: yf.download of one ticker returns
    # single-level columns (not the (ticker, field) MultiIndex), which corrupts the
    # concat below. Merge any size-1 tail into the previous chunk.
    if len(chunks) >= 2 and len(chunks[-1]) == 1:
        chunks[-2] = chunks[-2] + chunks[-1]
        chunks.pop()
    # PERF: download chunks CONCURRENTLY (was sequential → 36-39s for ~89 names).
    # Each chunk keeps threads=False internally (no nested yfinance threading); we
    # parallelise at the chunk level with a small bounded pool. Disjoint ticker sets
    # keep Yahoo rate-limit risk low. Concurrency is capped to avoid hammering.
    max_parallel = min(len(chunks), 4)
    safe_print(f"📡 Chunked download: {len(tickers)} tickers → {len(chunks)} batches of ≤{chunk_size + 1} ({max_parallel} parallel)")

    frames = []
    executor = ThreadPoolExecutor(max_workers=max_parallel)
    try:
        future_map = {executor.submit(_batch_download, chunk, period): i for i, chunk in enumerate(chunks)}
        for future in future_map:
            if is_cancelled():
                break
            try:
                df = future.result(timeout=45)
                if df is not None and not df.empty:
                    frames.append(df)
            except Exception as e:
                safe_print(f"⚠️ Chunk {future_map[future] + 1} failed: {e}")
    finally:
        executor.shutdown(wait=False, cancel_futures=True)

    if not frames:
        return pd.DataFrame()
    if len(frames) == 1:
        return frames[0]

    # Merge: both frames should be MultiIndex (ticker, field); concat on columns
    try:
        merged = pd.concat(frames, axis=1)
        # Drop duplicate columns that may arise from overlapping tickers
        merged = merged.loc[:, ~merged.columns.duplicated()]
        return merged
    except Exception as e:
        safe_print(f"⚠️ Chunk merge failed ({e}), returning largest batch")
        return max(frames, key=len)


# ---------------------------------------------------------------------------
# FUNNEL V2 / M2 — THEME RANKING & EVENT RADAR
# ---------------------------------------------------------------------------

# Maps free-text substrings (news, policy) → COMMODITY_TICKER_MAP keys.
# Intentionally broad — matched themes are weighted by mention count, not
# treated as confirmed signals.  LLM-based classification is a future add (§14).
_THEME_KEYWORD_MAP: dict[str, str] = {
    # Memory / Semiconductors
    "memory chip": "memory_chips", "dram": "memory_chips", "hbm": "memory_chips",
    "nand": "memory_chips", "flash memory": "memory_chips",
    "semiconductor": "semiconductors", " chip ": "semiconductors",
    "wafer": "semiconductors", "foundry": "semiconductors",
    # Energy
    "crude oil": "oil", "oil price": "oil", "petroleum": "oil", "brent": "oil",
    "natural gas": "natural_gas", "lng": "lng", "liquefied natural gas": "lng",
    "coal": "coal", "uranium": "uranium", "nuclear": "uranium",
    # Metals & materials
    "copper": "copper", "lithium": "lithium", "cobalt": "cobalt",
    "nickel": "nickel", "rare earth": "rare_earths", "palladium": "palladium",
    "platinum": "platinum", "aluminum": "aluminum", "aluminium": "aluminum",
    "iron ore": "iron_ore", "steel tariff": "iron_ore",
    # EVs / Batteries
    "electric vehicle": "ev_batteries", "ev battery": "ev_batteries",
    "battery supply": "ev_batteries",
    # Agriculture
    "wheat": "wheat", "soybean": "soy", "fertilizer": "fertilizer",
    "coffee": "coffee", "rice": "rice",
    # Solar / shipping / niche
    "solar panel": "solar_panels", "photovoltaic": "solar_panels",
    "shipping route": "shipping", "container ship": "shipping",
    "neon gas": "neon_gas", "gallium": "gallium", "germanium": "germanium",
}

# Canonical sector keys + their theme exposures. ALL sector joins (theme→pick,
# catalyst lookup, breadth) route through _canonical_sector() so the differing
# taxonomies — market_mechanics rotation names ("Consumer Discret", "Comm Services"),
# yfinance/_get_sector_for_ticker names ("Finance", "Consumer Defensive",
# "Consumer Cyclical"), and universe names — all collapse to ONE key.
_SECTOR_TO_THEMES: dict[str, list[str]] = {
    "Technology":             ["semiconductors", "memory_chips", "gallium", "germanium", "neon_gas"],
    "Energy":                 ["oil", "natural_gas", "lng", "coal", "uranium"],
    "Materials":              ["copper", "lithium", "rare_earths", "aluminum", "iron_ore"],
    "Industrials":            ["copper", "aluminum", "shipping", "iron_ore"],
    "Consumer Discretionary": ["lithium", "ev_batteries"],
    "Utilities":              ["uranium", "natural_gas"],
    "Healthcare":             [],
    "Financials":             [],
    "Real Estate":            [],
    "Communication Services": [],
    "Consumer Staples":       ["wheat", "soy", "coffee"],
}

# Substring → canonical sector. Order matters (most specific first).
_SECTOR_CANON_RULES: list[tuple[str, str]] = [
    ("semiconduct", "Technology"),
    ("technolog", "Technology"),
    ("financ", "Financials"),          # "Finance", "Financials", "Financial Services"
    ("bank", "Financials"),
    ("health", "Healthcare"),
    ("energy", "Energy"),
    ("consumer discret", "Consumer Discretionary"),
    ("consumer cyclical", "Consumer Discretionary"),
    ("consumer staple", "Consumer Staples"),
    ("consumer defensive", "Consumer Staples"),
    ("industrial", "Industrials"),
    ("utilit", "Utilities"),
    ("real estate", "Real Estate"),
    ("material", "Materials"),
    ("communication", "Communication Services"),
    ("comm services", "Communication Services"),
]


def _canonical_sector(name: str) -> str | None:
    """
    Collapse any sector-name variant (rotation / yfinance / universe taxonomy)
    to a single canonical key from _SECTOR_TO_THEMES. Returns None if unmatched.
    """
    if not name:
        return None
    nl = name.strip().lower()
    for substr, canon in _SECTOR_CANON_RULES:
        if substr in nl:
            return canon
    return None


# Pre-compiled word-boundary patterns for each keyword. The leading \b anchors at
# a word START — this is what kills the false positives "rice" ⊂ "p[rice]s" and
# "coal" ⊂ "[coal]ition". The optional trailing "s?" before \b still lets plurals
# match ("semiconductor" → "semiconductors", "chip" → "chips") without re-admitting
# prefix collisions ("coal" still won't match "coalition").
_THEME_KEYWORD_PATTERNS: list[tuple[Any, str]] = [
    (re.compile(r"\b" + re.escape(kw.strip()) + r"s?\b", re.IGNORECASE), theme)
    for kw, theme in _THEME_KEYWORD_MAP.items()
]


def _resolve_text_to_themes(text: str) -> list[str]:
    """
    Map free-form text to geo/commodity theme keys via _THEME_KEYWORD_MAP.
    Uses word-boundary matching (NOT substring) to avoid false positives such as
    "rice" inside "prices" or "coal" inside "coalition". No NLP/LLM yet (§14).
    """
    if not text:
        return []
    seen: set[str] = set()
    themes: list[str] = []
    for pattern, theme_key in _THEME_KEYWORD_PATTERNS:
        if theme_key not in seen and pattern.search(text):
            seen.add(theme_key)
            themes.append(theme_key)
    return themes


def _get_catalyst_themes_from_events() -> dict[str, float]:
    """
    Scan recent Trump/Truth Social posts and geo alerts for active themes.
    Returns {theme_key: intensity_0_to_1}, normalised by mention count
    (capped at 5 to prevent a single verbose post dominating).

    Both sources are best-effort and fail silently — an empty dict is safe.
    """
    from collections import Counter
    counts: Counter = Counter()

    # Source 1: Trump Truth Social posts (last 7 days)
    try:
        from tools.trump_tracker import get_latest_trump_posts
        posts_data = get_latest_trump_posts(days=7, max_posts=20)
        for post in posts_data.get("posts", []):
            for theme in _resolve_text_to_themes(post.get("text", "")):
                counts[theme] += 1
    except Exception:
        pass

    # Source 2: Geopolitical quick check (GDELT / web) — shared/cached with M1
    try:
        check = _cached_geo_check()
        if check.get("alert"):
            for commodity in check.get("commodities_at_risk", []):
                key = commodity.lower().replace(" ", "_")
                counts[key] += 2  # geo alert is a stronger/confirmed signal
    except Exception:
        pass

    if not counts:
        return {}

    # Absolute intensity (not relative-to-max): a single mention is a weak signal
    # (0.25), corroboration scales it up, 4+ mentions / a geo alert + mention = full.
    # This avoids one stray keyword hit maxing out a theme's catalyst weight.
    _DENOM = 4
    return {theme: round(min(count, _DENOM) / _DENOM, 3) for theme, count in counts.items()}


def _cycle_stage_from_trend(trend: str) -> str:
    """Classify market cycle stage from a rotation trend string."""
    tl = trend.lower()
    if "improving" in tl:
        return "early"
    if "leading" in tl and "overbought" in tl:
        return "late"
    if "leading" in tl:
        return "mid"
    if "weakening" in tl:
        return "late"
    return "neutral"


def _rank_themes(
    rotation_data: dict,
    macro_context: dict,
    macro_bullish_sectors: list[str],
    macro_bearish_sectors: list[str],
    universe_provenance: dict[str, list[str]] | None = None,
) -> list[dict]:
    """
    Funnel V2 Stage 1 — Event & Flow Radar (M2).

    Scores each sector/theme in the rotation map using:
        theme_score = 0.45 * rotation
                    + 0.25 * macro_alignment
                    + 0.20 * catalyst_intensity
                    + 0.10 * breadth

    Returns dicts sorted by theme_score descending, each including
    cycle_stage (early / mid / late / neutral) and the component scores.
    Uses only already-fetched Phase-0 data + a best-effort catalyst fetch —
    no extra blocking API calls when themes/sources are unavailable.
    """
    if universe_provenance is None:
        universe_provenance = {}

    # Fetch catalyst intensities (Trump posts + geo — cached/fast-fail)
    catalyst_map: dict[str, float] = {}
    try:
        catalyst_map = _get_catalyst_themes_from_events()
    except Exception:
        pass

    # Breadth: fraction of sector's candidates that carry ≥2 provenance sources.
    # Keyed by canonical sector — universe tags ("sector_static:Industrials",
    # "static:Technology") and rotation-map names ("Consumer Discret") use
    # different taxonomies, so both sides of the join must be canonicalized.
    sector_breadth: dict[str, float] = {}
    if universe_provenance:
        sector_candidates: dict[str, list[str]] = {}
        for sym, sources in universe_provenance.items():
            for src in sources:
                for prefix in ("sector:", "sector_yf:", "sector_static:", "static:"):
                    if src.startswith(prefix):
                        canon_sec = _canonical_sector(src[len(prefix):])
                        if canon_sec:
                            sector_candidates.setdefault(canon_sec, []).append(sym)
                        break
        for sec, syms in sector_candidates.items():
            if syms:
                multi = sum(1 for s in syms if len(universe_provenance.get(s, [])) >= 2)
                sector_breadth[sec] = round(multi / len(syms), 3)

    ranked: list[dict] = []
    seen: set[str] = set()

    for item in rotation_data.get("full_rotation_map", []) or []:
        if not isinstance(item, dict):
            continue
        sector_name = item.get("sector", "")
        if not sector_name or sector_name in seen:
            continue
        seen.add(sector_name)
        trend = str(item.get("trend", "Neutral ⚪"))
        tl = trend.lower()

        # ── rotation score (0-1) ──────────────────────────────────────
        if "improving" in tl:
            rot = 0.7
        elif "leading" in tl:
            rot = 1.0
        elif "weakening" in tl:
            rot = 0.25
        else:
            rot = 0.0

        # ── macro alignment (0-1) ─────────────────────────────────────
        su = sector_name.upper()
        is_favored = any(su in s or s in su for s in macro_bullish_sectors)
        is_bearish = any(su in s or s in su for s in macro_bearish_sectors)
        if is_favored:
            mac = 1.0
        elif is_bearish:
            mac = 0.0
        elif macro_context.get("liquidity") == "Expanding":
            mac = 0.7
        else:
            mac = 0.5

        # ── catalyst intensity (0-1) ──────────────────────────────────
        canon = _canonical_sector(sector_name)
        sector_themes = _SECTOR_TO_THEMES.get(canon, []) if canon else []
        cat = max((catalyst_map.get(t, 0.0) for t in sector_themes), default=0.0)

        # ── breadth (0-1) ─────────────────────────────────────────────
        breadth = sector_breadth.get(canon, 0.3) if canon else 0.3

        # ── weighted score ────────────────────────────────────────────
        theme_score = round(0.45 * rot + 0.25 * mac + 0.20 * cat + 0.10 * breadth, 3)

        # Driver narrative (for output readability)
        drivers: list[str] = []
        if rot >= 0.7:
            drivers.append(f"Rotation: {trend.split('⚠️')[0].strip()}")
        if is_favored:
            drivers.append("Macro: Tactically favored")
        if cat > 0:
            active = [t for t in sector_themes if catalyst_map.get(t, 0) > 0]
            if active:
                drivers.append(f"Catalyst: {', '.join(active[:2])}")

        ranked.append({
            "theme":             sector_name,
            "sector":            sector_name,
            "canonical_sector":  canon,   # for cross-taxonomy join with picks
            "theme_score":       theme_score,
            "rotation_score":    round(rot, 3),
            "macro_score":       round(mac, 3),
            "catalyst_score":    round(cat, 3),
            "breadth_score":     round(breadth, 3),
            "cycle_stage":       _cycle_stage_from_trend(trend),
            "drivers":           drivers,
            "trend":             trend,
        })

    ranked.sort(key=lambda x: -x["theme_score"])
    return ranked


# ---------------------------------------------------------------------------
# FUNNEL V2 / M3 — ADDITIVE SCORING + FLOW / ENTRY GOVERNOR
# ---------------------------------------------------------------------------

def _coerce_percent(value: Any) -> float | None:
    """Parse values like 0.67, '67%', or '+12.5%' into percentage points."""
    if value is None:
        return None
    if isinstance(value, (int, float, np.floating)):
        val = float(value)
        return val * 100 if abs(val) <= 1 else val
    if isinstance(value, str):
        cleaned = value.strip().replace("%", "").replace("+", "")
        if not cleaned:
            return None
        try:
            return float(cleaned)
        except ValueError:
            return None
    return None


def _linear_points(value: float, low: float, high: float, max_points: float) -> float:
    """Map value linearly to 0..max_points, clamped at low/high."""
    if high == low:
        return 0.0
    if value <= low:
        return 0.0
    if value >= high:
        return max_points
    return ((value - low) / (high - low)) * max_points


def _theme_for_sector(ticker_sector: str, sector_theme_map: dict[str, dict] | None) -> dict:
    """Return the ranked-theme record for a ticker sector using the canonical join."""
    if not sector_theme_map:
        return {}
    canon = _canonical_sector(ticker_sector)
    if not canon:
        return {}
    return sector_theme_map.get(canon, {}) or {}


def _m3_theme_points(theme_context: dict | None) -> tuple[float, list[str]]:
    if not theme_context:
        return 0.0, []
    raw = _safe_float(theme_context.get("theme_score")) or 0.0
    norm = raw / 100 if raw > 1 else raw
    max_points = _cfg_number("pillars.theme", _M3_PILLAR_WEIGHTS["theme"])
    points = round(max(0.0, min(norm, 1.0)) * max_points, 1)
    reasons = []
    if points > 0:
        label = theme_context.get("theme") or theme_context.get("sector") or "Theme"
        reasons.append(f"Theme leadership: {label} ({points:.1f}/{max_points:.0f})")
    for driver in theme_context.get("drivers", [])[:2]:
        reasons.append(str(driver))
    return points, reasons


def _m3_relstr_points(rs_alpha: float | None) -> tuple[float, list[str]]:
    """Map 3M alpha vs SPY in percentage points to the 0..25 RS pillar."""
    max_points = _cfg_number("pillars.relstr", _M3_PILLAR_WEIGHTS["relstr"])
    alpha = float(rs_alpha or 0.0)
    if alpha >= 15:
        points = max_points
    elif alpha >= 5:
        points = (max_points * 0.60) + (alpha - 5.0)
    elif alpha >= 0:
        points = (max_points * 0.40) + alpha
    elif alpha > -15:
        points = ((alpha + 15.0) / 15.0) * (max_points * 0.40)
    else:
        points = 0.0
    reasons = []
    if alpha >= 5:
        reasons.append(f"Relative strength: +{alpha:.1f}% vs SPY over 3M")
    elif alpha <= -10:
        reasons.append(f"Relative strength lag: {alpha:.1f}% vs SPY over 3M")
    return round(max(0.0, min(points, max_points)), 1), reasons


def _m3_forward_points(symbol: str, fund: dict, price: float) -> tuple[float, list[str], dict[str, Any]]:
    """
    Forward pillar, constrained to available data:
    forward-vs-trailing PE re-rating, analyst target direction, and a small
    historical beat-rate component. This is not true estimate-revision breadth.
    """
    reasons: list[str] = []
    details: dict[str, Any] = {}
    points = 0.0

    fwd_pe = _safe_float(fund.get("forward_pe"))
    trail_pe = _safe_float(fund.get("trailing_pe"))
    if fwd_pe and trail_pe and fwd_pe > 0 and trail_pe > 0 and fwd_pe < trail_pe:
        rerate_pct = ((trail_pe - fwd_pe) / trail_pe) * 100
        pe_points = _linear_points(rerate_pct, 5.0, 35.0, 10.0)
        points += pe_points
        details["pe_rerating_pct"] = round(rerate_pct, 1)
        if pe_points >= 3:
            reasons.append(f"Forward PE re-rating: {fwd_pe:.1f} vs trailing {trail_pe:.1f}")
    elif fwd_pe and 0 < fwd_pe < 30:
        points += 2.0
        details["fair_forward_pe"] = round(fwd_pe, 1)

    target = _safe_float(fund.get("analyst_target"))
    if target and price > 0:
        upside = ((target - price) / price) * 100
        target_points = _linear_points(upside, 0.0, 35.0, 10.0)
        points += target_points
        details["analyst_upside_pct"] = round(upside, 1)
        if target_points >= 3:
            reasons.append(f"Analyst target upside: +{upside:.0f}%")

    beat_points = 0.0
    try:
        from tools.market_mechanics import predict_earnings_surprise
        surprise = predict_earnings_surprise(symbol)
        beat_rate = _coerce_percent(surprise.get("beat_rate") if isinstance(surprise, dict) else None)
        if beat_rate is not None:
            beat_points = _linear_points(beat_rate, 50.0, 85.0, 5.0)
            details["historical_beat_rate_pct"] = round(beat_rate, 1)
            if beat_points >= 2:
                reasons.append(f"Historical beat-rate support: {beat_rate:.0f}%")
    except Exception:
        pass
    points += beat_points

    points = round(max(0.0, min(points, _cfg_number("pillars.forward", _M3_PILLAR_WEIGHTS["forward"]))), 1)
    return points, reasons, details


def _m3_quality_points(fund: dict, tech: dict) -> tuple[float, list[str], dict[str, Any]]:
    """Quality is a bonus band only; missing/weak fields never gate the idea."""
    reasons: list[str] = []
    details: dict[str, Any] = {}
    points = 0.0

    rev_growth = _safe_float(fund.get("revenue_growth"))
    earn_growth = _safe_float(fund.get("earnings_growth"))
    margin = _safe_float(fund.get("profit_margin"))
    fwd_pe = _safe_float(fund.get("forward_pe"))
    peg = _safe_float(fund.get("peg_ratio"))

    if rev_growth is not None:
        details["revenue_growth"] = rev_growth
        if rev_growth > 0.30:
            points += 6
            reasons.append(f"Revenue growth bonus: {rev_growth:.0%}")
        elif rev_growth > 0.15:
            points += 4
            reasons.append(f"Revenue growth bonus: {rev_growth:.0%}")
        elif rev_growth > 0.05:
            points += 2
        elif rev_growth > 0:
            points += 1

    if earn_growth is not None:
        details["earnings_growth"] = earn_growth
        if rev_growth is not None and earn_growth > rev_growth + 0.05 and earn_growth > 0:
            points += 4
            reasons.append("Earnings acceleration")
        elif earn_growth > 0.15:
            points += 3

    if margin is not None:
        details["profit_margin"] = margin
        if margin > 0.25:
            points += 5
            reasons.append(f"Strong margin bonus: {margin:.0%}")
        elif margin > 0.15:
            points += 3
        elif margin > 0.05:
            points += 1

    if peg is not None and 0 < peg < 1.5:
        points += 3
        details["peg_ratio"] = peg
    elif fwd_pe is not None and 0 < fwd_pe < 30:
        points += 2
        details["forward_pe"] = fwd_pe

    if tech.get("above_sma50") and (margin or 0) > 0.15:
        points += 2

    points = round(max(0.0, min(points, _cfg_number("pillars.quality", _M3_PILLAR_WEIGHTS["quality"]))), 1)
    return points, reasons, details


def _entry_stage_from_setup(
    tech: dict,
    setup_data: dict | None = None,
    theme_context: dict | None = None,
) -> tuple[str, float, list[str]]:
    """Classify the entry stage and return (stage, multiplier, reasons)."""
    setup_text = str((setup_data or {}).get("setup", ""))
    setup_upper = setup_text.upper()
    rsi = _safe_float((setup_data or {}).get("rsi")) or _safe_float(tech.get("rsi")) or 50.0
    price = _safe_float(tech.get("price")) or _safe_float((setup_data or {}).get("price")) or 0.0
    sma50 = _safe_float(tech.get("sma50"))
    above_sma50 = bool(tech.get("above_sma50"))
    month_return = _safe_float(tech.get("month_return")) or 0.0
    drawdown = _safe_float(tech.get("drawdown_pct")) or 0.0
    theme_cycle = str((theme_context or {}).get("cycle_stage", "neutral")).lower()
    over_sma50_pct = 0.0
    if price > 0 and sma50 and sma50 > 0:
        over_sma50_pct = ((price - sma50) / sma50) * 100

    reasons: list[str] = []
    entry_multipliers = _cfg_section("entry_multipliers")
    extended_over_sma50_pct = _cfg_number("extended_over_sma50_pct", _M3_EXTENDED_OVER_SMA50_PCT)
    if "EXTENDED" in setup_upper or rsi >= 75 or over_sma50_pct >= extended_over_sma50_pct:
        reasons.append(f"Entry stage: extended (RSI {rsi:.0f}, {over_sma50_pct:.0f}% over 50DMA)")
        return "extended", float(entry_multipliers.get("extended", _M3_ENTRY_MULTIPLIERS["extended"])), reasons

    inflowing_theme = theme_cycle in {"early", "mid"}
    if (
        "OVERSOLD" in setup_upper
        or "BB BOUNCE" in setup_upper
        or (rsi < 40 and drawdown <= -10 and inflowing_theme)
        or (drawdown <= -20 and inflowing_theme)
    ):
        reasons.append("Entry stage: accumulation base in an inflowing theme")
        return "accumulation_base", float(entry_multipliers.get("accumulation_base", _M3_ENTRY_MULTIPLIERS["accumulation_base"])), reasons

    if (
        "MOMENTUM BREAKOUT" in setup_upper
        or tech.get("golden_cross")
        or (above_sma50 and month_return > 5 and 45 <= rsi < 75)
    ):
        reasons.append("Entry stage: early breakout")
        return "early_breakout", float(entry_multipliers.get("early_breakout", _M3_ENTRY_MULTIPLIERS["early_breakout"])), reasons

    if above_sma50 and 55 <= rsi <= 70:
        reasons.append("Entry stage: mid-trend")
        return "mid_trend", float(entry_multipliers.get("mid_trend", _M3_ENTRY_MULTIPLIERS["mid_trend"])), reasons

    if theme_cycle == "late":
        reasons.append("Entry stage: late-cycle trend")
        return "mid_trend", float(entry_multipliers.get("mid_trend", _M3_ENTRY_MULTIPLIERS["mid_trend"])), reasons

    reasons.append("Entry stage: neutral breakout candidate")
    return "early_breakout", float(entry_multipliers.get("early_breakout", _M3_ENTRY_MULTIPLIERS["early_breakout"])), reasons


def _m3_risk_adjustment(headwind_data: dict | None, tech: dict) -> tuple[float, list[str], list[str], list[dict[str, Any]]]:
    """Convert headwind data into capped risk adjustment plus surfaced flags."""
    headwind_data = headwind_data or {}
    risk_adjust = 0.0
    risk_flags: list[str] = []
    reasons: list[str] = []
    adjustments: list[dict[str, Any]] = []

    short_pct = _safe_float(headwind_data.get("short_pct_float"))
    if short_pct is not None:
        if short_pct > 0.15:
            points = 8.0
            risk_adjust += points
            risk_flags.append(f"{short_pct*100:.0f}% short float - elevated bearish/covering-risk signal")
            reasons.append(f"Risk: heavy short interest ({short_pct*100:.0f}%)")
            adjustments.append({"type": "short_interest", "points": points, "detail": risk_flags[-1]})
        elif short_pct > 0.10:
            points = 4.0
            risk_adjust += points
            risk_flags.append(f"{short_pct*100:.0f}% short float - above-average bearish bets")
            adjustments.append({"type": "short_interest", "points": points, "detail": risk_flags[-1]})

    tone = headwind_data.get("management_tone")
    if tone == "Highly Cautious (Bearish)":
        points = 5.0
        risk_adjust += points
        risk_flags.append("Bearish management tone in last earnings call")
        reasons.append("Risk: bearish management tone")
        adjustments.append({"type": "management_tone", "points": points, "detail": risk_flags[-1]})

    days_to_earn = headwind_data.get("days_to_earnings")
    if days_to_earn is not None and 0 <= days_to_earn <= 7:
        rsi = _safe_float(tech.get("rsi")) or 50.0
        if rsi > 65:
            points = 5.0
            risk_adjust += points
            risk_flags.append(f"Earnings in {days_to_earn} days while RSI is elevated ({rsi:.0f})")
            reasons.append(f"Risk: earnings imminent ({days_to_earn}d)")
            adjustments.append({"type": "earnings_proximity", "points": points, "detail": risk_flags[-1]})
        else:
            risk_flags.append(f"Earnings in {days_to_earn} days - event volatility expected")
            adjustments.append({"type": "earnings_proximity", "points": 0.0, "detail": risk_flags[-1]})

    insider_signal = str(headwind_data.get("insider_signal", ""))
    if "SELLING" in insider_signal:
        points = 7.0
        risk_adjust += points
        risk_flags.append("Insiders selling recently")
        reasons.append("Risk: insider selling")
        adjustments.append({"type": "insider_selling", "points": points, "detail": risk_flags[-1]})

    return min(risk_adjust, _cfg_number("risk_cap", _M3_RISK_CAP)), list(dict.fromkeys(risk_flags)), reasons, adjustments


# How this module measures sector exposure — and, because two tools in one turn
# can be asked the same question, the label that keeps its answer from being read
# as a contradiction of the other's. See tools.sector_analysis, whose
# check_portfolio_allocation decomposes funds and therefore reports MORE Technology
# than this does out of the same book. Neither is wrong; they are different
# measures, and only the labels make that legible to a reader (human or judge).
SECTOR_EXPOSURE_BASIS_DIRECT = "directly-held names only; funds not decomposed"


def _portfolio_fit_adjustment(
    symbol: str,
    ticker_sector: str,
    portfolio_context: dict | None,
) -> dict[str, Any]:
    """
    M4 portfolio-fit overlay: cap concentration risk and surface it as sizing guidance.
    This informs ranking/sizing; it never silently deletes an otherwise-strong idea.

    Its exposure figure counts a holding toward a sector only when the whole ticker
    carries that sector's label — ``_get_sector_for_ticker`` returns ONE label per
    ticker, so every fund lands all-or-nothing. A broad ETF labels "Large Blend" or
    "Asset Allocation", which is no GICS sector, and contributes zero while its
    dollars stay in the denominator; a sector ETF like QQQ contributes its full
    value. On a fund-heavy book that is a large undercount (measured: a book 70% in
    broad funds reports 70% of itself against no sector at all), which is why the
    figure ships with its basis and ``unmapped_pct`` — the share it could not
    classify — rather than as a bare percentage.
    """
    empty = {"risk_adjust": 0.0, "risk_flags": [], "reasons": [], "adjustments": [], "portfolio_fit": {}}
    if not portfolio_context or not isinstance(portfolio_context, dict):
        return empty

    holdings = portfolio_context.get("holdings") or []
    total_value_usd = _safe_float(portfolio_context.get("total_value_usd"))
    if not holdings or not total_value_usd or total_value_usd <= 0:
        return empty

    # Cross-listing check runs against ALL holdings, not just same-sector ones:
    # a CDR twin (e.g. MA.TO) may map to sector "Unknown" and never enter the
    # sector loop, which is exactly how MA was mislabeled "not held".
    from tools.ticker_equivalence import find_equivalent_holding
    holding_syms = [
        str(h.get("symbol", "")).upper()
        for h in holdings
        if isinstance(h, dict) and h.get("symbol")
    ]
    equivalent_holding = find_equivalent_holding(symbol, holding_syms)
    overlap_flags: list[str] = []
    overlap_reasons: list[str] = []
    if equivalent_holding:
        overlap_flags.append(
            f"Economic twin already held ({equivalent_holding}) — overlap with an existing position, not new exposure"
        )
        overlap_reasons.append(
            f"Portfolio fit: overlap via cross-listed equivalent {equivalent_holding}"
        )
    base_fit = {
        "candidate_already_held": symbol.upper() in set(holding_syms),
        "economic_equivalent_held": equivalent_holding,
    }
    base_result = {
        **empty,
        "risk_flags": list(overlap_flags),
        "reasons": list(overlap_reasons),
        "portfolio_fit": dict(base_fit),
    }

    candidate_canon = _canonical_sector(ticker_sector)
    if not candidate_canon:
        return base_result

    sector_value = 0.0
    sector_holdings: list[str] = []
    unmapped_value = 0.0
    for holding in holdings:
        if not isinstance(holding, dict):
            continue
        h_sym = str(holding.get("symbol", "")).upper()
        if not h_sym:
            continue
        h_sector = _get_sector_for_ticker(h_sym)
        h_canon = _canonical_sector(h_sector)
        value_usd = _safe_float(holding.get("value_usd")) or 0.0
        if h_canon is None:
            # A broad ETF, balanced fund, cash line or private holding: one
            # ticker, one label, and labels like "Large Blend" belong to no GICS
            # sector. Its sector sleeves are invisible to this measure — count
            # the dollars so the incompleteness travels with the percentage.
            unmapped_value += value_usd
            continue
        if h_canon != candidate_canon:
            continue
        sector_value += value_usd
        sector_holdings.append(h_sym)

    if sector_value <= 0:
        return base_result

    exposure_pct = (sector_value / total_value_usd) * 100
    threshold_pct = _cfg_number("concentration.threshold_pct", 25.0)
    penalty_per_excess = _cfg_number("concentration.penalty_per_excess_pct", 1.0)
    risk_cap = _cfg_number("risk_cap", _M3_RISK_CAP)
    unmapped_pct = (unmapped_value / total_value_usd) * 100
    portfolio_fit = {
        "candidate_sector": candidate_canon,
        "current_sector_exposure_pct": round(exposure_pct, 1),
        "threshold_pct": round(threshold_pct, 1),
        "sector_holdings": sorted(set(sector_holdings)),
        "basis": SECTOR_EXPOSURE_BASIS_DIRECT,
        "unmapped_pct": round(unmapped_pct, 1),
        **base_fit,
    }
    if exposure_pct <= threshold_pct:
        return {**base_result, "portfolio_fit": portfolio_fit}

    excess_pct = exposure_pct - threshold_pct
    penalty = min(excess_pct * penalty_per_excess, risk_cap)
    # Say what the number measures, in the string itself. This figure and
    # check_portfolio_allocation's answer the same question by different rules —
    # this one labels each ticker once and drops funds it cannot label, that one
    # decomposes funds into their sector sleeves — so the two legitimately differ
    # and the look-through figure is the higher, more complete one. Unlabelled,
    # they read as a contradiction: on 2026-07-29 the compliance judge took this
    # string as proof that a true look-through Technology weight was fabricated,
    # and issued a 2/10 SOURCE FRAUD verdict over it.
    flag = (
        f"Adds to {exposure_pct:.0f}% {candidate_canon} exposure "
        f"({SECTOR_EXPOSURE_BASIS_DIRECT}) - size accordingly"
    )
    reason = (
        f"Risk: sector concentration {exposure_pct:.0f}% {candidate_canon} exposure, "
        f"{SECTOR_EXPOSURE_BASIS_DIRECT} (capped -{penalty:.0f})"
    )
    adjustment = {
        "type": "sector_concentration",
        "points": round(penalty, 1),
        "detail": flag,
        "exposure_pct": round(exposure_pct, 1),
        "threshold_pct": round(threshold_pct, 1),
        "cap": risk_cap,
    }
    return {
        "risk_adjust": round(penalty, 1),
        "risk_flags": overlap_flags + [flag],
        "reasons": overlap_reasons + [reason],
        "adjustments": [adjustment],
        "portfolio_fit": portfolio_fit,
    }


def _flow_confirmation_for_symbol(symbol: str, insider_signal: str = "") -> dict[str, Any]:
    """Run capped confirmation-only flow proxies for one finalist.

    Every signal here is a PROXY derived from public price/volume/options
    data, not actual dark-pool tape or exchange flow. ``flow_evidence``
    carries the underlying observations (source, counts, prints) so
    downstream verification can grade the claims instead of taking the
    summary strings on faith.
    """
    confirmations: list[str] = []
    evidence: list[dict[str, Any]] = []

    try:
        from tools.dark_pool import scan_dark_pool_proxy
        dark = scan_dark_pool_proxy(symbol) or {}
        alerts = dark.get("alerts", []) or []
        bullish_alerts = [
            a for a in alerts
            if "SELL" not in str(a.get("signature", "")).upper()
        ]
        hit_count = len(bullish_alerts) if alerts else int(dark.get("alerts_count", 0) or 0)
        if hit_count > 0:
            confirmations.append(f"Dark-pool/block proxy prints x{hit_count}")
            evidence.append({
                "signal": "dark_pool_proxy",
                "source": "1-min intraday volume-spike proxy (yfinance), not actual dark-pool tape",
                "count": hit_count,
                "prints": bullish_alerts[:3],
            })
    except Exception:
        pass

    try:
        from tools.options import check_whale_accumulation
        whale = check_whale_accumulation(symbol) or {}
        whale_count = int(whale.get("count", 0) or 0)
        if whale_count > 0:
            confirmations.append(f"ITM call sweep proxy x{whale_count}")
            evidence.append({
                "signal": "itm_call_sweep_proxy",
                "source": "options open-interest/volume proxy, not sweep tape",
                "count": whale_count,
            })
    except Exception:
        pass

    try:
        from tools.options import scan_unusual_activity
        unusual = scan_unusual_activity(symbol) or {}
        alerts = [str(a).upper() for a in unusual.get("alerts", []) or []]
        bullish = any("CALL" in a or "BULLISH" in a for a in alerts)
        bearish = any("PUT" in a for a in alerts)
        if bullish and not bearish:
            confirmations.append("Bullish unusual-options activity (proxy)")
            evidence.append({
                "signal": "unusual_options_proxy",
                "source": "options volume-vs-OI proxy",
                "alerts": alerts[:3],
            })
    except Exception:
        pass

    if "BUYING" in str(insider_signal).upper():
        confirmations.append("Insider buying (reported filings)")
        evidence.append({
            "signal": "insider_buying",
            "source": "reported insider transactions (headwind feed)",
            "detail": str(insider_signal),
        })

    signal_count = len(confirmations)
    flow_bonus_cfg = _cfg_section("flow_bonus")
    if signal_count >= 2:
        bonus = float(flow_bonus_cfg.get("two_plus", _M3_FLOW_BONUS["two_plus"]))
    elif signal_count == 1:
        bonus = float(flow_bonus_cfg.get("one", _M3_FLOW_BONUS["one"]))
    else:
        bonus = 0.0

    return {
        "flow_bonus": bonus,
        "flow_confirmations": confirmations,
        "flow_signal_count": signal_count,
        "flow_evidence": evidence,
    }


def _collect_bounded(future_map: dict, overall_budget: float, default=None) -> dict[str, Any]:
    """Collect {symbol: result} from an already-submitted future map under ONE
    wall-clock budget for the WHOLE batch.

    The pattern this replaces — iterating `future_map` in submission order and
    calling `future.result(timeout=N)` on each — restarts the clock per symbol,
    so N symbols cost up to N×timeout. The Phase-5 gates ran that way with 15
    finalists and a 15-18s per-future timeout, i.e. a ~225s worst case inside a
    150s scan; that is how Phase 5 sailed 33s past the scan deadline on
    2026-07-28 while every individual timeout looked reasonable.

    `default` (a zero-arg factory) fills symbols that failed or never finished,
    for gates whose scoring needs an entry per finalist. Without it, unfinished
    symbols are simply absent — which is the safe reading for a risk gate.
    """
    results: dict[str, Any] = {}
    deadline = time.perf_counter() + max(1.0, overall_budget)
    try:
        for fut in as_completed(list(future_map), timeout=overall_budget):
            sym = future_map[fut]
            try:
                value = fut.result()
                if value:
                    results[sym] = value
            except Exception:
                pass  # failed symbol — `default` (if any) fills it below
            if is_cancelled() or time.perf_counter() >= deadline:
                break
    except Exception:
        pass  # budget elapsed — stragglers stay absent rather than blocking
    if default is not None:
        for sym in future_map.values():
            results.setdefault(sym, default())
    return results


def _flow_confirmation_parallel(symbols: list[str], headwind_map: dict[str, dict] | None = None, max_workers: int = 4, overall_budget: float = 40.0) -> dict[str, dict]:
    """Run flow confirmation on finalists only."""
    if not symbols:
        return {}
    headwind_map = headwind_map or {}
    safe_print(f"🌊 Running flow confirmation for {len(symbols)} finalists...")
    results: dict[str, dict] = {}
    _prof = get_active_profile()
    executor = ThreadPoolExecutor(max_workers=min(len(symbols), max_workers))
    try:
        future_map = {
            executor.submit(
                run_under_profile,
                _prof,
                _flow_confirmation_for_symbol,
                sym,
                str(headwind_map.get(sym, {}).get("insider_signal", "")),
            ): sym
            for sym in symbols
        }
        results = _collect_bounded(
            future_map, overall_budget,
            default=lambda: {"flow_bonus": 0.0, "flow_confirmations": [],
                             "flow_signal_count": 0, "flow_evidence": []},
        )
    finally:
        executor.shutdown(wait=False, cancel_futures=True)
    confirmed = sum(1 for item in results.values() if item.get("flow_signal_count", 0) > 0)
    safe_print(f"🌊 Flow confirmation complete: {confirmed}/{len(symbols)} finalists confirmed")
    return results


def _setup_check_parallel(symbols: list[str], max_workers: int = 4, overall_budget: float = 30.0) -> dict[str, dict]:
    """Run check_setup on finalists only for the entry-stage multiplier."""
    if not symbols:
        return {}
    safe_print(f"🎚️ Checking entry setup for {len(symbols)} finalists...")
    results: dict[str, dict] = {}
    executor = ThreadPoolExecutor(max_workers=min(len(symbols), max_workers))
    try:
        try:
            from tools.screener import check_setup
        except Exception:
            return {}
        _prof = get_active_profile()
        future_map = {executor.submit(run_under_profile, _prof, check_setup, sym): sym for sym in symbols}
        results = _collect_bounded(future_map, overall_budget)
    finally:
        executor.shutdown(wait=False, cancel_futures=True)
    return results


@log_exceptions()
def _resolve_scoring_context(
    symbol: str,
    fund: dict,
    sector_trend_map: dict,
    macro_bullish_sectors: list,
    macro_bearish_sectors: list,
    macro_context: dict,
    sector_theme_map: dict,
) -> tuple[str, str, dict, list[str], dict]:
    """Resolve the per-candidate scoring context shared by Phases 4 and 5.

    Returns ``(ticker_sector, current_trend, local_macro, thematic_tags,
    theme_context)``. ``local_macro`` is a copy of ``macro_context`` annotated
    with this ticker's macro favored/disfavored alignment; ``theme_context`` is
    the ranked-theme join (empty for non-broad scans, where ``sector_theme_map``
    is empty). Extracted verbatim from the three identical inline copies that
    previously lived in the Phase-4 and Phase-5 scoring loops.
    """
    ticker_sector = _get_sector_for_ticker(symbol, fund=fund)
    sec_upper = ticker_sector.upper()
    current_trend = "Neutral ⚪"
    for k, v in sector_trend_map.items():
        if k in sec_upper or sec_upper in k:
            current_trend = v
            break
    is_favored = any(sec_upper in s for s in macro_bullish_sectors) or any(s in sec_upper for s in macro_bullish_sectors)
    is_disfavored = any(sec_upper in s for s in macro_bearish_sectors) or any(s in sec_upper for s in macro_bearish_sectors)
    local_macro = dict(macro_context)
    local_macro["is_macro_favored"] = is_favored
    local_macro["is_macro_disfavored"] = is_disfavored
    thematic_tags = _get_thematic_tags(symbol)
    theme_context = _theme_for_sector(ticker_sector, sector_theme_map)
    return ticker_sector, current_trend, local_macro, thematic_tags, theme_context


def _deep_score_v2(
    symbol: str,
    fund: dict,
    tech: dict,
    sector_trend: str,
    macro_context: dict,
    thematic_tags: list[str],
    portfolio_context: dict | None = None,
    **kwargs,
) -> dict | None:
    """
    Funnel V2 / M3 scoring path:
    additive base pillars first, then capped flow confirmation and an entry-stage
    multiplier on finalists. Quality/foundation data is informational, never a gate.
    """
    # sector_trend is the fallback rotation trend for sector-scoped scans;
    # broad scans get the trend from the M2 ranked theme map (theme_context).
    del thematic_tags      # Legacy thematic tags are still returned, not a score gate.

    price = _safe_float(tech.get("price")) or _safe_float(fund.get("current_price"))
    if not price or price <= 0:
        return None

    theme_context = kwargs.get("theme_context") or {}
    rs_alpha = _safe_float(kwargs.get("rs_alpha")) or 0.0
    flow_data = kwargs.get("flow_data") or {}
    setup_data = kwargs.get("setup_data") or {}
    headwind_data = kwargs.get("headwind_data") or {}
    apply_entry_gate = bool(kwargs.get("apply_entry_gate", False))
    provenance = kwargs.get("universe_provenance") or []

    reasons: list[str] = []
    component_details: dict[str, Any] = {}

    p_theme, theme_reasons = _m3_theme_points(theme_context)
    reasons.extend(theme_reasons)

    p_relstr, rs_reasons = _m3_relstr_points(rs_alpha)
    reasons.extend(rs_reasons)

    p_forward, forward_reasons, forward_details = _m3_forward_points(symbol, fund, price)
    reasons.extend(forward_reasons)
    component_details["forward"] = forward_details

    p_quality, quality_reasons, quality_details = _m3_quality_points(fund, tech)
    reasons.extend(quality_reasons)
    component_details["quality"] = quality_details

    base_score = round(p_theme + p_relstr + p_forward + p_quality, 1)

    headlines = fund.get("news_headlines", []) if isinstance(fund, dict) else []
    _news_score, catalyst_reasons, catalyst_labels = _score_news_catalysts(headlines)
    # M3 does not add ad-hoc news points to the base; headlines feed the pattern label.
    reasons.extend(catalyst_reasons[:2])

    raw_flow_bonus = _safe_float(flow_data.get("flow_bonus")) or 0.0
    effective_flow_bonus = raw_flow_bonus if base_score >= _MIN_SCORE_THRESHOLD else 0.0
    flow_confirmations = list(flow_data.get("flow_confirmations", []) or [])
    flow_evidence = list(flow_data.get("flow_evidence", []) or [])
    if flow_confirmations:
        reasons.append(f"Flow confirmation (proxy-derived): {', '.join(flow_confirmations[:2])}")
    if raw_flow_bonus and not effective_flow_bonus:
        reasons.append("Flow seen, but base score is below the qualification threshold")

    if apply_entry_gate:
        entry_stage, entry_multiplier, entry_reasons = _entry_stage_from_setup(tech, setup_data, theme_context)
        reasons.extend(entry_reasons)
    else:
        entry_stage, entry_multiplier = "base_scoring", 1.0

    ticker_sector = _get_sector_for_ticker(symbol, fund=fund)
    headwind_adjust, risk_flags, risk_reasons, risk_adjustments = _m3_risk_adjustment(headwind_data, tech)
    reasons.extend(risk_reasons)

    # Falling-knife guard: a Lagging sector means rotation momentum is still
    # against the trade no matter how cheap it screens. Penalize, flag, and
    # mark watchlist-only so a "dip-buy" in the worst sector can never render
    # with an empty risk column (profile rule: never catch a falling knife).
    rotation_trend = str(theme_context.get("trend") or sector_trend or "")
    falling_knife = "Lagging" in rotation_trend
    if falling_knife:
        fk_penalty = _cfg_number("falling_knife_penalty", 8.0)
        headwind_adjust += fk_penalty
        risk_flags.append(
            f"Falling-knife risk: {ticker_sector} rotation is {rotation_trend.strip()} - dip-buy against sector momentum"
        )
        reasons.append("Risk: sector Lagging in rotation (falling-knife guard)")
        risk_adjustments.append({
            "type": "falling_knife",
            "points": round(fk_penalty, 1),
            "detail": f"Sector rotation {rotation_trend.strip()}",
        })

    portfolio_risk = _portfolio_fit_adjustment(symbol, ticker_sector, portfolio_context)
    if portfolio_risk.get("risk_flags"):
        risk_flags.extend(portfolio_risk["risk_flags"])
        reasons.extend(portfolio_risk.get("reasons", []))
        risk_adjustments.extend(portfolio_risk.get("adjustments", []))

    raw_risk_adjust = round(headwind_adjust + float(portfolio_risk.get("risk_adjust", 0.0) or 0.0), 1)
    risk_cap = _cfg_number("risk_cap", _M3_RISK_CAP)
    risk_adjust = min(raw_risk_adjust, risk_cap)
    if raw_risk_adjust > risk_adjust:
        risk_adjustments.append({
            "type": "risk_cap",
            "points": round(raw_risk_adjust - risk_adjust, 1),
            "detail": f"Risk adjustment capped at {risk_cap:.0f}",
        })

    pre_risk_score = (base_score + effective_flow_bonus) * entry_multiplier
    if base_score < _MIN_SCORE_THRESHOLD:
        # Confirmation proxies cannot rescue a weak additive base into top_picks.
        pre_risk_score = min(pre_risk_score, _MIN_SCORE_THRESHOLD - 1)
    final_score = max(0.0, pre_risk_score - risk_adjust)
    score = int(round(final_score))

    rec_raw = str(fund.get("recommendation", ""))
    analyst_bullish = "buy" in rec_raw.lower() or "strong" in rec_raw.lower()
    signal_cats = _count_signal_categories(reasons)
    opportunity_type = _classify_opportunity(
        reasons,
        tech,
        _safe_float(tech.get("drawdown_pct")) or 0.0,
        analyst_bullish,
        catalyst_labels=catalyst_labels,
        signal_categories=signal_cats,
    )

    conviction = _conviction_label(score)
    if len(risk_flags) >= 3 and conviction in ("Exceptional", "High Conviction"):
        conviction = "Qualified (Risk-Capped)"
    if falling_knife and conviction in ("Exceptional", "High Conviction"):
        conviction = "Qualified (Risk-Capped)"

    foundation_check = _assess_foundation_quality(fund)
    theme_name = theme_context.get("theme") or theme_context.get("sector")
    why_bits = []
    if theme_name:
        why_bits.append(f"{theme_name} theme")
    if rs_alpha >= 5:
        why_bits.append(f"+{rs_alpha:.1f}% 3M RS vs SPY")
    if p_forward >= 8:
        why_bits.append("forward re-rating/upside")
    if flow_confirmations:
        why_bits.append("flow-confirmed")
    if entry_stage == "extended":
        why_bits.append("entry demoted as extended")
    if falling_knife:
        why_bits.append("watchlist-only (sector Lagging - falling-knife guard)")
    why_now = " + ".join(why_bits) if why_bits else "Additive funnel signals are modest; monitor for confirmation"

    return {
        "symbol": symbol,
        "score": score,
        "base_score": base_score,
        "conviction": conviction,
        "opportunity_type": opportunity_type,
        "price": round(price, 2),
        "sector": ticker_sector,
        "reasons": reasons,
        "risk_flags": risk_flags,
        "risk_adjustments": risk_adjustments,
        "portfolio_fit": portfolio_risk.get("portfolio_fit", {}),
        "foundation_check": foundation_check,
        "thematic": _get_thematic_tags(symbol),
        "catalysts": catalyst_labels if catalyst_labels else [],
        "signal_convergence": signal_cats,
        "description": fund.get("description", ""),
        "recent_news": headlines,
        "theme": theme_name,
        "theme_score": theme_context.get("theme_score"),
        "theme_cycle_stage": theme_context.get("cycle_stage"),
        "theme_drivers": theme_context.get("drivers", []),
        "entry_stage": entry_stage,
        "entry_multiplier": entry_multiplier,
        # Structural stop so an actionable pick reaches MarketAnalyst with a defined
        # stop + risk-per-share %, rather than the LLM having to invent one (or write
        # "Data Unavailable"). Prefer the entry-setup gate (screener.check_setup), but
        # fall back to the identical basis computed in _compute_technicals_batch:
        # setup_data comes from _setup_check_parallel, which is gated on is_broad, so
        # on every themed scan the gate never runs and only the fallback is populated.
        "stop_loss": _safe_float(setup_data.get("stop_loss")) or _safe_float(tech.get("stop_loss")),
        "stop_basis": setup_data.get("stop_basis") or tech.get("stop_basis"),
        "risk_pct": _safe_float(setup_data.get("risk_pct")) or _safe_float(tech.get("risk_pct")),
        "flow_bonus": raw_flow_bonus,
        "effective_flow_bonus": effective_flow_bonus,
        "flow_confirmations": flow_confirmations,
        "flow_evidence": flow_evidence,
        "watchlist_only": falling_knife,
        "promotion_condition": (
            "Sector rotation flips to Improving/Leading, or sector RSI > 40 with positive 1M momentum"
            if falling_knife else None
        ),
        "why_now": why_now,
        "universe_provenance": provenance,
        "score_breakdown": {
            "theme": p_theme,
            "relstr": p_relstr,
            "forward": p_forward,
            "quality": p_quality,
            "base": base_score,
            "flow_bonus": raw_flow_bonus,
            "effective_flow_bonus": effective_flow_bonus,
            "entry_multiplier": entry_multiplier,
            "raw_risk_adjust": raw_risk_adjust,
            "risk_adjust": risk_adjust,
        },
        "score_details": component_details,
    }


def _get_sector_tickers(sector_key: str) -> list[str]:
    """Get dynamic tickers for a specific sector via TradingView / Yahoo screener."""
    tickers = _tv_sector_constituents(sector_key, limit=30)
    if tickers:
        return tickers
    return _yf_sector_constituents(sector_key, limit=30)


def _get_all_sector_names() -> list[str]:
    """List the canonical 11 GICS sector names."""
    return list(_SECTOR_TO_THEMES.keys())


def _get_thematic_map() -> dict[str, dict]:
    """Return thematic group definitions."""
    return {}

# Application-level fallbacks for ETFs and Mutual Funds that yfinance consistently fails to classify.
# This prevents the portfolio from reporting these as "Unknown".
_API_SECTOR_FALLBACKS = {
    "CASH": "Cash & Equivalents", "CASH.TO": "Cash & Equivalents",
    "SCHD": "Diversified Equity", "FBTC": "Crypto Assets",
    "QTUM": "Technology", "FTEC": "Technology", "FSCSX": "Technology", "TEC.TO": "Technology",
    "AAPL": "Technology", "MSFT": "Technology", "AMD": "Technology", "AMAT": "Technology",
    "MRVL": "Technology", "QCOM": "Technology", "U": "Technology", "BB": "Technology",
    "MU": "Technology", "WDC": "Technology", "STX": "Technology", "NVDA": "Technology",
    "AVGO": "Technology", "TSM": "Technology", "INTC": "Technology", "GFS": "Technology",
    "LRCX": "Technology", "KLAC": "Technology", "ASML": "Technology", "TER": "Technology",
    "LSPD.TO": "Technology", "XLK": "Technology", "QQQ": "Technology", "AUR": "Technology",
    "MSFT.TO": "Technology", "KWEB": "Technology", "SHOP.TO": "Technology", "ZSP.TO": "Technology",
    "FSEAX": "Emerging Markets",
    "EMXC": "Asset Allocation", "ESGD": "Asset Allocation", "NZAC": "Asset Allocation",
    "DSI": "Asset Allocation", "SPYX": "Asset Allocation", "XEN.TO": "Asset Allocation",
    "XESG.TO": "Asset Allocation", "VGRO.TO": "Asset Allocation", "VEQT.TO": "Asset Allocation",
    "AMZN": "Consumer Cyclical", "AEO": "Consumer Cyclical", "AMZN.TO": "Consumer Cyclical", "ABNB": "Consumer Cyclical",
    "HD": "Consumer Cyclical", "LOW": "Consumer Cyclical", "TSLA": "Consumer Cyclical",
    "PG": "Consumer Defensive", "COST": "Consumer Defensive", "WMT": "Consumer Defensive", "KO": "Consumer Defensive",
    "PYPL": "Financial Services", "SOFI": "Financial Services", "SLF.TO": "Financial Services", "XLF": "Financial Services",
    "VCN.TO": "Financial Services", "HXT.TO": "Financial Services", "ARKF": "Financial Services",
    "VDY.TO": "Financial Services", "CM.TO": "Financial Services", "RY.TO": "Financial Services",
    "JPM": "Financial Services", "BAC": "Financial Services", "GS": "Financial Services",
    "VET.TO": "Energy", "CHPT": "Clean Energy", "XLE": "Energy", "OVV.TO": "Energy", "ZCLN.TO": "Clean Energy",
    "XOM": "Energy", "CVX": "Energy", "COP": "Energy", "SLB": "Energy",
    "UNP": "Industrials", "XLI": "Industrials", "CAT": "Industrials", "DE": "Industrials",
    "VNQ": "Real Estate",
    "XLV": "Healthcare", "PFE": "Healthcare", "CRSP": "Healthcare", "AMN": "Healthcare", "MRNA": "Healthcare",
    "LLY": "Healthcare", "UNH": "Healthcare", "JNJ": "Healthcare", "ABBV": "Healthcare", "MRK": "Healthcare",

    "VTI": "Large Blend", "SPY": "Large Blend",
    "T.TO": "Communication Services",
    "XSAB.TO": "Fixed Income", "XSTB.TO": "Fixed Income", "HYXF": "Fixed Income"
}

def _is_plausible_ticker(symbol: str) -> bool:
    """
    Cheap guard before any network lookup: real exchange tickers are short and have
    no spaces. Private/manual holdings like 'ACME BALANCED 2040 FUND' are NOT tickers and
    must never be sent to yfinance (they 404, and the portfolio-fit loop would repeat
    that 404 once per candidate — observed as dozens of identical 404s in the logs).
    """
    s = (symbol or "").strip()
    if not s or " " in s:
        return False
    # Allow letters, digits, dot, hyphen (e.g. BRK.B, RY.TO); cap length.
    core = s.replace(".", "").replace("-", "")
    return core.isalnum() and len(s) <= 8


# Process-level memo for the symbol-only sector lookup (no fund). _portfolio_fit_adjustment
# calls this for every holding on every candidate (O(candidates × holdings)) — memoizing
# turns ~60 repeat lookups of the same holding into 1 network call.
_sector_lookup_memo: dict[str, str] = {}


def _get_sector_for_ticker(symbol: str, fund: dict | None = None) -> str:
    """
    Resolve the sector for a ticker using a 4-tier lookup:
      1. Hardcoded API fallbacks (for uncooperative ETFs)
      2. Universe file (static, fast)
      3. yfinance fundamentals dict (dynamic, already fetched during scoring, or fetched on the fly)
      4. Knowledge graph (persisted from past interactions)
    Falls back to 'Unknown' only if all miss.
    """
    symbol_upper = symbol.upper()

    # Tier 0: non-ticker guard (private/manual fund names) — never hit the network.
    if not _is_plausible_ticker(symbol):
        return "Private/Manual Holding"

    # Memoised result for the no-fund (portfolio-fit) path.
    if fund is None and symbol_upper in _sector_lookup_memo:
        return _sector_lookup_memo[symbol_upper]

    # Tier 1: Known API fallbacks
    if symbol_upper in _API_SECTOR_FALLBACKS:
        return _API_SECTOR_FALLBACKS[symbol_upper]

    # Tier 3: Knowledge graph (persisted sector from past portfolio syncs / interactions)
    try:
        from tools.graph_memory import graph_memory
        if graph_memory.graph.has_node(symbol):
            # First check for private asset sector
            node = graph_memory.graph.nodes[symbol]
            if "sector_breakdown" in node:
                # For funds, return the dominant sector or 'Balanced Fund'
                return "Balanced Fund"

            # Check explicit sector mapping
            for neighbor in graph_memory.graph.neighbors(symbol):
                edge_data = graph_memory.graph.get_edge_data(symbol, neighbor)
                if edge_data and "type" in edge_data and edge_data["type"] == "IN_SECTOR":
                    return neighbor
    except Exception:
        pass

    # Tier 4: yfinance sector from fundamentals (already fetched, or fetched on demand)
    if fund and isinstance(fund, dict):
        yf_sector = fund.get("sector_yf", "").strip()
        if yf_sector:
            return yf_sector
    else:
        try:
            from tools.yf_utils import get_info_safe
            info = get_info_safe(symbol) or {}
            yf_sector = info.get("sector", "").strip()
            if not yf_sector:
                yf_sector = info.get("category", "").strip()
            if not yf_sector:
                q_type = info.get("quoteType", "").lower()
                if "etf" in q_type or "mutualfund" in q_type:
                    yf_sector = "Diversified Fund"
                elif "cryptocurrency" in q_type:
                    yf_sector = "Crypto Assets"
            if yf_sector:
                _sector_lookup_memo[symbol_upper] = yf_sector
                return yf_sector
        except Exception:
            pass
        # Memoise the miss too, so a non-resolving symbol isn't re-queried per candidate.
        _sector_lookup_memo[symbol_upper] = "Unknown"

    return "Unknown"


def _get_thematic_tags(symbol: str) -> list[str]:
    """Return thematic groups a ticker belongs to."""
    themes = _get_thematic_map()
    tags = []
    for theme_name, theme_data in themes.items():
        if symbol in theme_data.get("tickers", []):
            tags.append(theme_name)
    return tags


# ---------------------------------------------------------------------------
# 1. BATCH TECHNICAL SCREEN
# ---------------------------------------------------------------------------

@log_exceptions()
def _batch_download(tickers: list[str], period: str = _BATCH_DOWNLOAD_PERIOD, max_retries: int = 3) -> "pd.DataFrame":
    """
    Download historical data for many tickers in a single API call with retries.
    `max_retries` is exposed so the per-ticker gap-filler can use 1 attempt — a
    delisted/no-data ticker (e.g. acquired names like PXD/SWN/TELL) will never
    recover, so retrying it 3× with backoff is pure latency + log spam.
    """
    import time

    import pandas as pd
    if not tickers:
        return pd.DataFrame()

    for attempt in range(max_retries):
        try:
            safe_print(f"📡 Batch downloading {len(tickers)} tickers ({period})... [Attempt {attempt+1}/{max_retries}]")
            data = yf.download(
                tickers,
                period=period,
                group_by="ticker",
                threads=False,
                progress=False,
                auto_adjust=True,
            )
            if not data.empty:
                safe_print(f"✅ Batch download complete: {len(data)} rows")
                return data

            # If data is empty but no exception was raised (rare for yf.download)
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
        except Exception as e:
            if attempt < max_retries - 1:
                safe_print(f"⚠️ Batch download attempt {attempt+1} failed: {e}. Retrying...")
                time.sleep(2 ** attempt)
            else:
                safe_print(f"❌ Batch download failed after {max_retries} attempts: {e}")

    return pd.DataFrame()


@log_exceptions()
def _compute_technicals_batch(data, tickers: list[str]) -> dict[str, dict[str, Any]]:
    """
    Compute technical indicators from batch-downloaded DataFrame.
    Returns {symbol: {rsi, sma50, sma200, golden_cross, drawdown, returns, vol_spike, price, ...}}
    """
    results = {}

    for symbol in tickers:
        if is_cancelled():
            break
        try:
            # Extract the ticker's OHLC/volume from multi-level DataFrame.
            # High/Low are pulled for the structural stop below — _batch_download
            # already fetches them, so the stop costs no extra network call.
            if len(tickers) == 1:
                close = data["Close"].dropna()
                volume = data["Volume"].dropna()
                high, low = data.get("High"), data.get("Low")
            else:
                if symbol not in data.columns.get_level_values(0):
                    safe_print(f"⚠️ {symbol}: Missing from batch download")
                    continue
                close = data[symbol]["Close"].dropna()
                volume = data[symbol]["Volume"].dropna()
                high, low = data[symbol].get("High"), data[symbol].get("Low")

            if close.empty:
                safe_print(f"⚠️ {symbol}: Close data is empty")
                continue
            if len(close) < 20:
                safe_print(f"⚠️ {symbol}: Insufficient history ({len(close)} days)")
                continue

            current_price = float(close.iloc[-1])
            if current_price <= 0 or np.isnan(current_price):
                continue

            # --- RSI-14 ---
            delta = close.diff()
            gain = delta.where(delta > 0, 0.0).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
            rs = gain / loss
            rsi_series = 100 - (100 / (1 + rs))
            rsi = float(rsi_series.iloc[-1]) if not np.isnan(rsi_series.iloc[-1]) else 50.0

            # --- Moving Averages ---
            sma50 = float(close.rolling(50).mean().iloc[-1]) if len(close) >= 50 else None
            sma200 = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else None

            # Golden Cross: SMA50 > SMA200 AND crossed within last 10 bars
            golden_cross = False
            if sma50 and sma200 and len(close) >= 200:
                sma50_series = close.rolling(50).mean()
                sma200_series = close.rolling(200).mean()
                diff_now = float(sma50_series.iloc[-1] - sma200_series.iloc[-1])
                diff_10 = float(sma50_series.iloc[-10] - sma200_series.iloc[-10]) if len(close) >= 210 else diff_now
                golden_cross = diff_now > 0 and diff_10 <= 0

            # Death Cross (reverse)
            death_cross = False
            if sma50 and sma200 and len(close) >= 200:
                diff_now = float(sma50_series.iloc[-1] - sma200_series.iloc[-1])
                diff_10 = float(sma50_series.iloc[-10] - sma200_series.iloc[-10]) if len(close) >= 210 else diff_now
                death_cross = diff_now < 0 and diff_10 >= 0

            # --- Drawdown from 6mo high ---
            high_6mo = float(close.max())
            drawdown_pct = ((current_price - high_6mo) / high_6mo) * 100 if high_6mo > 0 else 0

            # --- Returns ---
            week_return = ((current_price - float(close.iloc[-5])) / float(close.iloc[-5]) * 100) if len(close) >= 5 else 0
            month_return = ((current_price - float(close.iloc[-22])) / float(close.iloc[-22]) * 100) if len(close) >= 22 else 0
            three_month_return = ((current_price - float(close.iloc[-66])) / float(close.iloc[-66]) * 100) if len(close) >= 66 else 0

            # --- Volume Spike ---
            avg_vol = float(volume.mean()) if len(volume) > 10 else 1.0
            recent_vol = float(volume.tail(3).mean()) if len(volume) >= 3 else avg_vol
            vol_spike = recent_vol / avg_vol if avg_vol > 0 else 1.0

            # --- Uptrend ---
            above_sma50 = current_price > sma50 if sma50 else False

            # --- Structural stop (basis mirrors screener.check_setup exactly) ---
            # The lower of the 20-day swing low and a 2x-ATR(14) volatility stop,
            # computed from OHLC already in hand. Derived here rather than only in
            # check_setup because that runs solely on BROAD scans, leaving every
            # themed scan with no stop — the market_dip lens then demands one per
            # pick, the model invents a round number, and judge Rule 4 flags it as
            # unanchored. A stop is also what tactical position sizing divides by,
            # so a fabricated one silently mis-sizes the trade.
            stop_loss = stop_basis = risk_pct = None
            try:
                if high is not None and low is not None:
                    high_s, low_s = high.dropna(), low.dropna()
                    if len(high_s) >= 20 and len(low_s) >= 20:
                        prev_close = close.shift()
                        true_range = (
                            (high_s - low_s).to_frame("hl")
                            .join((high_s - prev_close).abs().to_frame("hc"))
                            .join((low_s - prev_close).abs().to_frame("lc"))
                            .max(axis=1)
                        )
                        atr_14 = float(true_range.rolling(window=14).mean().iloc[-1])
                        swing_low_20 = float(low_s.tail(20).min())
                        if not np.isnan(atr_14):
                            candidate = min(current_price - 2 * atr_14, swing_low_20)
                            # On a fresh breakout the ATR stop can land above price;
                            # fall back to the swing low, then a conservative floor.
                            if not (candidate < current_price):
                                candidate = swing_low_20 if swing_low_20 < current_price else current_price * 0.92
                            if 0 < candidate < current_price:
                                stop_loss = round(candidate, 2)
                                stop_basis = f"lower of 20d swing low and 2x ATR (${atr_14:.2f})"
                                risk_pct = round((current_price - stop_loss) / current_price * 100, 1)
            except Exception:
                stop_loss = stop_basis = risk_pct = None

            results[symbol] = {
                "stop_loss": stop_loss,
                "stop_basis": stop_basis,
                "risk_pct": risk_pct,
                "price": round(current_price, 2),
                "rsi": round(rsi, 1),
                "sma50": round(sma50, 2) if sma50 else None,
                "sma200": round(sma200, 2) if sma200 else None,
                "golden_cross": golden_cross,
                "death_cross": death_cross,
                "above_sma50": above_sma50,
                "drawdown_pct": round(drawdown_pct, 1),
                "high_6mo": round(high_6mo, 2),
                "week_return": round(week_return, 1),
                "month_return": round(month_return, 1),
                "three_month_return": round(three_month_return, 1),
                "vol_spike": round(vol_spike, 2),
            }
        except Exception:
            # Silently skip tickers with bad data
            pass

    return results


# Daily-cache key prefix for per-symbol fast-screen technicals.
# Bump this whenever the tech dict gains a field a caller reads: entries cached
# earlier the same day are otherwise served under the old schema, and a missing
# field is indistinguishable from a computed None. v2 adds stop_loss/stop_basis/
# risk_pct, without which every themed scan would keep rendering "—" until the
# cache rolled over at midnight.
_TECH_CACHE_PREFIX = "funnel_tech_v2_"


def _compute_technicals_cached(candidates: list[str]) -> dict[str, dict[str, Any]]:
    """Per-symbol daily-cached wrapper around the batch technical screen.

    Splits candidates into names already screened earlier today (served from the
    daily file cache) and the uncached remainder, which are downloaded + computed
    once and then persisted. Freezing each symbol's technicals for the trading day
    keeps the fast-screen top-N cut stable across intraday re-runs — without it the
    cut drifts as live prices move — and matches the daily cadence the deep signals
    already use. Also avoids re-downloading names seen in an earlier run.
    """
    import tools.daily_cache as daily_cache

    technicals: dict[str, dict[str, Any]] = {}
    to_fetch: list[str] = []
    for sym in candidates:
        hit = daily_cache.get_cached(f"{_TECH_CACHE_PREFIX}{sym}")
        if isinstance(hit, dict) and hit:
            technicals[sym] = hit
        else:
            to_fetch.append(sym)

    if to_fetch:
        batch_data = (
            _batch_download_chunked(to_fetch)
            if len(to_fetch) > _DOWNLOAD_CHUNK_SIZE
            else _batch_download(to_fetch)
        )
        if not batch_data.empty:
            fresh = _compute_technicals_batch(batch_data, to_fetch)
            for sym, tech in fresh.items():
                daily_cache.set_cached(f"{_TECH_CACHE_PREFIX}{sym}", tech)
                technicals[sym] = tech

    return technicals


@log_exceptions()
def _fast_score(symbol: str, technicals: dict[str, Any], provenance: list[str] | None = None, rs_alpha: float = 0.0) -> int:
    """Quick score based on batch-computed technicals, plus a baseline bump for quality/thematic leaders. Range ~0-100."""
    score = 0

    # RSI signals
    rsi = technicals.get("rsi", 50)
    if rsi < 30:
        score += 15   # Oversold — dip-buy opportunity
    elif rsi < 40:
        score += 8
    elif rsi > 75:
        score -= 5    # Overbought risk

    # Trend
    if technicals.get("golden_cross"):
        score += 20
    elif technicals.get("above_sma50"):
        score += 8
    if technicals.get("death_cross"):
        score -= 10

    # Drawdown (value opportunity)
    dd = technicals.get("drawdown_pct", 0)
    if dd < -30:
        score += 12   # Deep discount
    elif dd < -15:
        score += 8    # Moderate pullback

    # Momentum
    mr = technicals.get("month_return", 0)
    if mr > 15:
        score += 12
    elif mr > 5:
        score += 6
    elif mr < -15:
        score += 4    # Bounce candidate

    # Volume confirmation
    vs = technicals.get("vol_spike", 1)
    if vs > 2.0:
        score += 10
    elif vs > 1.5:
        score += 5

    # Mega-cap & Thematic baseline bump (ensures high-quality fundamental stocks pass the technical filter)
    tags = _get_thematic_tags(symbol)
    if tags:
        score += 15

    # Check if it's a known liquid baseline ticker
    if symbol.upper() in _API_SECTOR_FALLBACKS or symbol.upper() in _SEMI_SEED:
        score += 10

    # Relative-strength alpha bonus (Funnel V2 M2) — RS vs SPY over 3 months.
    # rs_alpha is in percentage points (e.g. +12.5 means +12.5% above SPY return).
    # This is a pre-filter signal only; the full P_relstr pillar comes in M3.
    if rs_alpha > 10:
        score += 8    # Material outperformance vs market
    elif rs_alpha > 5:
        score += 4
    elif rs_alpha < -15:
        score -= 5    # Meaningfully lagging — needs a strong reason to advance

    # Provenance boost (Funnel V2 M1): multi-source themed names must survive
    # the fast-screen cut so they advance to deep scoring.
    # Without this, a cyclical at its base (low RSI bonus, no golden-cross yet)
    # would be culled before deep fundamentals run — the exact MU failure mode.
    if provenance:
        source_count = len(provenance)
        has_theme = any(s.startswith("theme:") for s in provenance)
        has_live_sector = any(s.startswith("sector:") for s in provenance)  # FMP, not static

        if source_count >= 3:
            score += 20   # Theme + inflowing sector + mover/guru — strong multi-signal
        elif source_count == 2:
            score += 12   # Two independent sources agree
        elif source_count == 1:
            score += 4    # Weak signal; bump just enough to not penalise single-source

        if has_theme:
            score += 8    # Geo/policy event producer (e.g. memory_chips → MU)
        if has_live_sector and not has_theme:
            score += 4    # In a confirmed-inflowing sector (FMP screener)

    return max(score, 0)


def _is_accumulation_like(tech: dict[str, Any]) -> bool:
    """Cheap fast-screen-stage proxy for the funnel's ideal entry archetype.

    Low/neutral RSI plus a pullback (below the 50DMA or meaningfully off the
    6-month high) = basing/accumulation, NOT a momentum chase. Used only for
    near-miss instrumentation: the deep-dive entry-stage classifier
    (_entry_stage_from_setup) is the real gate and additionally sees the BB
    setup + flow, which aren't computed at cut time.
    """
    rsi = tech.get("rsi", 50) or 50
    drawdown = tech.get("drawdown_pct", 0) or 0
    above_sma50 = tech.get("above_sma50", False)
    return rsi < 55 and (not above_sma50 or drawdown <= -10)


def _guru_feed_available() -> bool:
    """Whether the optional guru/media-sentiment feed is part of this build.

    Returns False in the public release where tools.guru_feed is stripped.
    Surfaced into the scan result as `guru_enabled` so the LLM can tell a
    feature that's switched off (don't penalize confidence) from a genuine
    per-ticker data gap (guru present but no hit for that name).
    """
    try:
        import tools.guru_feed  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# 2. DEEP FUNDAMENTALS (Individual calls, only for top candidates)
# ---------------------------------------------------------------------------

@cached(ttl=3600)
@log_exceptions()
def _fetch_fundamentals(symbol: str) -> dict[str, Any]:
    """Fetch fundamental data AND recent news for a single stock via yfinance.

    Cached because this was the only per-symbol fetcher in the scanner that was
    not: every sibling carries @cached, while this one re-issued `.info` + `.news`
    for all ~60 Phase-3 candidates on every scan (~15-30s, paid again on each
    re-scan).

    TTL is an hour rather than a day because the payload carries `current_price`,
    `52_week_high` and `50_day_ma` alongside the slow-moving fundamentals — a
    daily cache would score a re-scan against yesterday's prices. The decorator
    stamps `_as_of` on the MISS, so a hit reports the original fetch time and the
    staleness stays visible downstream rather than being laundered.

    The total-failure return below carries an `error` key deliberately: @cached
    refuses to store dicts containing "error", so a symbol that failed all three
    attempts is retried on the next scan instead of having its own failure cached
    for an hour.
    """
    max_retries = 3
    for attempt in range(max_retries):
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info

            # Grab latest news headlines (same API session, negligible extra cost)
            headlines = []
            try:
                raw_news = ticker.news
                if raw_news:
                    for n in raw_news[:5]:
                        content = n.get('content', n)
                        title = content.get('title') or content.get('headline') or ""
                        if title:
                            headlines.append(title)
            except Exception:
                pass

            return {
                "symbol": symbol,
                "peg_ratio": info.get('pegRatio'),
                "profit_margin": info.get('profitMargins'),
                "gross_margin": info.get('grossMargins'),
                "analyst_target": info.get('targetMeanPrice'),
                "revenue_growth": info.get('revenueGrowth'),
                "earnings_growth": info.get('earningsGrowth'),
                "trailing_pe": info.get('trailingPE'),
                "forward_pe": info.get('forwardPE'),
                "beta": info.get('beta'),
                "free_cashflow": info.get('freeCashflow'),
                "operating_cashflow": info.get('operatingCashflow'),
                "total_debt": info.get('totalDebt'),
                "total_cash": info.get('totalCash'),
                "ebitda": info.get('ebitda'),
                "recommendation": info.get('recommendationKey'),
                "current_price": info.get('currentPrice'),
                "52_week_high": info.get('fiftyTwoWeekHigh'),
                "50_day_ma": info.get('fiftyDayAverage'),
                "description": (info.get('longBusinessSummary') or "")[:120] + "...",
                "sector_yf": info.get('sector', ''),
                "industry": info.get('industry', ''),
                "news_headlines": headlines,
            }
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff: 1s, 2s
            else:
                safe_print(f"⚠️ Failed to fetch fundamentals for {symbol} after {max_retries} attempts: {e}")
                return {"symbol": symbol, "error": f"fundamentals unavailable after {max_retries} attempts"}


def _fetch_fundamentals_parallel(symbols: list[str], max_workers: int = 4) -> dict[str, dict]:
    """Fetch fundamentals for multiple symbols in parallel."""
    from tools.cache import fetch_parallel
    tasks = {sym: (lambda s=sym: _fetch_fundamentals(s)) for sym in symbols}
    return fetch_parallel(tasks, timeout=60, max_workers=max_workers)


def _warm_cache_parallel(fn, symbols: list[str], max_workers: int = 8, overall_budget: float = 30.0) -> None:
    """
    Pre-populate a per-symbol @cached function's cache in parallel so that a later
    SEQUENTIAL loop (e.g. the Phase-4 scoring loop) hits the cache instead of making
    one blocking network call per symbol.

    Bounded by `overall_budget` SECONDS for the WHOLE batch — not per call. This is
    the critical property: yfinance calls can hang indefinitely (observed ~1000s), so
    a per-future timeout in a sequential loop accumulates (N × timeout). We instead
    process completions as they arrive and hard-stop at the budget, cancelling the
    rest. Best-effort: stragglers stay cold (the in-loop call recomputes that one).
    """
    if not symbols:
        return
    from concurrent.futures import as_completed
    deadline = time.perf_counter() + max(1.0, overall_budget)
    _prof = get_active_profile()
    executor = ThreadPoolExecutor(max_workers=min(len(symbols), max_workers))
    try:
        futures = [executor.submit(run_under_profile, _prof, fn, sym) for sym in symbols]
        try:
            for fut in as_completed(futures, timeout=overall_budget):
                try:
                    fut.result()
                except Exception:
                    pass
                if is_cancelled() or time.perf_counter() >= deadline:
                    break
        except Exception:
            pass  # as_completed raised TimeoutError once the overall budget elapsed
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


# ---------------------------------------------------------------------------
# 3. DEEP SCORING (Full rubric — same signals as before + new outlook)
# ---------------------------------------------------------------------------

@log_exceptions()
def _conviction_label(score: int) -> str:
    tiers = _cfg_section("tiers")
    if score >= float(tiers.get("exceptional", 100)):
        return "Exceptional"
    if score >= float(tiers.get("high", 80)):
        return "High Conviction"
    if score >= float(tiers.get("qualified", 60)):
        return "Qualified"
    if score >= float(tiers.get("watchlist", 40)):
        return "Watchlist"
    return "Low Interest"


# ---------------------------------------------------------------------------
# NEWS CATALYST DETECTION
# ---------------------------------------------------------------------------

# Bullish catalysts — events that historically trigger outsized price moves
_BULLISH_CATALYSTS = [
    # Regulatory/Approval
    ("fda approv", 20, "FDA Approval"),
    ("fda clear", 15, "FDA Clearance"),
    ("regulatory approv", 15, "Regulatory Approval"),
    ("granted patent", 10, "Patent Grant"),
    # Financial
    ("beats estimate", 15, "Earnings Beat"),
    ("beats expectations", 15, "Earnings Beat"),
    ("record revenue", 12, "Record Revenue"),
    ("record earnings", 12, "Record Earnings"),
    ("raised guidance", 15, "Raised Guidance"),
    ("raises guidance", 15, "Raised Guidance"),
    ("raises outlook", 15, "Raised Outlook"),
    ("upgraded to buy", 12, "Analyst Upgrade"),
    ("upgrade", 8, "Analyst Upgrade"),
    ("price target raise", 10, "Target Raised"),
    ("target raised", 10, "Target Raised"),
    ("buyback", 8, "Share Buyback"),
    ("stock split", 8, "Stock Split"),
    ("dividend increase", 8, "Dividend Hike"),
    ("special dividend", 10, "Special Dividend"),
    # Strategic
    ("partnership with", 10, "Strategic Partnership"),
    ("partners with", 10, "Strategic Partnership"),
    ("contract win", 12, "Contract Win"),
    ("contract award", 12, "Contract Award"),
    ("wins contract", 12, "Contract Win"),
    ("awarded contract", 12, "Contract Award"),
    ("billion dollar deal", 15, "Mega Deal"),
    ("$b deal", 12, "Mega Deal"),
    ("acquisition of", 10, "Acquisition"),
    ("acquires", 10, "Acquisition"),
    ("expansion into", 8, "Market Expansion"),
    # Government/Policy
    ("government contract", 12, "Govt Contract"),
    ("defense contract", 12, "Defense Contract"),
    ("subsidy", 8, "Subsidy/Grant"),
    ("tax credit", 8, "Tax Credit"),
    ("included in", 8, "Index Inclusion"),
    ("added to s&p", 15, "S&P 500 Inclusion"),
    ("added to the s&p", 15, "S&P 500 Inclusion"),
    # Innovation
    ("breakthrough", 10, "Breakthrough"),
    ("first-ever", 10, "Innovation"),
    ("record-breaking", 8, "Record Performance"),
    ("ai integration", 8, "AI Adoption"),
    ("launch", 6, "Product Launch"),
    ("new product", 6, "Product Launch"),
]

_BEARISH_CATALYSTS = [
    # Narrative & Disruption Risks
    ("ai competition", -20, "AI Competition Threat"),
    ("ai disruption", -20, "AI Disruption Risk"),
    ("ai headwind", -15, "AI Headwind"),
    ("threat from ai", -20, "AI Threat"),

    # Structural & Macro Headwinds
    ("margin pressure", -15, "Margin Pressure"),
    ("margin compression", -15, "Margin Compression"),
    ("market share loss", -20, "Market Share Loss"),
    ("losing market share", -20, "Market Share Loss"),
    ("antitrust", -15, "Antitrust Risk"),
    ("price war", -15, "Price War"),
    ("slashing prices", -10, "Pricing Pressure"),
    ("demand slowdown", -15, "Demand Slowdown"),
    ("inventory glut", -12, "Inventory Glut"),
    ("short report", -20, "Short Seller Target"),

    # Financial & Analyst
    ("downgrade", -12, "Analyst Downgrade"),
    ("downgraded", -12, "Analyst Downgrade"),
    ("misses estimate", -12, "Earnings Miss"),
    ("misses expectations", -12, "Earnings Miss"),
    ("guidance cut", -15, "Guidance Cut"),
    ("lowers guidance", -15, "Guidance Cut"),
    ("lowers outlook", -15, "Outlook Cut"),
    ("cuts guidance", -15, "Guidance Cut"),
    ("disappointing", -8, "Disappointment"),
    ("lawsuit", -10, "Lawsuit"),
    ("sued by", -10, "Lawsuit"),
    ("sec investigation", -15, "SEC Investigation"),
    ("investigation", -8, "Investigation"),
    ("fraud", -15, "Fraud Allegation"),
    ("recall", -10, "Product Recall"),
    ("data breach", -12, "Data Breach"),
    ("layoff", -6, "Layoffs"),
    ("ceo resign", -10, "CEO Departure"),
    ("ceo depart", -10, "CEO Departure"),
    ("bankruptcy", -20, "Bankruptcy Risk"),
    ("debt downgrade", -12, "Credit Downgrade"),
    ("tariff", -6, "Tariff Risk"),
    ("ban", -8, "Regulatory Ban"),
    ("removed from", -10, "Index Removal"),
]


@log_exceptions()
def _score_news_catalysts(headlines: list[str]) -> tuple[int, list[str], list[str]]:
    """
    Scan news headlines for bullish/bearish catalysts.
    Returns: (score_delta, reason_strings, catalyst_labels)
    """
    if not headlines:
        return 0, [], []

    total_score = 0
    reasons = []
    labels = []
    seen_labels = set()  # Prevent double-counting same catalyst type

    for headline in headlines:
        hl = headline.lower()

        # Check bullish catalysts
        for keyword, points, label in _BULLISH_CATALYSTS:
            if keyword in hl and label not in seen_labels:
                total_score += points
                reasons.append(f"📰 {label}: \"{headline[:70]}\"")
                labels.append(label)
                seen_labels.add(label)
                break  # One catalyst per headline

        # Check bearish catalysts
        for keyword, points, label in _BEARISH_CATALYSTS:
            if keyword in hl and label not in seen_labels:
                total_score += points  # points are already negative
                reasons.append(f"⚠️ {label}: \"{headline[:70]}\"")
                labels.append(label)
                seen_labels.add(label)
                break

    # Cap to prevent single news event from dominating
    total_score = max(min(total_score, 30), -25)
    return total_score, reasons, labels


def _count_signal_categories(reasons: list[str]) -> int:
    """
    Count how many distinct signal categories fired.
    Used for Golden Opportunity convergence detection.
    """
    categories = set()
    for r in reasons:
        if any(x in r for x in ["Sector", "Macro"]):
            categories.add("macro")
        elif any(x in r for x in ["Growth", "Margin", "Earnings Accel", "Compounder", "PEG", "Cheap", "Discount", "Pullback", "Correction", "Analyst Upside"]):
            categories.add("fundamental")
        elif any(x in r for x in ["GOLDEN CROSS", "Oversold", "Death Cross", "Volume Spike", "Momentum"]):
            categories.add("technical")
        elif any(x in r for x in ["Thematic", "Insider"]):
            categories.add("alpha")
        elif any(x in r for x in ["📰", "FDA", "Contract", "Beat", "Upgrade", "Partnership", "Guidance", "Launch", "Breakthrough"]):
            categories.add("catalyst")
    return len(categories)


@log_exceptions()
def _classify_opportunity(reasons: list[str], technicals: dict, drawdown_pct: float,
                          is_buy: bool, catalyst_labels: list[str] = None,
                          signal_categories: int = 0) -> str:
    """Assign a human-friendly opportunity type. Prioritises Golden Opportunity convergence."""
    has_golden_cross = technicals.get("golden_cross", False)
    has_oversold = technicals.get("rsi", 50) < 30
    month_ret = technicals.get("month_return", 0)

    # 🔥 GOLDEN OPPORTUNITY — convergence of 3+ signal categories
    if signal_categories >= 3:
        return "🔥 Golden Opportunity"
    # High-impact catalyst + technical/fundamental support
    if catalyst_labels and len(catalyst_labels) >= 2 and signal_categories >= 2:
        return "🔥 Golden Opportunity"

    if drawdown_pct < -20 and is_buy:
        return "🏷️ On Sale"
    # Catalyst-driven (single strong catalyst)
    if catalyst_labels and any(c in catalyst_labels for c in [
        "FDA Approval", "Earnings Beat", "Raised Guidance", "Contract Win",
        "S&P 500 Inclusion", "Mega Deal", "Breakthrough"
    ]):
        return "📰 Catalyst Play"
    if has_golden_cross:
        return "🌟 Golden Cross"
    if has_oversold and is_buy:
        return "📉 Dip Buy"
    if any("Thematic" in r for r in reasons):
        return "🔮 Thematic Play"
    if any("Compounder" in r for r in reasons):
        return "👑 Quality Compounder"
    if month_ret > 15 and technicals.get("vol_spike", 1) > 1.5:
        return "🚀 Momentum Surge"
    return "📊 Opportunity"


@log_exceptions()
def _deep_score_value_v1(symbol: str, fund: dict, tech: dict, sector_trend: str,
                         macro_context: dict, thematic_tags: list[str], portfolio_context: dict | None = None, **kwargs) -> dict | None:
    """
    Full scoring rubric applied to one stock — the ORIGINAL, value-led rubric.

    There are two live scorers, and which one runs is decided by `is_broad`:

        broad scans           -> _deep_score_v2   (Funnel V2, top-down, RS/theme aware)
        named-sector + guru   -> this one

    That fork is deliberate as of 2026-07-31 and named accordingly, because
    reading `_deep_score` next to `_deep_score_v2` invited the assumption that
    this one was simply stale. It is not: it is what a sector or guru scan is
    still scored on today.

    Whether to retire it is a PRODUCT decision, not a cleanup, and it is not one
    to make from the code alone — it changes which names a sector scan surfaces,
    i.e. what the user is shown as a candidate. Settle it with the funnel
    backtest (tests/test_funnel_backtest.py), by scoring the same sector-scan
    universe both ways and comparing realised alpha by tier. Retire v1 only if v2
    wins on the sector path the way it was shown to on the broad path; the
    existing backtest result ("only High Conviction beats SPY") is about the
    broad funnel and does not transfer.

    `fund` = fundamentals from yfinance .info
    `tech` = technicals from batch compute
    `kwargs` may contain 'headwind_data' dict from _headwind_check()
    """
    score = 0
    reasons = []

    def get_float(key):
        val = fund.get(key)
        if val is not None:
            try:
                return float(val)
            except (ValueError, TypeError):
                pass
        return None

    price = tech.get("price") or get_float("current_price")
    if not price or price <= 0:
        return None

    # --- A. SECTOR MOMENTUM ---
    if "Leading" in sector_trend:
        score += 15
        reasons.append(f"🌊 Sector Leading ({sector_trend})")
    elif "Improving" in sector_trend:
        score += 10
        reasons.append(f"🔄 Sector Improving ({sector_trend})")
    elif "Weakening" in sector_trend:
        score -= 5
        reasons.append(f"⚠️ Sector Weakening ({sector_trend})")
    elif "Lagging" in sector_trend:
        # Falling-knife guard: worst rotation tier must penalize at least as
        # hard as Weakening, or cheap names in the worst sector screen clean.
        score -= 10
        reasons.append(f"🔻 Sector Lagging ({sector_trend}) - falling-knife guard")

    # --- A.5. MACRO ALIGNMENT ---
    if macro_context:
        if macro_context.get("is_macro_favored"):
            score += 20
            reasons.append("🌍 Strong Macro Tailwind")
        elif macro_context.get("is_macro_disfavored"):
            score -= 15
            reasons.append("⚠️ Macro Headwind")

    # --- A.6. CONCENTRATION PENALTY ---
    if portfolio_context and "holdings" in portfolio_context:
        # Sum up value_usd for all holdings
        total_usd = sum((h.get("value_usd") or 0) for h in portfolio_context["holdings"])
        if total_usd > 0:
            ticker_sector = _get_sector_for_ticker(symbol, fund=fund)
            if ticker_sector != "Unknown":
                sector_exposure_usd = 0
                for h in portfolio_context["holdings"]:
                    h_sym = h.get("symbol", "")
                    if _get_sector_for_ticker(h_sym) == ticker_sector:
                        sector_exposure_usd += (h.get("value_usd") or 0)

                exposure_pct = sector_exposure_usd / total_usd
                if exposure_pct > 0.25: # Overweight > 25%
                    penalty = int((exposure_pct - 0.25) * 100) * 2 # Steep penalty, e.g., 45% -> 40 points
                    score -= penalty
                    reasons.append(f"⚠️ Overweight Penalty (-{penalty}): Portfolio is {exposure_pct*100:.0f}% {ticker_sector}")

    # --- B. FUNDAMENTALS ---
    rev_growth = get_float("revenue_growth")
    earn_growth = get_float("earnings_growth")
    margin = get_float("profit_margin")
    fwd_pe = get_float("forward_pe")
    trail_pe = get_float("trailing_pe")
    peg = get_float("peg_ratio")
    target = get_float("analyst_target")
    get_float("52_week_high")
    beta = get_float("beta")
    rec_raw = fund.get("recommendation", "")
    is_buy = "buy" in str(rec_raw).lower() or "strong" in str(rec_raw).lower()
    foundation_check = _assess_foundation_quality(fund)
    preliminary_risk_flags = []

    # DATA QUALITY CHECK: Fast fail if we are missing extensive fundamental data
    # (prevents recommending stocks if the API drops or fails to return their core data)
    missing_data = sum(x is None for x in [rev_growth, earn_growth, margin, fwd_pe, trail_pe, peg, target])
    if missing_data >= 5:
        return {"symbol": symbol, "error": "insufficient_data"}

    # Earnings acceleration
    if rev_growth is not None and earn_growth is not None and earn_growth > (rev_growth + 0.05) and earn_growth > 0:
        score += 10
        reasons.append("📈 Earnings Acceleration (Margin Expansion)")

    # Revenue growth
    if rev_growth is not None:
        if rev_growth > 0.10:
            expanding_liq = macro_context and macro_context.get("liquidity") == "Expanding"
            if expanding_liq:
                score += 15
                reasons.append(f"🚀 High Growth with Expanding Liquidity ({rev_growth:.0%})")
            else:
                score += 10
                reasons.append(f"🚀 High Growth ({rev_growth:.0%})")
        elif rev_growth > 0:
            score += 5

    # Profit margin
    if margin is not None and margin > 0.15:
        score += 10
        reasons.append(f"💰 Strong Margins ({margin:.0%})")

    # Quality Compounder
    if margin and margin > 0.20 and tech.get("above_sma50"):
        score += 20
        reasons.append("👑 Quality Compounder (High Margins + Uptrend)")

    # Drawdown / On Sale
    dd = tech.get("drawdown_pct", 0)
    if dd < -20 and is_buy:
        if foundation_check.get("grade") == "Strong":
            score += 20
            reasons.append("📉 20%+ Pullback on Strong Fundamentals")
        else:
            score += 8
            reasons.append("📉 20%+ Pullback (Foundation Not Fully Proven)")
            preliminary_risk_flags.append(
                "⚠️ Pullback is not a proven quality discount — cash-flow, margin, or balance-sheet evidence is incomplete"
            )
        if beta and beta > 1.5:
            preliminary_risk_flags.append(
                f"⚠️ High beta ({beta:.2f}) — a 20% drawdown may be normal volatility rather than structural mispricing"
            )
    elif dd < -10 and is_buy:
        score += 10
        reasons.append("📉 10%+ Correction")

    # GARP (Growth At Reasonable Price) — must have positive PE (profitable)
    if fwd_pe and rev_growth and 0 < fwd_pe < 30 and rev_growth > 0.15:
        score += 20
        reasons.append(f"💎 Cheap for Growth (PE {fwd_pe:.1f})")

    # PEG ratio
    if peg is not None and 0 < peg < 1.5:
        score += 15
        reasons.append(f"✨ Excellent Value (PEG {peg:.2f})")

    # Analyst upside
    if target and price and price > 0:
        upside = ((target - price) / price) * 100
        if upside > 30:
            score += 15
            reasons.append(f"🎯 Analyst Upside (+{upside:.0f}%)")
        elif upside > 15:
            score += 10
            reasons.append(f"🎯 Analyst Upside (+{upside:.0f}%)")

    # Macro risk deduction
    if macro_context and (macro_context.get("crash_risk") == "High" or "INVERTED" in (macro_context.get("curve_status") or "")):
        if (fwd_pe and fwd_pe > 35) or (not margin) or (margin and margin < 0.05):
            score -= 15
            reasons.append("⚠️ High Risk Profile in Deteriorating Macro")

    # --- C. TECHNICALS (from batch data) ---
    if tech.get("golden_cross"):
        score += 15
        reasons.append("🌟 GOLDEN CROSS")
    elif tech.get("above_sma50"):
        score += 5

    if tech.get("rsi", 50) < 30:
        score += 15
        reasons.append(f"🟢 Oversold (RSI {tech['rsi']:.0f})")

    if tech.get("death_cross"):
        score -= 10
        reasons.append("💀 Death Cross (Bearish)")

    if tech.get("vol_spike", 1) > 2.0:
        score += 8
        reasons.append(f"🔥 Volume Spike ({tech['vol_spike']:.1f}x)")

    # --- D. FORWARD OUTLOOK (new) ---
    # D1. Analyst revision momentum: estimates going UP (requires positive PE)
    if fwd_pe and trail_pe and fwd_pe > 0 and trail_pe > 0 and fwd_pe < trail_pe * 0.85:
        score += 12
        reasons.append(f"📊 Analyst Estimates Rising (Forward PE {fwd_pe:.1f} vs Trailing {trail_pe:.1f})")

    # D2. Thematic tailwind
    if thematic_tags:
        bonus = min(len(thematic_tags) * 8, 15)
        score += bonus
        tag_str = ", ".join(thematic_tags[:2])
        reasons.append(f"🔮 Thematic Tailwind ({tag_str})")

    # D3. Momentum continuation (3-month trend)
    tmr = tech.get("three_month_return", 0)
    mr = tech.get("month_return", 0)
    if tmr > 20 and mr > 5:
        score += 10
        reasons.append(f"📈 Sustained Momentum ({tmr:.0f}% / 3mo)")
    elif tmr > 10 and mr > 0:
        score += 5

    # --- E. DEEP ALPHA (optional, fast-fail) ---
    # PERF: this is a per-name network call (yfinance .info). In the Phase-4 batch
    # loop (~60 candidates) it was the dominant cost (~100-200s, sequential). The
    # batch loop passes fetch_insider=False; finalists still get insider signals in
    # Phase 5 (headwind gate, parallel + cached). Single-stock callers keep it on.
    if kwargs.get("fetch_insider", True):
        try:
            from tools.insider_data import get_insider_and_short_data
            insider = get_insider_and_short_data(symbol)
            if insider and insider.get("insider_signal") == "🟢 Insiders BUYING recently":
                score += 15
                reasons.append("🕵️ Insider Cluster Buying")
            elif insider and insider.get("insider_signal") == "🔴 Insiders SELLING recently":
                score -= 12
                reasons.append("🔴 Insider Selling Detected")
        except Exception:
            pass

    # --- F. NEWS CATALYST SCORING ---
    headlines = fund.get("news_headlines", [])
    catalyst_score, catalyst_reasons, catalyst_labels = _score_news_catalysts(headlines)
    if catalyst_score != 0:
        score += catalyst_score
        reasons.extend(catalyst_reasons)

    # --- G. HEADWIND RISK FLAGS (from pre-fetched data) ---
    risk_flags = list(dict.fromkeys(preliminary_risk_flags))
    headwind_data = kwargs.get("headwind_data", {})
    if headwind_data:
        # Short interest
        short_pct = headwind_data.get("short_pct_float")
        if short_pct is not None:
            if short_pct > 0.15:
                score -= 15
                risk_flags.append(f"⚠️ {short_pct*100:.0f}% Short Float — Elevated bearish/covering-risk signal")
                reasons.append(f"⚠️ Heavy Short Interest ({short_pct*100:.0f}%)")
            elif short_pct > 0.10:
                score -= 8
                risk_flags.append(f"🟡 {short_pct*100:.0f}% Short Float — Above-average bearish bets")
                reasons.append(f"🟡 Elevated Short Interest ({short_pct*100:.0f}%)")

        # Management tone
        tone = headwind_data.get("management_tone")
        if tone == "Highly Cautious (Bearish)":
            score -= 10
            risk_flags.append("⚠️ Bearish Management Tone — Cautious language in last earnings call")
            reasons.append("⚠️ Bearish Management Tone")

        # Earnings proximity
        days_to_earn = headwind_data.get("days_to_earnings")
        if days_to_earn is not None and 0 <= days_to_earn <= 7:
            rsi = tech.get("rsi", 50)
            if rsi > 65:
                score -= 8
                risk_flags.append(f"📅 Earnings in {days_to_earn} days while RSI overbought ({rsi:.0f}) — High event risk")
                reasons.append(f"📅 Earnings Imminent ({days_to_earn}d) + Overbought")
            else:
                risk_flags.append(f"📅 Earnings in {days_to_earn} days — Event volatility expected")

        # Insider selling (from headwind check — supplements the _deep_alpha check above)
        insider_signal = headwind_data.get("insider_signal")
        if insider_signal and "SELLING" in insider_signal:
            # Only add risk flag if not already penalised in section E
            if not any("Insider Selling" in r for r in reasons):
                score -= 12
                risk_flags.append("🔴 Insiders SELLING — Net insider selling in last 90 days")
                reasons.append("🔴 Insider Selling Detected")

    # --- CLASSIFY (with convergence detection) ---
    signal_cats = _count_signal_categories(reasons)
    opportunity_type = _classify_opportunity(
        reasons, tech, dd, is_buy,
        catalyst_labels=catalyst_labels,
        signal_categories=signal_cats,
    )

    # --- CONVICTION CAP: 3+ risk flags → max "Qualified" ---
    raw_conviction = _conviction_label(score)
    if len(risk_flags) >= 3 and raw_conviction in ("Exceptional", "High Conviction"):
        capped_conviction = "Qualified (Risk-Capped)"
    else:
        capped_conviction = raw_conviction

    return {
        "symbol": symbol,
        "score": score,
        "conviction": capped_conviction,
        "opportunity_type": opportunity_type,
        "price": round(price, 2),
        "reasons": reasons,
        "risk_flags": risk_flags,
        "foundation_check": foundation_check,
        "sector": _get_sector_for_ticker(symbol, fund=fund),
        "thematic": thematic_tags,
        "catalysts": catalyst_labels if catalyst_labels else [],
        "signal_convergence": signal_cats,
        "description": fund.get("description", ""),
        "recent_news": fund.get("news_headlines", []),
    }


# ---------------------------------------------------------------------------
# 4. STABLE HELPERS
# ---------------------------------------------------------------------------

@log_exceptions()
def _stable_unique_symbols(symbols: list[str]) -> list[str]:
    """Preserve first occurrence while keeping recommendation inputs deterministic."""
    seen = set()
    ordered = []
    for symbol in symbols:
        if symbol not in seen:
            ordered.append(symbol)
            seen.add(symbol)
    return ordered


@log_exceptions()
def _rank_opportunities(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sort opportunities deterministically to keep recommendations reproducible."""
    return sorted(
        results,
        key=lambda item: (-float(item.get("score", 0)), str(item.get("symbol", "")))
    )


@log_exceptions()
def _select_qualified_opportunities(
    results: list[dict[str, Any]],
    *,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """
    Keep only opportunities that still clear the score gate.
    Falls back to a lower threshold only when nothing clears the primary bar.
    """
    qualified = [r for r in results if float(r.get("score", 0)) >= _MIN_SCORE_THRESHOLD]
    if not qualified:
        qualified = [
            r for r in results if float(r.get("score", 0)) >= _FALLBACK_MIN_SCORE
        ][:5]

    if limit is not None:
        return qualified[:limit]
    return qualified


# ---------------------------------------------------------------------------
# 4.5. HEADWIND RISK GATE
# ---------------------------------------------------------------------------

@log_exceptions()
def _headwind_check(symbol: str) -> dict[str, Any]:
    """
    Fast risk assessment for a single stock.
    Checks for material headwinds that should be surfaced before any buy recommendation:
      - Short interest (>10% of float)
      - Insider selling patterns
      - Bearish management tone in last earnings call
      - Earnings proximity (within 7 days = high event risk)

    Returns a dict of risk signals. Empty dict = no material headwinds detected.
    """
    result = {}

    try:
        # 1. Short interest + Insider activity (single API call via yfinance .info)
        from tools.insider_data import get_insider_and_short_data
        insider_data = get_insider_and_short_data(symbol)
        if insider_data and "error" not in insider_data:
            # Short interest
            short_info = insider_data.get("short_interest", {})
            short_str = short_info.get("short_percent_of_float", "N/A")
            if short_str != "N/A":
                try:
                    short_pct = float(short_str.replace("%", "")) / 100
                    if short_pct > 0.10:
                        result["short_pct_float"] = short_pct
                except (ValueError, TypeError):
                    pass

            # Insider signal
            insider_signal = insider_data.get("insider_signal", "")
            if insider_signal:
                result["insider_signal"] = insider_signal

    except Exception:
        pass

    try:
        # 2. Management tone (NLP on latest earnings call transcript)
        from tools.earnings_nlp import analyze_management_tone
        from tools.tool_errors import is_unavailable
        tone_data = analyze_management_tone(symbol)
        # 5.4: an unreadable transcript now says so instead of scoring the
        # provider's rate-limit notice and calling the result Neutral. Checked
        # explicitly rather than relying on the Neutral filter below to swallow
        # it — those are different findings and only one of them is a reading.
        if tone_data and not is_unavailable(tone_data) and "error" not in tone_data:
            tone_status = tone_data.get("tone_status", "")
            if tone_status and tone_status != "Neutral":
                result["management_tone"] = tone_status
    except Exception:
        pass

    try:
        # 3. Earnings proximity (check if earnings within 7 days)
        ticker = yf.Ticker(symbol)
        cal = ticker.calendar
        if cal is not None:
            from datetime import datetime
            if isinstance(cal, dict) and "Earnings Date" in cal:
                dates = cal["Earnings Date"]
                if dates and len(dates) > 0:
                    earnings_date_str = str(dates[0])[:10]
                    try:
                        earnings_dt = datetime.strptime(earnings_date_str, "%Y-%m-%d")
                        # CALENDAR days, not elapsed timedelta days. The earnings
                        # date parses at midnight, so subtracting a mid-day now()
                        # truncated every count one day low — and on the earnings
                        # day itself produced -1, failing the `0 <=` gate and
                        # silently dropping the flag on the single riskiest day.
                        days_to = (earnings_dt.date() - datetime.now().date()).days
                        if 0 <= days_to <= 7:
                            result["days_to_earnings"] = days_to
                    except (ValueError, TypeError):
                        pass
    except Exception:
        pass

    return result


def _headwind_check_parallel(symbols: list[str], max_workers: int = 4, overall_budget: float = 35.0) -> dict[str, dict]:
    """Run headwind checks for multiple symbols in parallel."""
    results = {}
    if not symbols:
        return results

    safe_print(f"🛡️ Running headwind risk checks for {len(symbols)} finalists...")

    _prof = get_active_profile()
    executor = ThreadPoolExecutor(max_workers=min(len(symbols), max_workers))
    try:
        future_map = {executor.submit(run_under_profile, _prof, _headwind_check, sym): sym for sym in symbols}
        # Absent = no headwind data, which the caller already treats as safe.
        results = _collect_bounded(future_map, overall_budget)
    finally:
        executor.shutdown(wait=False, cancel_futures=True)

    flagged = sum(1 for v in results.values() if v)
    safe_print(f"🛡️ Headwind check complete: {flagged}/{len(symbols)} stocks flagged with risk signals")
    return results


# ---------------------------------------------------------------------------
# 5. PUBLIC API — scan_sector_opportunities (same signature as before)
# ---------------------------------------------------------------------------

# Conviction tiers the funnel backtest found actually beat SPY (only High
# Conviction cleared it; Exceptional sits above it). Portfolio-fit previews are
# spent only on these — the Qualified bulk is dilutive, so previewing it would
# burn fetches on names not worth adding.
_IMPACT_PREVIEW_TIERS = {"Exceptional", "High Conviction"}
_IMPACT_PREVIEW_MAX = 3        # cap fetches added to a personalized scan
_IMPACT_PREVIEW_BUDGET_S = 45  # own wall-clock budget, separate from the scan's


def _compact_impact_preview(report: dict[str, Any]) -> dict[str, Any] | None:
    """Squeeze a full preview_candidate_impact report down to what belongs on a
    scan pick — headline, the three deltas, correlation, and IPS fit."""
    if not isinstance(report, dict) or report.get("error"):
        return None
    out: dict[str, Any] = {
        "headline": report.get("headline"),
        "proposed_size": (report.get("proposed_size") or {}).get("dollars"),
    }
    rd = report.get("risk_deltas")
    if isinstance(rd, dict) and "error" not in rd:
        if "beta" in rd:
            out["beta_delta"] = rd["beta"]["delta"]
        if "volatility" in rd:
            out["volatility_delta"] = rd["volatility"]["delta"]
        cvar_key = next((k for k in rd if k.startswith("cvar_")), None)
        if cvar_key:
            out["cvar_delta_dollars"] = rd[cvar_key]["delta_dollars"]
        out["correlation_to_portfolio"] = rd.get("candidate_correlation_to_portfolio")
    elif isinstance(rd, dict):
        out["risk_note"] = rd.get("error")
    flags = (report.get("ips_checks") or {}).get("flags", [])
    out["ips_flags"] = flags
    out["ips_fit"] = "would breach IPS" if flags else "within IPS limits"
    return out


def _attach_impact_previews(result: dict[str, Any], portfolio_context: dict | None) -> None:
    """Enrich the top High-Conviction picks in-place with a 4.9 candidate-impact
    preview ("what does adding this do to my book?").

    Personalized path only: skipped when there is no real portfolio (the neutral
    nightly scan passes portfolio_context=None and never reaches here anyway).
    Best-effort and bounded — never raises, never blocks a scan past its budget.
    """
    try:
        if not isinstance(result, dict) or result.get("error"):
            return
        if not isinstance(portfolio_context, dict) or portfolio_context.get("error"):
            return
        if not portfolio_context.get("holdings"):
            return
        picks = result.get("top_picks") or []
        targets = [p for p in picks if p.get("conviction") in _IMPACT_PREVIEW_TIERS][:_IMPACT_PREVIEW_MAX]
        if not targets:
            return
        from tools.candidate_impact import preview_candidate_impact
        deadline = time.perf_counter() + _IMPACT_PREVIEW_BUDGET_S
        for pick in targets:
            if time.perf_counter() > deadline:
                break
            symbol = pick.get("symbol")
            if not symbol:
                continue
            try:
                compact = _compact_impact_preview(preview_candidate_impact(symbol))
                if compact:
                    pick["impact_preview"] = compact
            except Exception:
                continue
    except Exception:
        return


@log_exceptions()
def scan_sector_opportunities(sector: str, _recursion_depth: int = 0) -> dict[str, Any]:
    """
    Scans a specific sector OR the entire market for high-potential investment opportunities.
    Incorporates portfolio context to penalize over-concentrated sectors.

    Args:
        sector: The sector to scan (e.g., "Tech", "Finance", "Energy")
                OR "All"/"Market" to scan everything.
        _recursion_depth: Internal — kept for backward compatibility, no longer used.
    """
    from agent.logger import log_to_component

    scan_start = time.perf_counter()
    # Broad/funnel scans (All/Market) do more staged work → larger budget.
    is_broad_scan = sector.upper() in [
        "ALL", "MARKET", "GENERAL", "ANY", "EVERYTHING",
        "ALL SECTORS", "BROAD MARKET", "BROAD",
    ]
    scan_timeout = _V2_SCAN_TIMEOUT if is_broad_scan else _SCAN_TIMEOUT
    # The pipeline self-aborts ~5s before the hard timeout so it stops cooperatively
    # (at a phase boundary) instead of leaving a runaway thread that keeps doing I/O
    # long after the caller already received a timeout error.
    deadline = scan_start + scan_timeout - 5
    log_to_component("tools", "OpportunityScanner", "Starting scan", {
        "sector": sector,
        "timeout": scan_timeout,
    })
    # Coalesce with an identical scan already running for this profile. Keyed by
    # the CANONICAL sector so the "All"/"Market"/"General" aliases collapse to
    # one entry — that alias set is precisely how the duplicate arrives.
    # The check and the claim MUST be one critical section. Registering the real
    # worker future later — after the portfolio read and the executor build —
    # leaves a window in which two callers both see an empty registry and both
    # become leaders, which is precisely the arrival pattern being deduplicated:
    # the planner submits `screen_stocks` and `scan_opportunities` to one
    # executor, so they land microseconds apart, not the 50ms a staggered test
    # implies. A placeholder future is claimed here and resolved by `_publish`
    # once the leader has an answer.
    _scan_profile = get_active_profile()
    _dedup_key = (str(_scan_profile), "All" if is_broad_scan else sector.upper())
    _shared: Future = Future()
    with _INFLIGHT_LOCK:
        _leader_future = _INFLIGHT_SCANS.get(_dedup_key)
        _is_follower = _leader_future is not None and not _leader_future.done()
        if not _is_follower:
            _INFLIGHT_SCANS[_dedup_key] = _shared
    if _is_follower:
        log_to_component("tools", "OpportunityScanner", "Joined in-flight scan", {
            "sector": sector, "key": _dedup_key[1],
        })
        safe_print(f"🔗 Reusing in-flight {sector} scan (deduplicated).")
        try:
            return _leader_future.result(timeout=scan_timeout)
        except FuturesTimeoutError:
            return {"sector": sector, "error": f"Scan timed out after {scan_timeout} seconds"}
        except Exception as e:
            return _empty_result(sector, f"❌ Scan failed: {e}")

    def _publish(payload: dict[str, Any]) -> dict[str, Any]:
        """Hand the leader's answer to every follower waiting on this key.

        Called on every leader return path, and once more from the outer
        ``finally`` as a backstop — an unresolved placeholder would leave each
        follower blocked for its own full ``scan_timeout`` and then report a
        timeout for a scan that never ran.
        """
        if not _shared.done():
            _shared.set_result(payload)
        return payload

    try:
        return _run_leader_scan(
            sector, scan_timeout, scan_start, deadline, _scan_profile, _publish,
        )
    finally:
        _publish(_empty_result(sector, "❌ Scan failed: no result was produced"))
        # Release only our own claim: a later scan may already hold the key, and
        # clobbering it would strand its followers.
        with _INFLIGHT_LOCK:
            if _INFLIGHT_SCANS.get(_dedup_key) is _shared:
                del _INFLIGHT_SCANS[_dedup_key]


def _run_leader_scan(
    sector: str,
    scan_timeout: int,
    scan_start: float,
    deadline: float,
    _scan_profile: Any,
    _publish: Callable[[dict[str, Any]], dict[str, Any]],
) -> dict[str, Any]:
    """The scan itself, for the caller that won the in-flight claim.

    Split out of ``scan_sector_opportunities`` so the claim/release bookkeeping
    reads as one unit and no return path can skip ``_publish``.
    """
    from agent.logger import log_to_component
    from tools.portfolio_csv import get_portfolio_decision_context
    portfolio_context = get_portfolio_decision_context()

    def _do_scan():
        try:
            # Outer wrapper just handles sector aliases
            upper = sector.upper()
            if upper in ["ALL", "MARKET", "GENERAL"]:
                return _scan_impl("All", portfolio_context=portfolio_context, deadline=deadline)
            else:
                return _scan_impl(sector, portfolio_context=portfolio_context, deadline=deadline)
        except Exception as e:
            elapsed = time.perf_counter() - scan_start
            log_to_component("tools", "OpportunityScanner", "Scan failed", {
                "sector": sector, "elapsed": round(elapsed, 1),
                "error": str(e), "error_type": type(e).__name__,
            }, level=40)
            raise

    def _do_scan_and_attach():
        result = _do_scan()
        # Personalized portfolio-fit (Roadmap 4.9): attach candidate-impact
        # previews to the High-Conviction picks. Runs INSIDE the shared worker
        # (already profile-bound) so a deduplicated follower receives the same
        # fully-formed result instead of racing the leader to mutate it.
        _attach_impact_previews(result, portfolio_context)
        return result

    # Timeout wrapper
    # The request-scoped profile was captured above (this thread is bound) and is
    # re-applied inside the worker: the active-profile ContextVar does not
    # propagate into a ThreadPoolExecutor worker, so without this the entire scan
    # pipeline — including profile-scoped memory/theses reads and the per-profile
    # daily cache — would resolve to the empty '_unbound' profile under the
    # multi-user guard.
    try:
        executor = ThreadPoolExecutor(max_workers=1)
        try:
            future = executor.submit(run_under_profile, _scan_profile, _do_scan_and_attach)
            try:
                result = future.result(timeout=scan_timeout)
                elapsed = time.perf_counter() - scan_start
                log_to_component("tools", "OpportunityScanner", "Scan completed", {
                    "sector": sector, "elapsed": round(elapsed, 1),
                    "picks": len(result.get("top_picks", [])) if isinstance(result, dict) else 0
                })
                return _publish(result)
            except FuturesTimeoutError:
                elapsed = time.perf_counter() - scan_start
                log_to_component("tools", "OpportunityScanner", "Scan timed out", {
                    "sector": sector, "elapsed": round(elapsed, 1),
                    "timeout": scan_timeout
                }, level=40)
                return _publish({"sector": sector, "error": f"Scan timed out after {scan_timeout} seconds"})
        finally:
            executor.shutdown(wait=False, cancel_futures=True)
    except Exception as e:
        safe_print(f"❌ Scan error: {e}")
        return _publish(_empty_result(sector, f"❌ Scan failed: {e}"))


def _empty_result(sector: str, summary: str) -> dict[str, Any]:
    return {
        "sector": sector,
        "sector_trend": "Unknown",
        "market_leaders": [],
        "macro_context": {},
        "top_picks": [],
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# 6. CORE PIPELINE
# ---------------------------------------------------------------------------

def _past_deadline(deadline: float | None) -> bool:
    """True once the scan's soft deadline has passed (cooperative-abort signal)."""
    return deadline is not None and time.perf_counter() > deadline


# Forward signal-log location (under gitignored user_data).
_SIGNAL_LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "user_data", "funnel_signal_log")


def _annotate_pick_novelty(picks: list[dict], lookback_days: int = 14) -> None:
    """
    Mark each pick NEW vs carried using the signal log, so repetition reads as
    information ("this signal has persisted for N scans") instead of a glitch.
    Mutates picks in place; best-effort — any failure leaves picks untouched.
    """
    try:
        from datetime import date, timedelta
        cutoff = (date.today() - timedelta(days=lookback_days)).isoformat()
        counts: dict[str, int] = {}
        first_seen: dict[str, str] = {}
        if os.path.isdir(_SIGNAL_LOG_DIR):
            for fname in sorted(os.listdir(_SIGNAL_LOG_DIR)):
                if not fname.endswith(".jsonl") or fname[:10] < cutoff:
                    continue
                day = fname[:10]
                with open(os.path.join(_SIGNAL_LOG_DIR, fname)) as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            snap = json.loads(line)
                        except Exception:
                            continue
                        for p in snap.get("picks", []) or []:
                            sym = (p.get("symbol") or "").upper()
                            if sym:
                                counts[sym] = counts.get(sym, 0) + 1
                                first_seen.setdefault(sym, day)
        for pick in picks:
            sym = (pick.get("symbol") or "").upper()
            n = counts.get(sym, 0)
            reasons = pick.get("reasons")
            if n == 0:
                pick["novelty"] = "NEW"
                if isinstance(reasons, list):
                    reasons.insert(0, f"🆕 NEW — first surfacing in the last {lookback_days}d")
            else:
                pick["novelty"] = f"carried:{n}"
                pick["first_surfaced"] = first_seen.get(sym)
                if isinstance(reasons, list):
                    reasons.append(
                        f"📌 Repeat signal — surfaced in {n} scan(s) since {first_seen.get(sym)} "
                        "(persistence, not a new idea)"
                    )
    except Exception as e:
        safe_print(f"⚠️ Pick-novelty annotation skipped (non-fatal): {e}")


def _near_miss_cut_reason(row: dict[str, Any]) -> str:
    """Attribute WHY a scored finalist was cut, most-specific gate first.

    A gate demotion usually *causes* the below-threshold score, so gate
    attribution takes precedence over the score symptom.
    """
    if row.get("entry_multiplier", 1.0) < 1.0:
        return "entry_gate_demotion"
    breakdown = row.get("score_breakdown") or {}
    if (breakdown.get("risk_adjust") or 0) < 0 or row.get("risk_flags"):
        return "risk_flag"
    try:
        score = float(row.get("score", 0))
    except (TypeError, ValueError):
        score = 0.0
    if score < _MIN_SCORE_THRESHOLD:
        return "below_score_threshold"
    return "outside_top_n"


def _collect_near_misses(
    ranked_finalists: list[dict],
    selected: list[dict],
    all_scored: list[dict],
    max_n: int = 12,
) -> list[dict[str, Any]]:
    """
    Miss-detector input: the names the funnel scored but did NOT surface, so the
    backtest can measure false negatives (the MU-at-$347 failure mode) and hold
    each gate empirically accountable.

    Tier A — finalists that went through the full Phase-5 gate stack (entry
    gate, risk/headwind, flow) but were dropped by the final selection; the cut
    is attributed to the specific gate. Tier B — the next names in line from
    Phase-4 scoring that never made finalist ("not_finalist").
    """
    selected_syms = {p.get("symbol") for p in selected}
    finalist_syms = {r.get("symbol") for r in ranked_finalists}
    misses: list[dict[str, Any]] = []

    def _compact(row: dict, reason: str) -> dict[str, Any]:
        return {
            "symbol": row.get("symbol"),
            "price": row.get("price"),
            "score": row.get("score"),
            "conviction": row.get("conviction"),
            "theme": row.get("theme"),
            "entry_stage": row.get("entry_stage"),
            "entry_multiplier": row.get("entry_multiplier"),
            "cut_reason": reason,
        }

    for row in ranked_finalists:
        sym = row.get("symbol")
        if not sym or sym in selected_syms:
            continue
        misses.append(_compact(row, _near_miss_cut_reason(row)))

    for row in all_scored:
        if len(misses) >= max_n:
            break
        sym = row.get("symbol")
        if not sym or sym in finalist_syms or sym in selected_syms:
            continue
        misses.append(_compact(row, "not_finalist"))

    return misses[:max_n]


def _log_funnel_signals(result: dict[str, Any], macro_context: dict | None = None,
                        near_misses: list[dict] | None = None) -> None:
    """
    M5 — Forward signal-log (§11 Tier-2).

    Append a compact, point-in-time snapshot of a broad scan to
    user_data/funnel_signal_log/{YYYY-MM-DD}.jsonl so that flow/score signals —
    which cannot be reconstructed from historical price data — accumulate a
    walk-forward record. After ~1-2 quarters this enables forward validation of
    the flow_bonus and entry-stage multiplier. Best-effort: any failure is swallowed.
    """
    try:
        from datetime import datetime, timezone
        os.makedirs(_SIGNAL_LOG_DIR, exist_ok=True)
        now = datetime.now(UTC)
        picks = []
        for p in (result.get("top_picks") or [])[:_FINAL_TOP_N]:
            if not isinstance(p, dict):
                continue
            picks.append({
                "symbol": p.get("symbol"),
                "price": p.get("price"),
                "score": p.get("score"),
                "conviction": p.get("conviction"),
                "theme": p.get("theme"),
                "theme_cycle_stage": p.get("theme_cycle_stage"),
                "entry_stage": p.get("entry_stage"),
                "flow_confirmations": p.get("flow_confirmations", []),
                "score_breakdown": p.get("score_breakdown", {}),
                "universe_provenance": p.get("universe_provenance", []),
            })
        snapshot = {
            "ts": now.isoformat(),
            "market_status": result.get("market_status"),
            "macro": macro_context or result.get("macro_context", {}),
            "ranked_themes": [
                {"theme": t.get("theme"), "theme_score": t.get("theme_score"),
                 "cycle_stage": t.get("cycle_stage")}
                for t in (result.get("ranked_themes") or [])[:8]
            ],
            "picks": picks,
            # Miss-detector: names scored but not surfaced, tagged with the gate
            # that cut them — lets the backtest measure false negatives.
            "near_misses": near_misses or [],
        }
        path = os.path.join(_SIGNAL_LOG_DIR, f"{now.strftime('%Y-%m-%d')}.jsonl")
        with open(path, "a") as f:
            f.write(json.dumps(snapshot, default=str) + "\n")
    except Exception as e:
        safe_print(f"⚠️ Signal-log write skipped (non-fatal): {e}")


class _ScanContext:
    """Mutable state threaded through the _scan_impl phases.

    Attribute names match the local variable names used inside each phase body,
    so a phase unpacks ``x = ctx.x`` at the top and writes ``ctx.x = x`` back at
    the bottom without renaming anything in between.
    """

    def __init__(self, sector: str, portfolio_context: dict | None = None,
                 deadline: float | None = None):
        self.sector = sector
        self.portfolio_context = portfolio_context
        self.deadline = deadline
        u = sector.upper()
        self.is_guru = u in ["GURU", "MAD MONEY", "JIM GURU", "GURU PICKS"]
        self.is_broad = (not self.is_guru) and u in [
            "ALL", "MARKET", "ANY", "EVERYTHING", "ALL SECTORS", "BROAD MARKET", "BROAD"]
        self.is_mega_cap = u in ["MEGA-CAP", "MEGA CAP", "TOP 50"]
        self.is_growth = u in ["GROWTH LEADERS", "GROWTH"]
        self.is_defensive = u in ["VALUE", "DEFENSIVE", "VALUE & DEFENSIVE"]
        # Guru overlay (populated in Phase 1; initialised for all scans).
        self.guru_metadata: dict = {}
        self.guru_summary: dict = {"picks": [], "ticker_metadata": {}, "sweet_spot_count": 0}
        self.guru_enabled = False
        # Phase 0
        self.rotation_data: dict = {}
        self.macro_context: dict = {}
        self.macro_bullish_sectors: list = []
        self.macro_bearish_sectors: list = []
        self.sector_trend_map: dict = {}
        # Phase 1 / 1.5
        self.universe_provenance: dict[str, list[str]] = {}
        self.candidates: list = []
        self.ranked_themes: list[dict] = []
        self._sector_theme_map: dict[str, dict] = {}
        # Phase 2
        self.technicals: dict = {}
        self.technical_symbols: list = []
        self.missing_technical_symbols: list = []
        self.candidate_rs: dict[str, float] = {}
        self.top_candidates: list = []
        self.phase2_start = 0.0
        # Phase 3
        self.fundamentals: dict = {}
        self.fundamental_symbols: list = []
        self.missing_fundamental_symbols: list = []
        self.fundamental_missing_fields: dict = {}
        self.deep_symbols: list = []
        # Phase 4
        self.results: list = []
        self.scoring_failures: list = []
        self.no_score_failures: list = []
        self.current_trend = "Neutral ⚪"
        self.high_quality: list = []
        self.pre_headwind_picks: list = []
        self.phase4_start = 0.0


def _scan_phase0_macro(ctx: "_ScanContext"):
    """Phase 0 — macro & sector-rotation context (fetched once, reused)."""
    from agent.logger import log_to_component
    sector = ctx.sector
    rotation_data = ctx.rotation_data
    macro_context = ctx.macro_context
    macro_bullish_sectors = ctx.macro_bullish_sectors
    macro_bearish_sectors = ctx.macro_bearish_sectors
    sector_trend_map = ctx.sector_trend_map

    # ---------------------------------------------------------------
    # Phase 0: Macro & Sector Rotation (fetched once, reused)
    # ---------------------------------------------------------------
    safe_print("🔄 Fetching macro & sector rotation context...")
    phase0_start = time.perf_counter()

    rotation_data = {"leading_sectors": [], "lagging_sectors": [], "full_rotation_map": [], "market_status": "Neutral"}
    macro_context = {"liquidity": "Unknown", "crash_risk": "Unknown", "curve_status": "Unknown"}
    macro_bullish_sectors = []
    macro_bearish_sectors = []

    try:
        from tools.market_mechanics import detect_sector_rotation
        rotation_data = detect_sector_rotation()
    except Exception as e:
        safe_print(f"⚠️ Sector rotation failed: {e}")

    if is_cancelled():
        return _empty_result(sector, "🛑 Cancelled.")

    try:
        from tools.fred_api import get_systemic_risk_indicators, get_treasury_yields
        systemic = get_systemic_risk_indicators()
        yields = get_treasury_yields()
        macro_context = {
            "liquidity": systemic.get("liquidity_status", "Unknown"),
            "m2_growth": systemic.get("m2_growth_yoy", "0%"),
            "crash_risk": systemic.get("crash_risk", "Unknown"),
            "curve_status": yields.get("curve_status", "Unknown"),
        }
    except Exception as e:
        safe_print(f"⚠️ Macro indicators failed: {e}")

    try:
        from tools.macro_strategy import analyze_macro_context
        macro_strat = analyze_macro_context()
        macro_bullish_sectors = [s.upper() for s in macro_strat.get("strategy", {}).get("tactical_opportunity", [])]
        macro_bearish_sectors = [s.upper() for s in macro_strat.get("strategy", {}).get("sectors_to_underweight", [])]
    except Exception:
        pass

    phase0_elapsed = time.perf_counter() - phase0_start
    log_to_component("tools", "OpportunityScanner", "Phase 0 complete (macro)", {
        "elapsed": round(phase0_elapsed, 1),
    })

    # Build sector trend map
    sector_trend_map = {}
    for item in rotation_data.get("full_rotation_map", []):
        sec_name = item["sector"].upper()
        trend = item["trend"]
        if "TECHNOLOGY" in sec_name: sector_trend_map["TECHNOLOGY"] = trend
        elif "FINANCIAL" in sec_name: sector_trend_map["FINANCE"] = trend
        elif "HEALTH" in sec_name: sector_trend_map["HEALTHCARE"] = trend
        elif "ENERGY" in sec_name: sector_trend_map["ENERGY"] = trend
        elif "CONSUMER DISCRET" in sec_name: sector_trend_map["CONSUMER DISCRETIONARY"] = trend
        elif "CONSUMER STAPLES" in sec_name: sector_trend_map["CONSUMER STAPLES"] = trend
        elif "INDUSTRIALS" in sec_name: sector_trend_map["INDUSTRIALS"] = trend
        elif "UTILITIES" in sec_name: sector_trend_map["UTILITIES"] = trend
        elif "REAL ESTATE" in sec_name: sector_trend_map["REAL ESTATE"] = trend
        elif "MATERIALS" in sec_name: sector_trend_map["MATERIALS"] = trend
        elif "COMMUNICATION" in sec_name: sector_trend_map["COMMUNICATION"] = trend
        # Semiconductor rotated through Technology ETF typically
        sector_trend_map.setdefault("SEMICONDUCTORS", sector_trend_map.get("TECHNOLOGY", "Neutral ⚪"))

    if is_cancelled():
        return _empty_result(sector, "🛑 Cancelled.")

    # --- write-back cross-phase state ---
    ctx.rotation_data = rotation_data
    ctx.macro_context = macro_context
    ctx.macro_bullish_sectors = macro_bullish_sectors
    ctx.macro_bearish_sectors = macro_bearish_sectors
    ctx.sector_trend_map = sector_trend_map
    return None


def _scan_phase1_universe(ctx: "_ScanContext"):
    """Phase 1 — assemble the candidate universe for the requested mode."""
    from agent.logger import log_to_component
    sector = ctx.sector
    is_guru = ctx.is_guru
    is_broad = ctx.is_broad
    is_mega_cap = ctx.is_mega_cap
    is_growth = ctx.is_growth
    is_defensive = ctx.is_defensive
    guru_metadata = ctx.guru_metadata
    guru_summary = ctx.guru_summary
    guru_enabled = ctx.guru_enabled
    rotation_data = ctx.rotation_data
    universe_provenance = ctx.universe_provenance
    candidates = ctx.candidates

    # ---------------------------------------------------------------
    # Phase 1: Universe Assembly
    # ---------------------------------------------------------------
    safe_print("📋 Assembling candidate universe...")
    phase1_start = time.perf_counter()

    # Always fetch Guru metadata to use as an overlay signal across all scans
    try:
        from tools.guru_feed import get_guru_feed_summary
        guru_summary = get_guru_feed_summary()
        guru_metadata = guru_summary.get("ticker_metadata", {})
    except ImportError:
        guru_summary = {"picks": [], "ticker_metadata": {}, "sweet_spot_count": 0}
        guru_metadata = {}
    except Exception:
        guru_metadata = {}

    # Whether the optional guru feed exists in this build (vs. simply having no
    # hit for a given ticker). Surfaced in the result so confidence grading can
    # ignore guru absence when the feature is switched off entirely.
    guru_enabled = _guru_feed_available()

    # Provenance dict populated only for the broad/dynamic path; empty for all others.
    # Always initialised here so downstream code can access it unconditionally.
    universe_provenance: dict[str, list[str]] = {}

    if is_guru:
        # Guru virtual sector: inject tickers from Media Sentiment feed
        safe_print("📺 Guru Mode: Fetching Media Sentiment picks...")
        try:
            from tools.guru_feed import get_guru_universe
            candidates = get_guru_universe(freshness_filter="active")
            safe_print(f"📺 Guru Feed: {len(candidates)} active tickers, "
                       f"{guru_summary.get('sweet_spot_count', 0)} in sweet spot")
        except ImportError:
            candidates = []
            safe_print("📺 Guru Feed: Module excluded from public release.")
        except Exception as e:
            # Module present but failed at runtime (network/parse) — degrade to an
            # empty result instead of crashing the explicitly-selected Guru scan,
            # consistent with the overlay/universe guru paths.
            candidates = []
            safe_print(f"📺 Guru Feed: unavailable ({e}) — returning no picks.")
    elif is_growth:
        # Growth Leaders: Tech, Discretionary, Communication
        all_tickers = []
        for sec_name in ["Technology", "Consumer Discretionary", "Communication Services"]:
            all_tickers.extend(_get_sector_tickers(sec_name))
        candidates = _stable_unique_symbols(all_tickers)[:50]
        safe_print(f"📈 Growth Mode: Isolating top {len(candidates)} tech/growth leaders...")
    elif is_defensive:
        # Value & Defensive: Healthcare, Staples, Utilities, Financials
        all_tickers = []
        for sec_name in ["Healthcare", "Consumer Staples", "Utilities", "Financials"]:
            all_tickers.extend(_get_sector_tickers(sec_name))
        candidates = _stable_unique_symbols(all_tickers)[:50]
        safe_print(f"🛡️ Defensive Mode: Isolating top {len(candidates)} defensive/value names...")
    elif is_mega_cap or is_broad:
        # Funnel V2 M1: dynamic universe from live market state (rotation + movers + themes + guru)
        # replaces the old [:4]-per-sector static stub that silently excluded MU and other
        # momentum names that hadn't yet entered the static universe file.
        # Reuse the guru tickers already fetched above to avoid a redundant feed call.
        _guru_tickers = list(guru_metadata.keys()) if guru_metadata else None
        candidates, universe_provenance = _assemble_dynamic_universe(
            rotation_data, guru_tickers=_guru_tickers
        )
        candidates = _stable_unique_symbols(candidates)
        safe_print(f"🌐 Dynamic Universe Mode: {len(candidates)} candidates from live market state")
    else:
        # Specific sector
        candidates = _get_sector_tickers(sector)
        if not candidates:
            # Try web search for unknown sectors
            safe_print(f"  ⚡ No pre-defined universe for '{sector}', trying web search...")
            try:
                from tools.web_search import search_news
                search_results = search_news(f"best {sector} stocks to buy 2026", max_results=5)
                if isinstance(search_results, str):
                    found = re.findall(r'\b[A-Z]{2,5}\b', search_results)
                    blacklist = {"THE", "AND", "FOR", "TOP", "BEST", "ETF", "USD", "CAD", "NYSE", "NASDAQ",
                                 "STOCK", "NEWS", "INC", "CORP", "CEO", "IPO", "AI", "EPS", "YTD", "USA", "CNN"}
                    candidates = [t for t in found if t not in blacklist][:20]
            except Exception:
                pass
        candidates = _stable_unique_symbols(candidates)

    phase1_elapsed = time.perf_counter() - phase1_start
    safe_print(f"📊 Universe: {len(candidates)} unique tickers assembled in {phase1_elapsed:.1f}s")
    phase1_log = {
        "candidates": len(candidates), "elapsed": round(phase1_elapsed, 1),
    }
    if is_guru:
        phase1_log["candidate_symbols"] = candidates
    else:
        phase1_log["sample_symbols"] = candidates[:20]
    log_to_component("tools", "OpportunityScanner", "Phase 1 complete (universe)", phase1_log)

    if not candidates:
        return _empty_result(sector, "No candidates found for this sector.")

    if is_cancelled():
        return _empty_result(sector, "🛑 Cancelled.")

    # --- write-back cross-phase state ---
    ctx.guru_summary = guru_summary
    ctx.guru_metadata = guru_metadata
    ctx.guru_enabled = guru_enabled
    ctx.universe_provenance = universe_provenance
    ctx.candidates = candidates
    return None


def _scan_phase1_5_themes(ctx: "_ScanContext"):
    """Phase 1.5 — rank themes (broad scans) and build the sector→theme map."""
    is_broad = ctx.is_broad
    rotation_data = ctx.rotation_data
    macro_context = ctx.macro_context
    macro_bullish_sectors = ctx.macro_bullish_sectors
    macro_bearish_sectors = ctx.macro_bearish_sectors
    universe_provenance = ctx.universe_provenance
    ranked_themes = ctx.ranked_themes
    _sector_theme_map = ctx._sector_theme_map

    # ---------------------------------------------------------------
    # Phase 1.5: Theme Ranking / Event & Flow Radar (M2, broad scans only)
    # Uses only already-fetched Phase-0 data + a best-effort catalyst check.
    # No blocking API calls — fails silently in all error cases.
    # ---------------------------------------------------------------
    ranked_themes: list[dict] = []
    if is_broad and not is_cancelled():
        try:
            safe_print("🧭 Phase 1.5: Ranking themes by rotation + macro + catalyst...")
            ranked_themes = _rank_themes(
                rotation_data, macro_context,
                macro_bullish_sectors, macro_bearish_sectors,
                universe_provenance,
            )
            if ranked_themes:
                top3 = ", ".join(f"{t['theme']}({t['theme_score']:.2f})" for t in ranked_themes[:3])
                safe_print(f"📊 Theme radar: top themes → {top3}")
        except Exception as e:
            safe_print(f"⚠️ Theme ranking failed (non-fatal): {e}")

    # Build a fast lookup keyed by CANONICAL sector so the Phase-4 join works
    # across taxonomies (rotation names vs _get_sector_for_ticker names).
    _sector_theme_map: dict[str, dict] = {}
    for t in ranked_themes:
        canon = t.get("canonical_sector")
        if canon and canon not in _sector_theme_map:
            _sector_theme_map[canon] = t

    # --- write-back cross-phase state ---
    ctx.ranked_themes = ranked_themes
    ctx._sector_theme_map = _sector_theme_map
    return None


def _scan_phase2_technicals(ctx: "_ScanContext"):
    """Phase 2 — batch technical screen, RS alpha, fast-score and top-N cut."""
    from agent.logger import log_to_component
    sector = ctx.sector
    is_guru = ctx.is_guru
    is_broad = ctx.is_broad
    universe_provenance = ctx.universe_provenance
    candidates = ctx.candidates
    technicals = ctx.technicals
    technical_symbols = ctx.technical_symbols
    missing_technical_symbols = ctx.missing_technical_symbols
    candidate_rs = ctx.candidate_rs
    top_candidates = ctx.top_candidates
    phase2_start = ctx.phase2_start

    # ---------------------------------------------------------------
    # Phase 2: Batch Technical Screen
    # ---------------------------------------------------------------
    safe_print(f"📡 Phase 2: Batch technical screen for {len(candidates)} tickers...")
    phase2_start = time.perf_counter()

    # Per-symbol daily-cached technicals (chunked download under the hood for the
    # uncached remainder). Caching freezes each name's fast-screen inputs for the
    # trading day so the top-N cut is stable across intraday re-runs.
    technicals = _compute_technicals_cached(candidates)
    if not technicals:
        log_to_component("tools", "OpportunityScanner", "Phase 2 failed (empty batch data)", {
            "sector": sector,
            "candidates": len(candidates),
            "candidate_symbols": candidates if is_guru else candidates[:20],
        })
        return _empty_result(sector, "Batch download returned no data.")

    missing_technical_symbols = [sym for sym in candidates if sym not in technicals]

    # --- GAP FILLER: Attempt to recover missing technicals individually ---
    # Skip names already known to be delisted (acquired/merged) so we don't re-attempt
    # them every scan. Use a SINGLE attempt (max_retries=1) — a no-data ticker won't
    # recover, and the old 3×-with-backoff retry was the source of the repeated
    # "possibly delisted" spam for PXD/SWN/TELL.
    recoverable = [s for s in missing_technical_symbols if s.upper() not in _known_delisted]
    if recoverable and not is_cancelled():
        fill_count = min(len(recoverable), 10)
        safe_print(f"🔍 Gap Filler: Attempting to recover {fill_count} tickers individually...")
        for sym in recoverable[:fill_count]:
            if is_cancelled():
                break
            try:
                single_data = _batch_download([sym], max_retries=1)
                if not single_data.empty:
                    single_tech = _compute_technicals_batch(single_data, [sym])
                    if sym in single_tech:
                        technicals[sym] = single_tech[sym]
                        import tools.daily_cache as daily_cache
                        daily_cache.set_cached(f"{_TECH_CACHE_PREFIX}{sym}", single_tech[sym])
                        safe_print(f"✨ Gap Filler: Successfully recovered {sym}")
                        continue
                # No data and no exception → treat as delisted; remember it.
                _known_delisted.add(sym.upper())
            except Exception as e:
                safe_print(f"⚠️ Gap Filler: Failed to recover {sym}: {e}")
                _known_delisted.add(sym.upper())
        if _known_delisted:
            safe_print(f"🗑️ Marked {len(_known_delisted)} symbol(s) delisted this session "
                       "(skipped in future universes).")

    safe_print(f"✅ Final technicals computed for {len(technicals)} tickers")
    technical_symbols = [sym for sym in candidates if sym in technicals]
    missing_technical_symbols = [sym for sym in candidates if sym not in technicals]

    # ── M2: RS alpha computation (broad scans only) ──────────────────────
    # Fetch SPY's 3-month return so we can compute each candidate's alpha.
    # Uses a separate single-ticker download — cheap and cached within the
    # daily cache key.  Fails silently; rs_alpha defaults to 0.0 in that case.
    candidate_rs: dict[str, float] = {}
    if is_broad and technicals:
        try:
            # Reuse _compute_technicals_batch for the SPY benchmark so the 3M-return
            # extraction handles single- vs multi-level columns identically to the
            # candidates (avoids the Open-vs-Close pitfall of ad-hoc column slicing).
            spy_data = _batch_download(["SPY"])
            spy_tech = _compute_technicals_batch(spy_data, ["SPY"]) if not spy_data.empty else {}
            spy_3m = spy_tech.get("SPY", {}).get("three_month_return")
            if spy_3m is not None:
                for sym, tech in technicals.items():
                    sym_3m = tech.get("three_month_return") or 0.0
                    candidate_rs[sym] = round(float(sym_3m) - float(spy_3m), 2)
        except Exception:
            pass

    # Fast-score and rank (pass provenance + RS alpha so themed/outperforming
    # names survive the top-N cut even before deep fundamentals run)
    scored = []
    for sym, tech_data in technicals.items():
        fs = _fast_score(
            sym, tech_data,
            provenance=universe_provenance.get(sym),
            rs_alpha=candidate_rs.get(sym, 0.0),
        )
        scored.append((sym, fs, tech_data))

    # Sort by fast-score desc, with the SYMBOL as a deterministic tiebreak.
    # _fast_score returns an int, so many candidates tie at the rank-N boundary;
    # without the symbol key the survivors of the top-N cut were decided by
    # technicals dict insertion order (i.e. chunked-download completion order),
    # which is non-deterministic under concurrency. That silently dropped an
    # ideal-setup name (observed: GOOGL qualifying in two intraday runs then
    # vanishing in a third with byte-identical cached signals). Stable now.
    scored.sort(key=lambda x: (-x[1], x[0]))

    # Take top N for deep dive
    fast_screen_top_n = int(_cfg_number("fast_screen_top_n", _FAST_SCREEN_TOP_N))
    top_candidates = scored[:fast_screen_top_n]

    # Exploration slots: ADDITIVE deep-dive budget for the least-recently-scanned
    # names below the cut. The fast screen is deterministic on a mostly-stable
    # universe, so without exploration the same top-60 wins every day and the
    # long tail is never examined (and the Miss Detector never sees it).
    exploration_slots = int(_cfg_number("exploration_slots", _EXPLORATION_SLOTS))
    if is_broad and exploration_slots > 0 and len(scored) > fast_screen_top_n:
        exploration = _exploration_candidates(
            scored, fast_screen_top_n, exploration_slots, _load_scan_ledger())
        top_candidates = top_candidates + exploration
        for sym, _fs, _td in exploration:
            prov = universe_provenance.setdefault(sym, [])
            if "exploration" not in prov:
                prov.append("exploration")
        safe_print(
            f"🧭 Exploration: +{len(exploration)} least-recently-scanned name(s) added to the deep dive: "
            f"{', '.join(s for s, _, _ in exploration)}"
        )

    # Record every deep-scanned symbol so future exploration rotates the tail.
    _update_scan_ledger([s for s, _, _ in top_candidates])
    safe_print(f"🏆 Top {len(top_candidates)} candidates selected for deep analysis")

    # Near-miss instrumentation: accumulation-stage names that just missed the
    # deep-dive cut (the next 30 ranks below it). These are the candidates a
    # "protected lane" would rescue — log them with rank + score so we can measure
    # whether quality setups are actually being culled pre-analysis before deciding
    # to build that lane (vs. simply widening fast_screen_top_n).
    near_miss = [
        {
            "symbol": s, "rank": fast_screen_top_n + i + 1, "fast_score": fs,
            "rsi": td.get("rsi"), "drawdown_pct": td.get("drawdown_pct"),
            "above_sma50": td.get("above_sma50"),
            "provenance": universe_provenance.get(s),
        }
        for i, (s, fs, td) in enumerate(scored[fast_screen_top_n:fast_screen_top_n + 30])
        if _is_accumulation_like(td)
    ]
    if near_miss:
        safe_print(
            f"🔎 {len(near_miss)} accumulation-stage name(s) just missed the deep-dive cut "
            f"(ranks {fast_screen_top_n + 1}-{fast_screen_top_n + 30}): "
            f"{', '.join(m['symbol'] for m in near_miss)}"
        )
        log_to_component("tools", "OpportunityFunnel", "fast-screen-near-miss", {
            "cut": fast_screen_top_n,
            "universe_size": len(scored),
            "near_miss_count": len(near_miss),
            "near_miss": near_miss,
        })

    phase2_elapsed = time.perf_counter() - phase2_start
    phase2_log = {
        "tickers_analyzed": len(technicals), "top_n": len(top_candidates),
        "near_miss_count": len(near_miss),
        "elapsed": round(phase2_elapsed, 1),
    }
    if is_guru or missing_technical_symbols:
        phase2_log.update({
            "technical_symbols": technical_symbols if is_guru else technical_symbols[:20],
            "missing_technical_count": len(missing_technical_symbols),
            "missing_technical_symbols": (
                missing_technical_symbols if is_guru else missing_technical_symbols[:20]
            ),
        })
    log_to_component("tools", "OpportunityScanner", "Phase 2 complete (batch technicals)", phase2_log)

    if is_cancelled():
        return _empty_result(sector, "🛑 Cancelled.")

    # --- write-back cross-phase state ---
    ctx.technicals = technicals
    ctx.technical_symbols = technical_symbols
    ctx.missing_technical_symbols = missing_technical_symbols
    ctx.candidate_rs = candidate_rs
    ctx.top_candidates = top_candidates
    ctx.phase2_start = phase2_start
    return None


def _scan_phase3_fundamentals(ctx: "_ScanContext"):
    """Phase 3 — deep fundamentals for the top candidates (parallel)."""
    from agent.logger import log_to_component
    sector = ctx.sector
    deadline = ctx.deadline
    is_guru = ctx.is_guru
    top_candidates = ctx.top_candidates
    fundamentals = ctx.fundamentals
    fundamental_symbols = ctx.fundamental_symbols
    missing_fundamental_symbols = ctx.missing_fundamental_symbols
    fundamental_missing_fields = ctx.fundamental_missing_fields
    deep_symbols = ctx.deep_symbols

    # ---------------------------------------------------------------
    # Phase 3: Deep Fundamentals (parallel, top candidates only)
    # ---------------------------------------------------------------
    deep_symbols = [sym for sym, _, _ in top_candidates]
    safe_print(f"🔬 Phase 3: Deep fundamentals for {len(deep_symbols)} candidates (parallel)...")
    phase3_start = time.perf_counter()

    fundamentals = _fetch_fundamentals_parallel(deep_symbols)
    fundamental_symbols = [sym for sym in deep_symbols if sym in fundamentals]
    missing_fundamental_symbols = [sym for sym in deep_symbols if sym not in fundamentals]
    fundamental_missing_fields = {}
    for sym in deep_symbols:
        fund = fundamentals.get(sym, {})
        if not isinstance(fund, dict):
            fundamental_missing_fields[sym] = list(_SCORING_FUNDAMENTAL_FIELDS)
            continue
        missing_fields = _missing_scoring_fields(fund)
        if missing_fields:
            fundamental_missing_fields[sym] = missing_fields

    phase3_elapsed = time.perf_counter() - phase3_start
    safe_print(f"✅ Fundamentals fetched in {phase3_elapsed:.1f}s")
    phase3_log = {
        "fetched": len(fundamentals), "elapsed": round(phase3_elapsed, 1),
    }
    if is_guru or missing_fundamental_symbols or fundamental_missing_fields:
        phase3_log.update({
            "deep_symbols": deep_symbols if is_guru else deep_symbols[:20],
            "fundamental_symbols": fundamental_symbols if is_guru else fundamental_symbols[:20],
            "missing_fundamental_count": len(missing_fundamental_symbols),
            "missing_fundamental_symbols": (
                missing_fundamental_symbols if is_guru else missing_fundamental_symbols[:20]
            ),
            "fundamental_missing_fields": (
                fundamental_missing_fields if is_guru else dict(list(fundamental_missing_fields.items())[:20])
            ),
        })
    log_to_component("tools", "OpportunityScanner", "Phase 3 complete (fundamentals)", phase3_log)

    if is_cancelled():
        return _empty_result(sector, "🛑 Cancelled.")
    if _past_deadline(deadline):
        safe_print("⏱️ Deadline reached before Phase 4 — returning partial/empty result cooperatively.")
        return _empty_result(sector, "⏱️ Scan deadline reached before scoring; try a narrower sector.")

    # --- write-back cross-phase state ---
    ctx.fundamentals = fundamentals
    ctx.fundamental_symbols = fundamental_symbols
    ctx.missing_fundamental_symbols = missing_fundamental_symbols
    ctx.fundamental_missing_fields = fundamental_missing_fields
    ctx.deep_symbols = deep_symbols
    return None


def _scan_phase4_scoring(ctx: "_ScanContext"):
    """Phase 4 — deep scoring + forward outlook → pre-headwind picks."""
    sector = ctx.sector
    portfolio_context = ctx.portfolio_context
    deadline = ctx.deadline
    is_broad = ctx.is_broad
    guru_metadata = ctx.guru_metadata
    macro_context = ctx.macro_context
    macro_bullish_sectors = ctx.macro_bullish_sectors
    macro_bearish_sectors = ctx.macro_bearish_sectors
    sector_trend_map = ctx.sector_trend_map
    universe_provenance = ctx.universe_provenance
    _sector_theme_map = ctx._sector_theme_map
    candidate_rs = ctx.candidate_rs
    top_candidates = ctx.top_candidates
    fundamentals = ctx.fundamentals
    deep_symbols = ctx.deep_symbols
    results = ctx.results
    scoring_failures = ctx.scoring_failures
    no_score_failures = ctx.no_score_failures
    current_trend = ctx.current_trend
    high_quality = ctx.high_quality
    pre_headwind_picks = ctx.pre_headwind_picks
    phase4_start = ctx.phase4_start

    # ---------------------------------------------------------------
    # Phase 4: Deep Scoring + Forward Outlook
    # ---------------------------------------------------------------
    safe_print("📊 Phase 4: Deep scoring + forward outlook...")
    phase4_start = time.perf_counter()

    # PERF: the M3 forward pillar (_m3_forward_points) calls predict_earnings_surprise(symbol)
    # for every candidate. That helper is @cached daily, but a cold cache means ~60 SEQUENTIAL
    # yfinance earnings-date calls inside the scoring loop below (observed: 100-200s, the cause
    # of the broad-scan 90s timeouts). Warm the cache in parallel first so each in-loop call is
    # a cache hit. No change to the scoring math — just moves the I/O off the critical path.
    #
    # This warm is NOT gated on is_broad. A named-sector scan reaches the same scoring loop
    # and pays the same sequential cost; gating it here only meant sector scans ate the
    # 100-200s the comment above measures. The step is budget-capped and non-fatal, so a
    # smaller candidate set simply finishes the warm early.
    if not is_cancelled() and not _past_deadline(deadline):
        try:
            from tools.market_mechanics import predict_earnings_surprise
            _warm_start = time.perf_counter()
            # Cap the warm step to whatever time remains before the soft deadline
            # (minus headroom for the scoring loop), never more than 30s.
            _remaining = (deadline - time.perf_counter()) if deadline else 30.0
            _warm_budget = max(5.0, min(30.0, _remaining - 20.0))
            _warm_cache_parallel(predict_earnings_surprise, deep_symbols,
                                 max_workers=8, overall_budget=_warm_budget)
            safe_print(f"🔥 Warmed earnings-surprise cache for {len(deep_symbols)} names "
                       f"in {time.perf_counter() - _warm_start:.1f}s (budget {_warm_budget:.0f}s)")
        except Exception as e:
            safe_print(f"⚠️ Earnings-surprise warm failed (non-fatal): {e}")

    results = []
    failed_count = 0
    scoring_failures = []
    no_score_failures = []
    current_trend = "Neutral ⚪"
    for sym, _fast_sc, tech_data in top_candidates:
        if is_cancelled() or _past_deadline(deadline):
            safe_print(f"⏱️ Stopping Phase-4 scoring early at {len(results)} scored (deadline/cancel).")
            break

        fund = fundamentals.get(sym, {"symbol": sym})
        if isinstance(fund, dict) and "error" in fund:
            fund = {"symbol": sym}

        ticker_sector, current_trend, local_macro, tags, theme_context = _resolve_scoring_context(
            sym, fund, sector_trend_map, macro_bullish_sectors, macro_bearish_sectors,
            macro_context, _sector_theme_map,
        )
        if is_broad:
            scored_result = _deep_score_v2(
                sym,
                fund,
                tech_data,
                current_trend,
                local_macro,
                tags,
                portfolio_context=portfolio_context,
                theme_context=theme_context,
                rs_alpha=candidate_rs.get(sym, 0.0),
                universe_provenance=universe_provenance.get(sym, []),
                apply_entry_gate=False,
            )
        else:
            scored_result = _deep_score_value_v1(sym, fund, tech_data, current_trend, local_macro, tags, portfolio_context=portfolio_context)
        if scored_result:
            if "error" in scored_result:
                failed_count += 1
                scoring_failures.append({
                    "symbol": sym,
                    "error": scored_result.get("error", "unknown_error"),
                    "missing_fields": _missing_scoring_fields(fund) if isinstance(fund, dict) else list(_SCORING_FUNDAMENTAL_FIELDS),
                    "technical_price": tech_data.get("price"),
                    "fundamental_price": fund.get("current_price") if isinstance(fund, dict) else None,
                })
                continue

            # Attach provenance so the UI/agent knows how this name entered the funnel
            prov = universe_provenance.get(sym)
            if prov:
                scored_result["universe_provenance"] = prov

            # Attach M2 theme / cycle-stage data from Phase 1.5 ranking.
            # Join by CANONICAL sector so picks in Finance / Consumer / Comm
            # (whose _get_sector_for_ticker name differs from the rotation name)
            # still receive their theme annotation.
            _pick_canon = _canonical_sector(ticker_sector)
            if _sector_theme_map and _pick_canon and _pick_canon in _sector_theme_map:
                tm = _sector_theme_map[_pick_canon]
                scored_result["theme"]             = tm["theme"]
                scored_result["theme_score"]       = tm["theme_score"]
                scored_result["theme_cycle_stage"] = tm["cycle_stage"]
                scored_result["theme_drivers"]     = tm["drivers"]

            # Add sector label for broad scans
            if is_broad and ticker_sector != "Unknown":
                scored_result["reasons"].insert(0, f"🏢 Sector: {ticker_sector}")

            # Annotate with Guru metadata universally (overlay signal)
            if sym in guru_metadata:
                cm = guru_metadata[sym]
                scored_result["source"] = "📺 Guru Pick"
                scored_result["guru_signal"] = cm.get("signal", "FEATURED")
                scored_result["guru_freshness"] = cm.get("freshness", "AGING")
                scored_result["guru_mentions"] = cm.get("mention_count", 1)
                scored_result["guru_headline"] = cm.get("headline", "")
                scored_result["guru_date"] = cm.get("date", "")

                # Freshness-aware label in reasons
                freshness_icon = {
                    "JUST_AIRED": "🔴", "SWEET_SPOT": "🟢", "AGING": "🟡"
                }.get(cm.get("freshness"), "⚪")
                signal_str = cm.get("signal", "FEATURED")
                mentions = cm.get("mention_count", 1)
                guru_label = f"{freshness_icon} 📺 Guru {signal_str}"
                if mentions > 1:
                    guru_label += f" (×{mentions} mentions)"
                scored_result["reasons"].insert(0, guru_label)
                if ticker_sector != "Unknown":
                    scored_result["reasons"].insert(1, f"🏢 Sector: {ticker_sector}")

            results.append(scored_result)
        else:
            tech_price = tech_data.get("price")
            fundamental_price = fund.get("current_price") if isinstance(fund, dict) else None
            no_score_failures.append({
                "symbol": sym,
                "reason": "missing_or_invalid_price" if not (tech_price or fundamental_price) else "score_not_returned",
                "technical_price": tech_price,
                "fundamental_price": fundamental_price,
            })

    # Rank
    results = _rank_opportunities(results)

    # Filter
    high_quality = _select_qualified_opportunities(results)

    # Final picks (before headwind gate)
    if is_broad:
        final_top_n = int(_cfg_number("final_top_n", _FINAL_TOP_N))
        pre_headwind_picks = high_quality[:final_top_n]
    else:
        pre_headwind_picks = high_quality

    if is_cancelled():
        return _empty_result(sector, "🛑 Cancelled.")

    # --- write-back cross-phase state ---
    ctx.results = results
    ctx.scoring_failures = scoring_failures
    ctx.no_score_failures = no_score_failures
    ctx.current_trend = current_trend
    ctx.high_quality = high_quality
    ctx.pre_headwind_picks = pre_headwind_picks
    ctx.phase4_start = phase4_start
    return None


def _phase5_gate_slice(share: float, remaining: float | None) -> float:
    """Seconds a Phase-5 gate may claim; 0.0 means skip it entirely.

    The gate gets its weighted share of the budget, floored at
    ``_PHASE5_MIN_GATE_S`` so it is never half-run — but always CLAMPED by the
    time that is actually left, minus the reserve the scan needs to rescore and
    summarize.

    The clamp is the whole point. A floor that consults only the share and not
    the clock re-creates the bug the budget exists to fix: three gates each
    flooring at 6s spend a flat ~15s no matter how little time remains, which
    sails past the hard timeout and turns a degraded-but-real answer back into
    "Scan timed out" — measured at +3.6s over the hard timeout with 6s left on
    the deadline. Yielding a gate costs an entry multiplier; overrunning costs
    the entire answer.

    ``remaining`` of None means no deadline is in force — the gate takes its
    nominal share.
    """
    left = remaining if remaining is not None else share + _PHASE5_RESERVE_S
    allowed = min(max(share, _PHASE5_MIN_GATE_S), left - _PHASE5_RESERVE_S)
    return allowed if allowed >= _PHASE5_MIN_GATE_S else 0.0


def _scan_phase5_finalize(ctx: "_ScanContext"):
    """Phase 5 — headwind/flow gate, final ranking, summary + result assembly."""
    from agent.logger import log_to_component
    sector = ctx.sector
    portfolio_context = ctx.portfolio_context
    deadline = ctx.deadline
    is_guru = ctx.is_guru
    is_broad = ctx.is_broad
    guru_metadata = ctx.guru_metadata
    guru_summary = ctx.guru_summary
    guru_enabled = ctx.guru_enabled
    rotation_data = ctx.rotation_data
    macro_context = ctx.macro_context
    macro_bullish_sectors = ctx.macro_bullish_sectors
    macro_bearish_sectors = ctx.macro_bearish_sectors
    sector_trend_map = ctx.sector_trend_map
    universe_provenance = ctx.universe_provenance
    candidates = ctx.candidates
    ranked_themes = ctx.ranked_themes
    _sector_theme_map = ctx._sector_theme_map
    technicals = ctx.technicals
    technical_symbols = ctx.technical_symbols
    missing_technical_symbols = ctx.missing_technical_symbols
    candidate_rs = ctx.candidate_rs
    phase2_start = ctx.phase2_start
    fundamentals = ctx.fundamentals
    fundamental_symbols = ctx.fundamental_symbols
    missing_fundamental_symbols = ctx.missing_fundamental_symbols
    fundamental_missing_fields = ctx.fundamental_missing_fields
    deep_symbols = ctx.deep_symbols
    results = ctx.results
    scoring_failures = ctx.scoring_failures
    no_score_failures = ctx.no_score_failures
    current_trend = ctx.current_trend
    high_quality = ctx.high_quality
    pre_headwind_picks = ctx.pre_headwind_picks
    phase4_start = ctx.phase4_start

    # ---------------------------------------------------------------
    # Phase 5: Headwind Risk Gate (top candidates only)
    # ---------------------------------------------------------------
    safe_print(f"🛡️ Phase 5: Headwind risk check for {len(pre_headwind_picks)} finalists...")
    phase5_start = time.perf_counter()

    finalist_symbols = [r["symbol"] for r in pre_headwind_picks]

    # Phase 5 is the last stage that does network I/O, and until now it was the
    # only one with no deadline awareness at all: it ran its three gates
    # unconditionally, so a scan already past its soft deadline kept fetching for
    # another half-minute and the caller had long since given up. Share whatever
    # time is actually left across the gates, weighted the way they cost, and
    # degrade to the ungated picks when there is none — a finalist list with no
    # entry multiplier is still a real answer; a timeout is not.
    _remaining = (deadline - time.perf_counter()) if deadline else _PHASE5_GATE_BUDGET_S

    def _gate_slice(share: float) -> float:
        """This gate's slice, measured against the clock as it is right now."""
        return _phase5_gate_slice(
            share, (deadline - time.perf_counter()) if deadline else None)

    if _remaining < _PHASE5_MIN_GATE_S + _PHASE5_RESERVE_S:
        safe_print(f"⏱️ Phase 5: {_remaining:.0f}s left before deadline — "
                   f"skipping headwind/setup/flow gates, returning ungated finalists.")
        log_to_component("tools", "OpportunityScanner", "Phase 5 gates skipped (deadline)", {
            "sector": sector, "remaining_s": round(_remaining, 1),
            "finalists": len(finalist_symbols),
        }, level=30)
        headwind_map, setup_map, flow_map = {}, {}, {}
    else:
        # Weights sum to 1.0. Gates run in priority order — headwind is the risk
        # gate and runs on every scan; setup and flow are broad-scan scoring
        # inputs, so they yield first when the clock is short.
        _gate_budget = min(_PHASE5_GATE_BUDGET_S, _remaining - _PHASE5_RESERVE_S)
        _hw_budget = _gate_slice(_gate_budget * 0.40)
        headwind_map = _headwind_check_parallel(
            finalist_symbols, overall_budget=_hw_budget) if _hw_budget else {}
        _setup_budget = _gate_slice(_gate_budget * 0.27) if is_broad else 0.0
        setup_map = _setup_check_parallel(
            finalist_symbols, overall_budget=_setup_budget) if _setup_budget else {}
        _flow_budget = _gate_slice(_gate_budget * 0.33) if is_broad else 0.0
        flow_map = _flow_confirmation_parallel(
            finalist_symbols, headwind_map,
            overall_budget=_flow_budget) if _flow_budget else {}
        if is_broad and not (_setup_budget and _flow_budget):
            log_to_component("tools", "OpportunityScanner", "Phase 5 gate dropped (deadline)", {
                "sector": sector, "remaining_s": round(_remaining, 1),
                "headwind_s": round(_hw_budget, 1),
                "setup_s": round(_setup_budget, 1), "flow_s": round(_flow_budget, 1),
            }, level=30)

    # Re-score finalists with headwinds; broad scans also apply M3 flow + entry gate.
    final_picks = []
    for result in pre_headwind_picks:
        sym = result["symbol"]
        hw = headwind_map.get(sym, {})
        if is_broad:
            # M3: every finalist is rescored after Tier-C flow/setup checks, even
            # when no headwind exists, because the entry multiplier is part of the
            # final conviction.
            fund = fundamentals.get(sym, {"symbol": sym})
            tech_data = technicals.get(sym, {})
            if not tech_data:
                final_picks.append(result)  # Keep original if no tech data
                continue

            ticker_sector, current_trend, local_macro, tags, theme_context = _resolve_scoring_context(
                sym, fund, sector_trend_map, macro_bullish_sectors, macro_bearish_sectors,
                macro_context, _sector_theme_map,
            )
            rescored = _deep_score_v2(
                sym,
                fund,
                tech_data,
                current_trend,
                local_macro,
                tags,
                portfolio_context=portfolio_context,
                theme_context=theme_context,
                rs_alpha=candidate_rs.get(sym, 0.0),
                universe_provenance=universe_provenance.get(sym, []),
                flow_data=flow_map.get(sym, {}),
                setup_data=setup_map.get(sym, {}),
                headwind_data=hw,
                apply_entry_gate=True,
            )
            if rescored and "error" not in rescored:
                if is_broad and ticker_sector != "Unknown":
                    rescored["reasons"].insert(0, f"🏢 Sector: {ticker_sector}")

                # Re-apply Guru annotations if they existed
                if "source" in result:
                    rescored["source"] = result["source"]
                    for key in ["guru_signal", "guru_freshness", "guru_mentions", "guru_headline", "guru_date"]:
                        if key in result:
                            rescored[key] = result[key]

                    # Re-apply Guru label to reasons
                    if sym in guru_metadata:
                        cm = guru_metadata[sym]
                        freshness_icon = {
                            "JUST_AIRED": "🔴", "SWEET_SPOT": "🟢", "AGING": "🟡"
                        }.get(cm.get("freshness"), "⚪")
                        signal_str = cm.get("signal", "FEATURED")
                        mentions = cm.get("mention_count", 1)
                        guru_label = f"{freshness_icon} 📺 Guru {signal_str}"
                        if mentions > 1:
                            guru_label += f" (×{mentions} mentions)"
                        rescored["reasons"].insert(0, guru_label)

                final_picks.append(rescored)
            else:
                final_picks.append(result)  # Keep original on rescore failure
        elif hw:
            # Legacy sector/guru scans keep the old value-oriented scorer.
            fund = fundamentals.get(sym, {"symbol": sym})
            tech_data = technicals.get(sym, {})
            if not tech_data:
                final_picks.append(result)
                continue

            # theme_context is unused by the legacy v1 scorer (kept for one shared helper).
            ticker_sector, current_trend, local_macro, tags, _ = _resolve_scoring_context(
                sym, fund, sector_trend_map, macro_bullish_sectors, macro_bearish_sectors,
                macro_context, _sector_theme_map,
            )
            # Pass portfolio_context so the concentration/over-weight penalty (applied in
            # Phase 4) is preserved on rescore — otherwise a headwind-bearing, concentrated
            # finalist would have its penalty silently dropped and its final score inflated.
            rescored = _deep_score_value_v1(sym, fund, tech_data, current_trend, local_macro, tags,
                                   portfolio_context=portfolio_context, headwind_data=hw)
            if rescored and "error" not in rescored:
                # Re-apply Guru annotations if they existed
                if "source" in result:
                    rescored["source"] = result["source"]
                    for key in ["guru_signal", "guru_freshness", "guru_mentions", "guru_headline", "guru_date"]:
                        if key in result:
                            rescored[key] = result[key]
                final_picks.append(rescored)
            else:
                final_picks.append(result)
        else:
            result.setdefault("risk_flags", [])
            final_picks.append(result)

    # Re-rank after headwind adjustments
    final_picks = _rank_opportunities(final_picks)
    near_misses: list[dict] = []
    if is_broad:
        final_top_n = int(_cfg_number("final_top_n", _FINAL_TOP_N))
        ranked_finalists = final_picks
        final_picks = _select_qualified_opportunities(final_picks, limit=final_top_n)
        near_misses = _collect_near_misses(ranked_finalists, final_picks, results)
        # NEW vs carried-N-scans tags — must run BEFORE the signal log is written
        # so today's scan doesn't count itself as a prior surfacing.
        _annotate_pick_novelty(final_picks)
    else:
        final_picks = _select_qualified_opportunities(final_picks)

    phase5_elapsed = time.perf_counter() - phase5_start
    total_elapsed = time.perf_counter() - phase2_start  # Skip Phase 0 (macro) from timing

    flagged_count = sum(1 for r in final_picks if r.get("risk_flags"))
    final_symbols = {p.get("symbol") for p in final_picks}
    finalist_symbols_set = set(finalist_symbols)
    risk_adjustment_diagnostics = [
        {
            "symbol": r.get("symbol"),
            "score": r.get("score"),
            "base_score": r.get("base_score"),
            "entry_stage": r.get("entry_stage"),
            "risk_flags": r.get("risk_flags", []),
            "risk_adjustments": r.get("risk_adjustments", []),
            "score_breakdown": r.get("score_breakdown", {}),
            "portfolio_fit": r.get("portfolio_fit", {}),
        }
        for r in final_picks
        if r.get("risk_flags") or r.get("risk_adjustments")
    ]
    final_exclusions = []
    for r in results:
        sym = r.get("symbol")
        if not sym or sym in final_symbols:
            continue
        if sym in finalist_symbols_set:
            reason = "Excluded after finalist risk/entry rescore or final ranking"
        elif float(r.get("score", 0) or 0) >= _MIN_SCORE_THRESHOLD:
            reason = "Cleared score threshold but fell outside finalist set"
        else:
            reason = f"Below qualification threshold ({_MIN_SCORE_THRESHOLD})"
        final_exclusions.append({
            "symbol": sym,
            "score": r.get("score"),
            "base_score": r.get("base_score"),
            "reason": reason,
            "risk_flags": r.get("risk_flags", []),
        })
    phase4_elapsed = time.perf_counter() - phase4_start
    log_to_component("tools", "OpportunityScanner", "Phase 4+5 complete (scoring + headwinds)", {
        "scored": len(results), "qualified": len(high_quality),
        "final_picks": len(final_picks), "flagged": flagged_count,
        "scoring_failures": len(scoring_failures),
        "no_score_failures": len(no_score_failures),
        "risk_adjusted": len(risk_adjustment_diagnostics),
        "final_exclusions": len(final_exclusions),
        "phase4_elapsed": round(phase4_elapsed, 1),
        "phase5_elapsed": round(phase5_elapsed, 1),
        "total_pipeline_elapsed": round(total_elapsed, 1),
    })
    scoring_failure_by_symbol = {item["symbol"]: item for item in scoring_failures}
    no_score_by_symbol = {item["symbol"]: item for item in no_score_failures}
    scanner_diagnostics = {
        "sector": sector,
        "candidate_count": len(candidates),
        "candidate_symbols": candidates if is_guru else candidates[:20],
        "universe_provenance_sample": (
            {s: universe_provenance[s] for s in list(universe_provenance)[:20]}
            if universe_provenance else {}
        ),
        "technical_count": len(technicals),
        "technical_symbols": technical_symbols if is_guru else technical_symbols[:20],
        "missing_technical_count": len(missing_technical_symbols),
        "missing_technical_symbols": missing_technical_symbols if is_guru else missing_technical_symbols[:20],
        "deep_symbol_count": len(deep_symbols),
        "deep_symbols": deep_symbols if is_guru else deep_symbols[:20],
        "fundamental_count": len(fundamentals),
        "fundamental_symbols": fundamental_symbols if is_guru else fundamental_symbols[:20],
        "missing_fundamental_count": len(missing_fundamental_symbols),
        "missing_fundamental_symbols": missing_fundamental_symbols if is_guru else missing_fundamental_symbols[:20],
        "fundamental_missing_fields": (
            fundamental_missing_fields if is_guru else dict(list(fundamental_missing_fields.items())[:20])
        ),
        "scored_count": len(results),
        "qualified_count": len(high_quality),
        "final_pick_count": len(final_picks),
        "scoring_failures": scoring_failures if is_guru else scoring_failures[:20],
        "no_score_failures": no_score_failures if is_guru else no_score_failures[:20],
        "risk_adjustments": risk_adjustment_diagnostics if is_guru else risk_adjustment_diagnostics[:20],
        "final_exclusions": final_exclusions if is_guru else final_exclusions[:20],
    }
    if (
        is_guru or missing_technical_symbols or scoring_failures or no_score_failures
        or risk_adjustment_diagnostics or final_exclusions
    ):
        log_to_component(
            "tools",
            "OpportunityScanner",
            "Scanner diagnostics by gate",
            scanner_diagnostics,
        )

    # Build summary
    leading_secs = rotation_data.get("leading_sectors", [])
    lagging_secs = rotation_data.get("lagging_sectors", [])
    market_status = rotation_data.get("market_status", "Neutral")
    guru_feed_payload = {}
    if is_guru:
        final_by_symbol = {p.get("symbol"): p for p in final_picks}
        scored_by_symbol = {r.get("symbol"): r for r in results}
        active_symbols = set(candidates)
        active_feed_picks = [
            p for p in guru_summary.get("picks", [])
            if p.get("ticker") in active_symbols and p.get("freshness") != "EXPIRED"
        ]
        feed_rows = []
        for pick in active_feed_picks:
            ticker = pick.get("ticker", "")
            final_match = final_by_symbol.get(ticker)
            scored_match = scored_by_symbol.get(ticker)
            if final_match:
                pipeline_status = "Qualified opportunity"
                score = final_match.get("score", "Data Unavailable")
                conviction = final_match.get("conviction", "Data Unavailable")
                exclusion_reason = ""
            elif scored_match:
                pipeline_status = "Filtered out"
                score = scored_match.get("score", "Data Unavailable")
                conviction = scored_match.get("conviction", "Data Unavailable")
                try:
                    score_val = float(score) if score != "Data Unavailable" else 0
                except (ValueError, TypeError):
                    score_val = 0
                if score_val >= _MIN_SCORE_THRESHOLD:
                    exclusion_reason = f"Excluded: Cleared threshold ({_MIN_SCORE_THRESHOLD}) but fell outside the top {_FINAL_TOP_N} picks"
                else:
                    exclusion_reason = f"Below qualification threshold ({_MIN_SCORE_THRESHOLD})"
            else:
                pipeline_status = "Data gap"
                score = "Data Unavailable"
                conviction = "Data Unavailable"
                scoring_failure = scoring_failure_by_symbol.get(ticker)
                no_score_failure = no_score_by_symbol.get(ticker)
                if ticker not in technicals:
                    exclusion_reason = "Missing or insufficient technical price history from batch download"
                elif ticker not in fundamentals:
                    exclusion_reason = "Fundamentals fetch did not return data for scoring"
                elif scoring_failure:
                    missing_fields = scoring_failure.get("missing_fields", [])
                    missing_text = ", ".join(missing_fields) if missing_fields else "core fields"
                    exclusion_reason = (
                        f"Scoring failed: {scoring_failure.get('error', 'unknown_error')} "
                        f"(missing: {missing_text})"
                    )
                elif no_score_failure:
                    exclusion_reason = (
                        f"Scoring returned no result: {no_score_failure.get('reason', 'unknown_reason')}"
                    )
                else:
                    exclusion_reason = "Insufficient market, technical, or fundamental data for scoring"

            feed_rows.append({
                "ticker": ticker,
                "signal": pick.get("signal", "FEATURED"),
                "freshness": pick.get("freshness", "Data Unavailable"),
                "date": pick.get("date", "Data Unavailable"),
                "headline": pick.get("headline", ""),
                "url": pick.get("url", ""),
                "mention_count": pick.get("mention_count", 1),
                "segment_type": pick.get("segment_type", "Data Unavailable"),
                "pipeline_status": pipeline_status,
                "score": score,
                "conviction": conviction,
                "exclusion_reason": exclusion_reason,
            })

        guru_feed_payload = {
            "total_picks": len(active_symbols),
            "displayed_top_picks": len(final_picks),
            "filtered_out_count": max(len(active_symbols) - len(final_picks), 0),
            "sweet_spot_count": sum(1 for p in active_feed_picks if p.get("freshness") == "SWEET_SPOT"),
            "buy_count": sum(1 for p in active_feed_picks if p.get("signal") == "BUY"),
            "sell_count": sum(1 for p in active_feed_picks if p.get("signal") == "SELL"),
            "featured_count": sum(1 for p in active_feed_picks if p.get("signal") == "FEATURED"),
            "picks": feed_rows,
        }

    summary_parts = []
    if is_guru:
        summary_parts.append(f"📺 Analyzed {len(candidates)} Guru picks")
    else:
        summary_parts.append(f"Scanned {len(candidates)} stocks")
    if is_broad:
        summary_parts.append(f"across {len(_get_all_sector_names())} sectors")
    summary_parts.append(f"in {total_elapsed:.0f}s.")
    summary_parts.append(f"Market: {market_status}.")
    if leading_secs:
        summary_parts.append(f"Leaders: {', '.join(leading_secs[:3])}.")
    if lagging_secs:
        summary_parts.append(f"Lagging: {', '.join(lagging_secs[:3])}.")
    summary_parts.append(f"Found {len(final_picks)} high-conviction opportunities.")
    data_gap_symbols = set(missing_technical_symbols) | set(missing_fundamental_symbols)
    data_gap_symbols.update(item["symbol"] for item in scoring_failures)
    data_gap_symbols.update(item["symbol"] for item in no_score_failures)
    if data_gap_symbols:
        summary_parts.append(
            f"⚠️ {len(data_gap_symbols)} candidates were excluded due to incomplete source data."
        )

    result_sector = "📺 Guru Picks (Media Sentiment)" if is_guru else (
        "Broad Market (High Conviction)" if is_broad else sector
    )

    result = {
        "sector": result_sector,
        "sector_trend": current_trend if not is_broad else market_status,
        "market_status": market_status,
        "market_leaders": leading_secs,
        "market_laggards": lagging_secs,
        "rotation_map": rotation_data.get("full_rotation_map", []),
        "macro_context": macro_context,
        "ranked_themes": ranked_themes,   # M2: ordered theme list with cycle_stage
        "top_picks": final_picks,
        "summary": " ".join(summary_parts),
        "diagnostics": scanner_diagnostics,
        "guru_enabled": guru_enabled,
        **({"guru_feed": guru_feed_payload} if is_guru else {}),
    }

    # M5: forward signal-log — append a compact snapshot of each broad scan so the
    # flow/score pillars (which cannot be reconstructed historically, §11) accumulate
    # a walk-forward record for later validation. Best-effort; never breaks a scan.
    if is_broad:
        _log_funnel_signals(result, macro_context, near_misses=near_misses)

    return result


def _scan_impl(sector: str, portfolio_context: dict | None = None, deadline: float | None = None) -> dict[str, Any]:
    """Run the opportunity funnel for ``sector`` and return a ranked result.

    Thin orchestrator over the phase helpers below. Each phase reads/writes the
    shared ``_ScanContext`` and may short-circuit by returning an ``_empty_result``
    (cancellation, no candidates, or a soft ``deadline`` hit at a phase boundary);
    the final phase returns the assembled scan result.

    Pipeline: macro → universe → themes → technical screen → fundamentals →
    scoring → headwind/flow gate + assembly.
    """
    ctx = _ScanContext(sector, portfolio_context, deadline)
    for phase in (
        _scan_phase0_macro,
        _scan_phase1_universe,
        _scan_phase1_5_themes,
        _scan_phase2_technicals,
        _scan_phase3_fundamentals,
        _scan_phase4_scoring,
    ):
        early = phase(ctx)
        if early is not None:
            return early
    return _scan_phase5_finalize(ctx)

