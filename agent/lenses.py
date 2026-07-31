"""
MarketAnalyst lens contracts.

Quick-action buttons send ``[MarketAnalyst lens=<name>]`` as a prefix. The
MarketAnalyst synthesis prompt branches on the lens so each button produces a
distinct, non-overlapping output contract instead of the same generic market
summary.

Each contract specifies (Goal, Primary deliverable, Secondary, Do NOT, Lead with).
The **Do NOT** lines cross-disclaim sibling lenses so they stay mutually exclusive:

    portfolio_audit  — audit CURRENT holdings for risk            (Analyze button)
    external_screen  — surface NEW candidates worth attention     (Scan button)
    guru_validation  — validate media-guru picks via the pipeline (Guru button)
    market_dip       — anticipatory selloff deployment plan       (Dip Plan button)

To add a lens: add an entry here (with cross-disclaimers in its "Do NOT" line) and
point a button at ``[MarketAnalyst lens=<name>]``.
"""

from __future__ import annotations

import re

LENS_CONTRACTS: dict[str, str] = {
    "portfolio_audit": (
        "<lens_contract name=\"portfolio_audit\">\n"
        "Goal: Audit the user's CURRENT holdings for uncompensated risk.\n"
        "Primary deliverable: Concentration, FX, correlation, and sector-gap findings on holdings the user already owns.\n"
        "Secondary: ONE external suggestion only if cash is above the user's target band AND a clear gap exists.\n"
        "Do NOT: surface external screen tables, momentum rotation ranks, or guru-pick lists. Those belong to other lenses.\n"
        "Lead with: the single biggest uncompensated risk in dollar terms.\n"
        "</lens_contract>\n"
    ),
    "external_screen": (
        "<lens_contract name=\"external_screen\">\n"
        "Goal: Surface NEW external tickers worth attention right now.\n"
        "Primary deliverable: Ranked external picks with foundation grade, conviction, and one-line thesis each.\n"
        "Secondary: Per-pick portfolio-fit note (1 line) — held / watchlist / sector-gap candidate / overlap with existing position.\n"
        "Do NOT: produce a full portfolio audit, sector allocation tables, or FX exposure tables (those are the portfolio_audit lens's job). Do NOT build a market-drawdown deployment plan, staged buy-triggers, or a cash-tranche schedule — that is the Dip Plan (market_dip) lens. You surface candidates; you do not time a selloff or schedule deployment.\n"
        "Lead with: the single highest-conviction external pick (and why now, not why ever).\n"
        "</lens_contract>\n"
    ),
    "guru_validation": (
        "<lens_contract name=\"guru_validation\">\n"
        "Goal: Validate which Media Guru picks survived the opportunity pipeline and are worth a second look.\n"
        "Primary deliverable: For each cleared pick — guru source, signal type, freshness, pipeline status, and the foundation/headwind result.\n"
        "Secondary: Distinguish the full scanned feed count from the smaller cleared set so the user sees the filter ratio.\n"
        "Do NOT: produce a portfolio audit, sector tables, or FX tables. Do NOT produce a generic external screen (that is the Scan / external_screen lens) or a market-dip deployment plan (that is the Dip Plan / market_dip lens) — your universe is ONLY media-guru-sourced picks from the pipeline. Do NOT include picks that failed the pipeline as if they passed.\n"
        "Lead with: count cleared / count scanned and the single strongest cleared pick.\n"
        "</lens_contract>\n"
    ),
    "market_dip": (
        "<lens_contract name=\"market_dip\">\n"
        "Goal: An ANTICIPATORY market-selloff opportunity & capital-deployment plan — what high-quality names to accumulate when the market tanks, at what levels, with how much cash. This is about being READY: it works in calm markets too (show how close we are to dip-buy territory), not only mid-crash.\n"
        "Primary deliverable: (1) Market drawdown context — VIX, SPY/index drawdown, breadth, and Fear & Greed → classify the regime (Normal / Pullback / Correction / Bear) and state whether we are in dip-buying territory yet. (2) A ranked dip shopping list of HIGH-QUALITY names oversold on the weakness — sourced from BOTH the user's watchlist AND a broad quality screen — each with an entry zone, a structural stop (support / 40-week MA / ATR), and a STAGED buy trigger (a start level plus an add-if-it-drops-further level, e.g. 'start at X, add if it breaks Y or SPY falls another Z%').\n"
        "Secondary: A staged cash-deployment plan — quote available dry powder, and split it into 2-3 tranches to deploy as the market falls further (never all at one level), stating each tranche's dollar size and its dollar-at-risk to the stop. Size against the user's OWN risk limits from their profile; if they have stated none, report the dollar-at-risk and do not measure it against any limit.\n"
        "Do NOT: chase falling knives — a broken thesis or fundamental deterioration is NOT a dip; exclude such names and say why. Do NOT deploy all cash at one level. Do NOT recommend low-quality names just because they dropped. Do NOT anchor stops to round numbers — anchor to market structure. Do NOT produce a generic 'what's worth attention now' external screen or momentum-rotation pick list — that is the Scan (external_screen) lens; your defining contribution is the drawdown TIMING plus staged trigger levels and tranche deployment, NOT a flat pick list. Do NOT pull from media-guru picks — that is the Guru (guru_validation) lens.\n"
        "Lead with: how deep the current dip is (are we in buy territory yet?) and the single highest-conviction name to accumulate on weakness.\n"
        "</lens_contract>\n"
    ),
}

# Lenses whose contract never authorizes a concrete trade call (portfolio_audit is
# descriptive; external_screen/guru_validation explicitly surface candidates without
# entry timing or sizing). Their output must never get auto-logged into the
# recommendation ledger as a BUY/SELL/etc — that would contradict RiskManager's own
# compliance judgment on the same turn ("no unauthorized trade recommendation made").
# market_dip is deliberately excluded: its contract requires concrete entry zones,
# stops, and staged tranches, which IS a real trade call worth tracking.
SCREENER_ONLY_LENSES = frozenset({"portfolio_audit", "external_screen", "guru_validation"})


def extract_lens(raw_query: str) -> str | None:
    """Parse ``[MarketAnalyst lens=<name>]`` out of the raw user query.

    Returns the lens name (lowercased) when it matches one of the known contracts,
    otherwise ``None``. Tolerant of whitespace and quotes.
    """
    if not raw_query:
        return None
    m = re.match(r"\s*\[\s*MarketAnalyst\s+lens\s*=\s*['\"]?([A-Za-z_]+)['\"]?\s*\]", str(raw_query))
    if not m:
        return None
    name = m.group(1).lower()
    if name not in LENS_CONTRACTS:
        import logging
        logging.getLogger(__name__).warning(
            "Unknown MarketAnalyst lens '%s' — falling back to default behavior", name
        )
        return None
    return name
