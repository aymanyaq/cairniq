"""Macro regime analogue matching — Advisor Roadmap 2.7 (label authored constants).

This module contains the codebase's clearest example of the failure class 2.7
exists to close: **the app itself supplying invented figures that the model then
reports in good faith.** `match_historical_regime` returned `forecast_3mo` /
`forecast_1yr` strings — "Crash (-50%) as bubble bursts", "Strong Rally (+15%)" —
that were typed into a dict by hand, from a function whose NAME asserts a
historical match. Nothing in the payload distinguished them from a measurement,
so no grounding audit could catch them: the numbers came from our own source, so
they looked sourced.

Two halves, and only one of them was ever a computation:

  * the **similarity match** IS computed — a weighted Euclidean distance from the
    live macro inputs to each regime's stats, and the regime stats themselves are
    coarse but real history (1970s inflation ~9%, 2008 fed funds ~2%);
  * the **outcomes** were never computed from anything. They are one author's
    prose about what happened next.

So the fields are named for what they are (`authored_scenario_3mo` /
`authored_scenario_1yr`, replacing `forecast_*`), the payload carries
`basis: "authored constant"` with a per-field `basis_detail`, and it points at
the measured alternative. `tools/episode_replay.py` (4.3) answers the neighbouring
question — "what did THIS portfolio actually do in that episode" — from real daily
paths, and stamps `basis: "measured"`.

The marker is only half the fix. `agent/tool_output.py::annotate_authored_basis`
attaches the attribution instruction at the seam where tool output becomes model
context, so the prose layer ATTRIBUTES on the marker rather than merely carrying it.
"""
from tools.exception_logger import log_exceptions

# What the outcome strings actually are, stated once. Every payload carries it.
AUTHORED_BASIS = "authored constant"
_BASIS_NOTE = (
    "The scenario outcomes below were typed into this module by hand — they are one "
    "author's description of what followed each regime, not a measurement of it. Only "
    "the similarity score is computed from the live inputs. For a measured answer to "
    "'what would this drawdown do to my portfolio', use replay_historical_episode, "
    "which replays the actual daily paths."
)


@log_exceptions()
def _normalize_trend_label(market_trend: str) -> str:
    """Normalize free-form market trend labels into a small comparable set."""
    trend = str(market_trend or "").strip().lower()
    if "bull" in trend or "risk-on" in trend:
        return "bull"
    if "bear" in trend or "risk-off" in trend or "correction" in trend:
        return "bear"
    return "neutral"


@log_exceptions()
def match_historical_regime(
    inflation_rate: float,
    fed_rate: float,
    market_trend: str,
    pe_ratio: float = 20.0
) -> dict:
    """
    Match current market conditions to the nearest of six hand-described regimes.

    The name of this function overstates it: it matches macro COORDINATES, and the
    "what happened next" it returns alongside is authored prose, not a replay. The
    match is a computation; the outcome is an opinion. See the module docstring.

    Args:
        inflation_rate: Current CPI (e.g. 3.5 for 3.5%)
        fed_rate: Current Fed Funds Rate (e.g. 5.25)
        market_trend: 'bull', 'bear', or 'neutral'
        pe_ratio: S&P 500 P/E Ratio

    Returns:
        The closest analogue and a COMPUTED similarity score, plus AUTHORED
        scenario strings marked as such by ``basis`` / ``basis_detail``.
    """

    # Historical Database (Simplified)
    # Each regime has: typical stats, description, and what happened next (outcome)
    regimes = {
        "1970s_Stagflation": {
            "stats": {"inflation": 9.0, "rate": 11.0, "pe": 12.0, "trend": "bear"},
            "desc": "High inflation, stagnant growth, high rates.",
            "next_3mo": "Choppy / Down",
            "next_1yr": "Bear Market (-20%) until inflation breaks.",
            "risks": "Inflation persistence, Policy error"
        },
        "1994_Soft_Landing": {
            "stats": {"inflation": 3.0, "rate": 5.5, "pe": 18.0, "trend": "bull"},
            "desc": "Fed raised rates rapidly to kill inflation without causing recession.",
            "next_3mo": "Volatility as rates peak",
            "next_1yr": "Strong Bull Market (+25%) once pause confirmed.",
            "risks": "Rates staying high too long"
        },
        "1999_Tech_Bubble": {
            "stats": {"inflation": 2.5, "rate": 5.0, "pe": 30.0, "trend": "bull"},
            "desc": "Explosive tech rally driven by hype, ignoring valuations.",
            "next_3mo": "Melt-up (Blow-off top)",
            "next_1yr": "Crash (-50%) as bubble bursts.",
            "risks": "Valuation compression, Tech earnings miss"
        },
        "2008_Financial_Crisis": {
            "stats": {"inflation": 4.0, "rate": 2.0, "pe": 15.0, "trend": "bear"}, # Pre-crash
            "desc": "Systemic leverage collapse, housing bust.",
            "next_3mo": "Crash (-30%)",
            "next_1yr": "Recession and slow recovery.",
            "risks": "Credit freeze, Bank failures"
        },
        "2020_Pandemic_Stimulus": {
            "stats": {"inflation": 1.5, "rate": 0.0, "pe": 22.0, "trend": "bull"},
            "desc": "Massive liquidity injection, rates at zero.",
            "next_3mo": "Strong Rally (+15%)",
            "next_1yr": "Inflation spike followed by correction.",
            "risks": "Overheating, Asset bubbles"
        },
        "2023_AI_Boom": {
            "stats": {"inflation": 3.4, "rate": 5.25, "pe": 24.0, "trend": "bull"},
            "desc": "High rates but massive tech productivity theme (AI).",
            "next_3mo": "Tech leadership, broader market lag",
            "next_1yr": "Bull trend depends on earnings delivery.",
            "risks": "AI hype fatigue, Rates 'Higher for Longer'"
        }
    }

    # Simple Distance Matching (Euclidean)
    best_match = None
    min_dist = float('inf')

    scores = {}
    normalized_trend = _normalize_trend_label(market_trend)

    for name, data in regimes.items():
        s = data["stats"]
        # Normalize weights: Inflation (3x), Rate (2x), PE (1x)
        # Assuming inputs are already floats
        try:
             d_inf = abs(float(inflation_rate) - s["inflation"])
             d_rate = abs(float(fed_rate) - s["rate"])
             d_pe = abs(float(pe_ratio) - s["pe"])
             d_trend = 0.0 if normalized_trend == s["trend"] else 2.0 if normalized_trend == "neutral" else 4.0
             dist = (3 * d_inf) + (2 * d_rate) + (1 * d_pe) + d_trend
        except Exception:
             dist = 100.0 # High penalty for bad data

        scores[name] = dist

        if dist < min_dist:
            min_dist = dist
            best_match = name

    # Calculate pseudo-probability (softmax-ish inverse distance)
    # Invert scores (lower is better)
    total_inv = sum(1/(d+0.1) for d in scores.values())
    prob = (1/(min_dist+0.1)) / (total_inv + 0.0001)

    match_data = regimes[best_match]

    return {
        "matched_regime": best_match.replace("_", " "),
        "match_score": round(prob * 100, 1), # backward-compatible alias
        "similarity_score": round(prob * 100, 1),
        "description": match_data["desc"],
        # 2.7: named for what they are. These were `forecast_3mo` / `forecast_1yr`,
        # which read as derived output; nothing derives them.
        "authored_scenario_3mo": match_data["next_3mo"],
        "authored_scenario_1yr": match_data["next_1yr"],
        "key_risks": match_data["risks"],
        "trend_alignment": normalized_trend,
        "basis": AUTHORED_BASIS,
        "basis_note": _BASIS_NOTE,
        # Per-field, because this payload genuinely mixes the two and a single
        # blanket label would be wrong in one direction or the other.
        "basis_detail": {
            "similarity_score": (
                "computed — weighted Euclidean distance from the supplied macro "
                "inputs to each regime's stats"
            ),
            "authored_scenario_3mo": "authored constant — typed by hand, not measured",
            "authored_scenario_1yr": "authored constant — typed by hand, not measured",
            "key_risks": "authored constant — typed by hand",
        },
        "measured_alternative": "replay_historical_episode",
        "methodology_note": "Similarity score is a heuristic analogue match, not a statistical probability.",
        # Sort scores to show top similar periods
        "similar_periods": sorted(
             [{"period": k.replace("_", " "), "score": round((1/(v+0.1))/(total_inv+0.0001) * 100, 1)} for k,v in scores.items()],
             key=lambda x: x['score'],
             reverse=True
        )[:3]
    }

if __name__ == "__main__":
    # Test: Currentish data
    print(match_historical_regime(3.2, 5.25, "bull", 25.0))
