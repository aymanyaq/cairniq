"""
Asset Location Engine & Tax Efficiency Scoring (Theme 4.7).

Evaluates portfolio holdings across account types (Tax-Deferred, Tax-Free, Taxable)
to measure tax-placement efficiency (0-100 scale), detect tax drag leakages
(e.g., US dividend withholding tax in TFSAs, high income/bonds in taxable accounts),
and suggest non-taxable asset placement swaps across accounts.

**Jurisdiction is data, not a hardcoded assumption.** A tax-free shelter does not
have one tax treatment: US dividends in a Canadian TFSA lose 15% at source with no
way to recover it, and the same dividends in a US Roth IRA lose nothing at all.
The first version of this engine applied the Canadian rule to every TAX_FREE
account, so it would have told a US user their Roth suffers a withholding drag
that does not exist. Every jurisdiction-specific rule now reads a per-jurisdiction
policy table, fires only for jurisdictions the table covers, and stays SILENT
(no score change, a named note instead) where the jurisdiction is unknown —
the same "withhold the number rather than assert it" contract ``_is_us_ticker``
already applies to ambiguous listings.

**Where the jurisdiction comes from (4.7a).** Two sources, and the user's own
statement wins. ``account_jurisdictions`` in user memory holds a country per
account, entered at Context › Account Jurisdictions; an account absent from it
falls back to the account NAME, which is what this engine had before the store
existed. TFSA/RRSP are Canadian instruments, Roth/401(k)/IRA are US ones,
ISA/SIPP are UK ones — naming one names its jurisdiction. An account named only
"Registered" or "Pension" names a class without naming a country, and that is
reported as uncovered rather than guessed.

**Why the name is not enough, and is kept anyway.** It is weak evidence that has
inverted the answer twice — ``ISA`` matched *Visa* and ``REGISTERED`` matched
*Non-Registered*, both scoring a fully taxable account as sheltered — and it is
simply silent for "Brokerage", "Joint" or "Pension". It stays as the fallback
because it is right far more often than it is wrong and it needs nobody to type
anything; it is no longer the ceiling, because ``jurisdiction_source`` now says
which of the two answered. ``REGIONAL_LOCALE`` is still never consulted: it is a
display locale, and tax residency is a property of the ACCOUNT — one household
can hold accounts in two countries.

**This table is not tax advice and has not been reviewed by a tax professional.**
Roadmap 4.7 carries that as a named prerequisite before any of this reaches a
surface that tells the user what to do. ``TAX_POLICY_VERSION`` is stamped on
every payload so a downstream consumer (2.3 provenance, 3.8's gate) can refuse
to act on a version it does not recognise.
"""

import re
from typing import Any

import pandas as pd
import yfinance as yf

from tools.cache import cached
from tools.exception_logger import log_exceptions
from tools.portfolio_csv import load_portfolio
from tools.yf_utils import IMPLAUSIBLE_YIELD, dividend_yield_fraction

# Bump on ANY change to the shelter table or the policy table below. Consumers
# gate on it; an unrecognised version must fail closed rather than be ignored.
TAX_POLICY_VERSION = "2026-07-30.1"

# The percent-vs-fraction reader was found here, but the field is read in ten
# places and the bug was in nine of the others too. It now lives beside the
# `info` fetchers in `tools/yf_utils.py`; these names are kept so that the
# pinning tests written against this module still describe where it was caught.
_IMPLAUSIBLE_YIELD = IMPLAUSIBLE_YIELD
_dividend_yield_fraction = dividend_yield_fraction

# Ordered most-specific-first, and the ORDER IS LOAD-BEARING (asserted by test),
# for the same reason 5.9's insider classifier is ordered: the generic keys are
# substrings of the specific ones. "Roth 401(k)" must be read as tax-free before
# the 401(k) rule claims it, and LIRA must be settled before IRA.
#
# Word boundaries matter as much as order. Without them "ISA" matches "VISA" and
# "REGISTERED" matches "NON-REGISTERED" — the second was a live defect here: a
# non-registered (fully taxable) account was classified TAX_DEFERRED and scored
# as if it were sheltered.
#
# (pattern, shelter, tax_class, jurisdiction | None)
_SHELTER_RULES: list[tuple] = [
    # Explicitly taxable wordings that CONTAIN a shelter keyword. First, always.
    (r"NON[\s\-_]*REGISTERED", "NON_REGISTERED", "TAXABLE", None),
    (r"\bNON[\s\-_]*REG\b", "NON_REGISTERED", "TAXABLE", None),
    # Tax-free shelters (tax-free growth AND tax-free withdrawal)
    (r"ROTH[\s\-_]*401", "ROTH_401K", "TAX_FREE", "US"),
    (r"\bROTH\b", "ROTH_IRA", "TAX_FREE", "US"),
    (r"\bTFSA\b", "TFSA", "TAX_FREE", "CA"),
    (r"\bFHSA\b", "FHSA", "TAX_FREE", "CA"),
    (r"\bISA\b", "ISA", "TAX_FREE", "UK"),
    # Tax-deferred shelters (sheltered growth, taxed on withdrawal)
    (r"\bRRSP\b", "RRSP", "TAX_DEFERRED", "CA"),
    (r"\bRRIF\b", "RRIF", "TAX_DEFERRED", "CA"),
    (r"\bLIRA\b", "LIRA", "TAX_DEFERRED", "CA"),
    (r"\bLIF\b", "LIF", "TAX_DEFERRED", "CA"),
    (r"\bDCPP\b", "DCPP", "TAX_DEFERRED", "CA"),
    (r"\bSIPP\b", "SIPP", "TAX_DEFERRED", "UK"),
    (r"401[\s\-_]*\(?\s*K\s*\)?", "401K", "TAX_DEFERRED", "US"),
    (r"403[\s\-_]*\(?\s*B\s*\)?", "403B", "TAX_DEFERRED", "US"),
    (r"\bIRA\b", "IRA", "TAX_DEFERRED", "US"),
    (r"\bSUPER(ANNUATION)?\b", "SUPER", "TAX_DEFERRED", "AU"),
    # Shelter CLASSES that name no country. Deliberately jurisdiction-None:
    # the bucket is knowable, the tax treatment is not.
    (r"\bPENSION\b", "PENSION", "TAX_DEFERRED", None),
    (r"\bREGISTERED\b", "REGISTERED", "TAX_DEFERRED", None),
]

# What a jurisdiction's TAX-FREE shelter does to US-source dividends, and what
# the local remedy is. Absent from this table = not covered = no rule fires.
#
#   CA — TFSA: 15% withheld at source under Art. X of the Canada-US treaty and
#        NOT recoverable, because no Canadian tax is payable on the income to
#        claim a foreign tax credit against. An RRSP is exempt outright (Art.
#        XXI recognises it as a retirement plan), which is what makes the swap
#        real money rather than a preference.
#   US — a Roth/Roth-401(k) held by a US person receiving US-source dividends
#        is a domestic account holding domestic securities. Nothing is withheld,
#        so the drag this engine was written to find does not exist.
#   UK — an ISA is NOT a treaty-recognised pension, so 15% applies (with a valid
#        W-8BEN) and cannot be reclaimed inside the wrapper. A SIPP IS
#        recognised and receives 0%. So the leak is real but the remedy is a
#        specific wrapper, not the generic tax-deferred bucket.
_US_DIVIDEND_WITHHOLDING: dict[str, dict[str, Any]] = {
    "CA": {
        "withheld_in_tax_free": True,
        "rate": 0.15,
        "tax_free_label": "TFSA",
        "remedy_shelter": "RRSP",
        "remedy_note": (
            "an RRSP, which the Canada-US treaty exempts from the withholding entirely"
        ),
    },
    "US": {
        "withheld_in_tax_free": False,
        "rate": 0.0,
        "tax_free_label": "Roth",
        "remedy_shelter": None,
        "remedy_note": "",
    },
    "UK": {
        "withheld_in_tax_free": True,
        "rate": 0.15,
        "tax_free_label": "ISA",
        "remedy_shelter": "SIPP",
        "remedy_note": (
            "a SIPP, which the US-UK treaty recognises as a pension and exempts; "
            "an ISA is not recognised and cannot reclaim the withholding"
        ),
    },
}


def stored_account_jurisdictions() -> dict[str, str]:
    """The user's OWN per-account jurisdictions (4.7a), or ``{}``.

    Never raises: an unreadable store falls back to name inference, which is
    where this engine was before the store existed.
    """
    try:
        from tools.memory import get_account_jurisdictions

        return get_account_jurisdictions()
    except Exception:
        return {}


def classify_account(
    account_name: str,
    jurisdictions: dict[str, str] | None = None,
) -> dict[str, str | None]:
    """Resolve an account name to its tax class, shelter and jurisdiction.

    Returns ``{"tax_class", "shelter", "jurisdiction", "jurisdiction_source",
    "jurisdiction_inferred", "jurisdiction_conflict"}``. ``jurisdiction`` is
    ``None`` whenever no country is known — which is a REPORTED state, not a
    default to fall back on. Unrecognized and empty names are TAXABLE
    (conservative) with no jurisdiction.

    **What the user STATED beats what the name implies (4.7a).** The shelter and
    tax class still come from the name — a TFSA is a TFSA — but the country does
    not, because the name is weak evidence for it: it has inverted the answer
    twice ("ISA" in *Visa*, "REGISTERED" in *Non-Registered*) and it is simply
    silent for "Brokerage" or "Pension". ``jurisdiction_source`` says which
    answered, so a surface can show that they disagreed rather than quietly
    resolving it.

    ``jurisdictions`` is the stated map; pass it to avoid re-reading the store
    once per position. ``None`` reads it, and ``{}`` means "name inference only",
    which is what every caller got before this store existed.
    """
    inferred: str | None = None
    result: dict[str, str | None] = {
        "tax_class": "TAXABLE", "shelter": None, "jurisdiction": None,
    }

    acc_upper = ""
    if account_name and isinstance(account_name, str):
        acc_upper = account_name.upper().strip()

    if acc_upper:
        for pattern, shelter, tax_class, jurisdiction in _SHELTER_RULES:
            if re.search(pattern, acc_upper):
                inferred = jurisdiction
                result = {
                    "tax_class": tax_class,
                    "shelter": shelter,
                    "jurisdiction": jurisdiction,
                }
                break

    if jurisdictions is None:
        jurisdictions = stored_account_jurisdictions()

    from tools.memory import JURISDICTION_UNKNOWN, normalize_account_key

    stated = (jurisdictions or {}).get(normalize_account_key(account_name))

    if stated == JURISDICTION_UNKNOWN:
        # ANSWERED, and the answer is "no country I can name". Fails closed
        # exactly like an unanswered account, and is reported differently —
        # otherwise a finished profile is indistinguishable from an untouched one.
        result["jurisdiction"] = None
        source = "declared_unknown"
    elif stated:
        result["jurisdiction"] = stated
        source = "stated"
    elif inferred:
        source = "inferred_from_name"
    else:
        source = "unknown"

    result["jurisdiction_source"] = source
    result["jurisdiction_inferred"] = inferred
    # Both answered and they disagree. Not resolved here and not treated as an
    # error: the user's statement wins, and the disagreement is worth surfacing
    # because one of the two is wrong and only the user knows which.
    result["jurisdiction_conflict"] = bool(
        stated and stated != JURISDICTION_UNKNOWN and inferred and stated != inferred
    )
    return result


def classify_account_type(account_name: str) -> str:
    """
    Categorize an account name into standard tax shelter buckets:
    - 'TAX_DEFERRED': RRSP, LIRA, 401K, DCPP, PENSION, TRADITIONAL IRA
    - 'TAX_FREE': TFSA, ROTH IRA, ISA
    - 'TAXABLE': Non-Registered, Margin, Individual, Joint, Personal

    Unrecognized and empty account names default to TAXABLE (conservative).
    The bucket alone does NOT determine tax treatment — see ``classify_account``
    for the jurisdiction the rules actually key on.
    """
    return classify_account(account_name)["tax_class"] or "TAXABLE"


def _is_us_ticker(symbol: str, info: dict[str, Any] | None = None) -> bool | None:
    """Return whether the holding is confirmed to be US-listed.

    A bare ticker is ambiguous for cross-listed securities, so absent an
    exchange confirmation this returns ``None`` rather than asserting a US
    withholding-tax exposure. ``False`` means a known non-US listing or cash.
    """
    sym = symbol.upper().strip()
    if not sym or sym in ["CASH", "USD", "CAD"]:
        return False

    info = info or {}
    exchange = " ".join(
        str(info.get(key) or "").upper()
        for key in ("exchange", "fullExchangeName", "market")
    )
    us_exchange_codes = {"NMS", "NCM", "NGM", "NYQ", "ASE", "PCX", "BTS", "IEX"}
    non_us_exchange_codes = {"TOR", "VAN", "CNQ", "NEO"}
    if any(code in exchange.split() for code in us_exchange_codes) or any(
        marker in exchange for marker in ("NASDAQ", "NEW YORK STOCK EXCHANGE", "NYSE")
    ):
        return True
    if any(code in exchange.split() for code in non_us_exchange_codes) or any(
        marker in exchange for marker in ("TORONTO", "TSX VENTURE", "CANADIAN SECURITIES")
    ):
        return False

    suffix = sym.rsplit(".", 1)[-1] if "." in sym else ""
    if suffix in {"TO", "V", "CN", "NE", "L", "DE", "PA", "AS", "HK", "AX"}:
        return False
    return None


def _get_asset_tax_characteristics(symbol: str, info: dict[str, Any]) -> dict[str, Any]:
    """
    Extract key tax attributes for a holding:
    - asset_type: 'EQUITY', 'BOND', 'REIT', 'CASH'
    - dividend_yield: float (e.g. 0.035 for 3.5%)
    - is_us_listed: True for a confirmed US listing, False for a confirmed
      non-US listing, and None when listing data is ambiguous
    - is_high_income: bool (>3.0% yield or bond/REIT/cash)
    """
    sym = symbol.upper().strip()

    if sym == "CASH" or "USD" in sym or "CAD" in sym:
        return {
            "asset_type": "CASH",
            "dividend_yield": 0.0,
            "is_us_listed": False,
            "is_high_income": True,
        }

    quote_type = (info.get("quoteType") or "").upper()
    sector = (info.get("sector") or "").upper()
    industry = (info.get("industry") or "").upper()
    name = (info.get("longName") or info.get("shortName") or "").upper()

    raw_yield = _dividend_yield_fraction(info)

    is_reit = "REIT" in sector or "REIT" in industry or "REAL ESTATE" in sector or "REIT" in name
    is_bond = (
        "BOND" in name
        or "TREASURY" in name
        or "AGGREGATE" in name
        or "INCOME" in name
        or "FIXED INCOME" in industry
        or quote_type == "MUTUALFUND" and "BOND" in name
    )

    asset_type = "EQUITY"
    if is_reit:
        asset_type = "REIT"
    elif is_bond:
        asset_type = "BOND"

    is_high_income = raw_yield >= 0.03 or asset_type in ["BOND", "REIT", "CASH"]

    return {
        "asset_type": asset_type,
        "dividend_yield": raw_yield,
        "is_us_listed": _is_us_ticker(sym, info),
        "is_high_income": is_high_income,
    }


def _evaluate_position_location(
    symbol: str,
    account_name: str,
    account_type: str,
    asset_tax: dict[str, Any],
    value_base: float,
    jurisdiction: str | None = None,
) -> dict[str, Any]:
    """
    Score the placement efficiency (0-100) of a single holding in its account.
    Detects specific tax leakages & sub-optimal locations.

    Rules are evaluated independently (not elif) so multiple drags stack.
    For example a US REIT in a Canadian TFSA suffers BOTH the 15% withholding
    drag AND the high-income misplacement penalty.

    ``jurisdiction`` gates every rule whose answer differs by country. ``None``
    means the account did not name a country-specific shelter: those rules do
    not fire, the score is not moved, and a note names what was skipped. A
    penalty asserted under the wrong country's rules is worse than a missing
    one, because the user acts on it.
    """
    score = 100
    issues = []
    ideal_placement = []
    notes = []
    matched_rule = False

    asset_type = asset_tax["asset_type"]
    div_yield = asset_tax["dividend_yield"]
    is_us = asset_tax["is_us_listed"]
    is_high_inc = asset_tax["is_high_income"]

    withholding = _US_DIVIDEND_WITHHOLDING.get(jurisdiction or "")

    # Rule 1: US dividend-paying equities in a TAX-FREE shelter.
    # JURISDICTION-GATED — this is the rule that was Canada-only and unmarked.
    # It costs a Canadian TFSA holder 15% of every US dividend, and it costs a
    # US Roth holder nothing whatsoever.
    if is_us and div_yield >= 0.01 and account_type == "TAX_FREE":
        if withholding is None:
            # Fail closed and SAY SO. An unnamed jurisdiction is not a clean bill
            # of health, and a silent skip here reads identically to "no leak".
            notes.append(
                f"US withholding not assessed for {symbol} in {account_name}: no tax "
                f"jurisdiction is on file for that account, and the treatment differs by "
                f"country (15% and unrecoverable in a Canadian TFSA, zero in a US Roth). "
                f"Set the account's country at Context › Account Jurisdictions to enable "
                f"this check."
            )
        elif withholding["withheld_in_tax_free"]:
            rate = withholding["rate"]
            drag_pct = div_yield * rate * 100
            score -= 25
            issues.append(
                f"US dividend withholding drag: {rate * 100:.0f}% US withholding tax "
                f"(~{drag_pct:.2f}% annual drag) is withheld in {account_name} "
                f"({withholding['tax_free_label']}) and cannot be recovered — there is no "
                f"local tax on the income to credit it against."
            )
            if withholding["remedy_shelter"]:
                ideal_placement.append("TAX_DEFERRED")
                issues.append(
                    f"Ideal location for {symbol}: {withholding['remedy_note']}."
                )
            matched_rule = True
        else:
            # Covered jurisdiction, no drag (US person, US dividends, Roth).
            # Explicitly optimal rather than merely unpenalized.
            score = 100
            ideal_placement.append("TAX_FREE")
            matched_rule = True

    # Rule 2: High-income assets (Bonds, REITs, High Dividend >3%) held in Taxable accounts
    # Ordinary interest income and REIT distributions in taxable accounts face the highest marginal tax rates.
    if is_high_inc and account_type == "TAXABLE" and symbol != "CASH":
        score -= 30
        issues.append(
            f"High tax drag in taxable account: {symbol} ({asset_type}, yield {div_yield*100:.1f}%) "
            f"generates income taxed at full marginal rate in {account_name}."
        )
        ideal_placement.extend(["TAX_DEFERRED", "TAX_FREE"])
        matched_rule = True

    # Rule 2b: High-income assets in a tax-free shelter — not penalized for
    # income tax, but if they are ALSO US-listed, Rule 1 may already have docked
    # the withholding. Mild opportunity-cost penalty: contribution room spent on
    # income is room not sheltering capital-gains growth. Jurisdiction-neutral —
    # every tax-free wrapper this engine knows is contribution-capped.
    if is_high_inc and account_type == "TAX_FREE" and asset_type in ["BOND", "REIT"] and symbol != "CASH":
        score -= 10
        shelter_label = (withholding or {}).get("tax_free_label") or "tax-free"
        issues.append(
            f"Opportunity cost: {symbol} ({asset_type}) uses {shelter_label} room for income "
            f"that could shelter higher-growth capital gains instead."
        )
        if "TAX_DEFERRED" not in ideal_placement:
            ideal_placement.append("TAX_DEFERRED")
        matched_rule = True

    # Rule 3: US dividend equities in a TAX-DEFERRED shelter -> OPTIMAL.
    # Jurisdiction-neutral in outcome, for different reasons per country: a
    # Canadian RRSP and a UK SIPP are treaty-exempt, and a US IRA/401(k) has no
    # foreign withholding to suffer in the first place.
    if not matched_rule and is_us and div_yield >= 0.01 and account_type == "TAX_DEFERRED":
        score = 100
        ideal_placement.append("TAX_DEFERRED")
        matched_rule = True

    # Rule 4: High Growth / Low Yield Equity in TAX_FREE -> OPTIMAL
    if not matched_rule and asset_type == "EQUITY" and div_yield < 0.015 and account_type == "TAX_FREE":
        score = 100
        ideal_placement.append("TAX_FREE")
        matched_rule = True

    # Rule 5: high-growth / low-yield equity in a taxable account -> favourable.
    # Deferral is the jurisdiction-neutral reason: unrealized gains are untaxed
    # until sold, and realized gains are taxed more lightly than ordinary income
    # in every jurisdiction in the table (CA's 50% inclusion rate, US long-term
    # capital-gains rates, UK's separate CGT rates). The RATE is jurisdiction
    # data; the ranking is not, which is why no rate is quoted here.
    if not matched_rule and asset_type == "EQUITY" and div_yield < 0.015 and account_type == "TAXABLE":
        score = 85
        ideal_placement.extend(["TAX_FREE", "TAXABLE"])
        matched_rule = True

    # Fallback: generic placement
    if not matched_rule:
        if account_type in ["TAX_FREE", "TAX_DEFERRED"]:
            score = 90
            ideal_placement.append(account_type)
        else:
            score = 70
            ideal_placement.extend(["TAX_FREE", "TAX_DEFERRED"])

    if not ideal_placement:
        ideal_placement.append("TAX_FREE")

    return {
        "symbol": symbol,
        "account": account_name,
        "account_type": account_type,
        "jurisdiction": jurisdiction,
        "value_base": value_base,
        "score": max(0, score),
        "asset_type": asset_type,
        "dividend_yield_pct": f"{div_yield * 100:.2f}%",
        "is_us_listed": is_us,
        "ideal_placement": list(set(ideal_placement)),
        "issues": issues,
        # Checks that did NOT run, and why. Never folded into `issues`: a skipped
        # check is not a finding, and an empty `notes` is the only thing that
        # makes an empty `issues` mean "clean".
        "notes": notes,
    }


@cached(key_func=lambda: "asset_location_analysis", ttl=3600)
@log_exceptions()
def analyze_asset_location() -> dict[str, Any]:
    """
    Perform a complete Asset Location & Tax-Efficiency analysis across all portfolio accounts.

    Returns:
        Dict containing:
        - overall_score: Weighted asset location efficiency score (0-100)
        - rating: 'OPTIMAL', 'GOOD', 'SUBOPTIMAL', 'POOR'
        - total_value_base: Total portfolio value analyzed
        - account_breakdown: Breakdown of assets and scores by account
        - position_evaluations: Detailed position placement scores
        - tax_leakages: List of identified tax drag issues
        - recommended_swaps: Recommended non-taxable asset swaps across accounts
    """
    holdings = load_portfolio()
    if isinstance(holdings, dict) and "error" in holdings:
        return holdings

    if not holdings or not isinstance(holdings, list):
        return {"note": "No portfolio holdings available to analyze."}

    # Gather unique non-cash symbols
    symbols = list(set(
        item.get("symbol", "").upper().strip()
        for item in holdings
        if item.get("symbol") and item.get("symbol", "").upper() != "CASH" and not item.get("is_private_asset")
    ))

    # Fetch ticker info from yfinance (batch/fallback)
    ticker_info_map = {}
    if symbols:
        for sym in symbols:
            try:
                t = yf.Ticker(sym)
                ticker_info_map[sym] = t.info or {}
            except Exception:
                ticker_info_map[sym] = {}

    evaluations = []
    total_val_base = 0.0
    weighted_score_sum = 0.0
    tax_leakages = []
    skipped_checks = []

    account_stats = {}

    # Read the stated map ONCE. `classify_account` reads it per call when it is
    # not supplied, which is right for the single-account callers in tax_policy
    # and wrong inside a loop over every holding.
    stated_jurisdictions = stored_account_jurisdictions()

    for item in holdings:
        sym = item.get("symbol", "").upper().strip()
        if not sym or item.get("is_private_asset"):
            continue

        acc_name = item.get("account", "Unknown Account")
        acc_class = classify_account(acc_name, jurisdictions=stated_jurisdictions)
        acc_type = acc_class["tax_class"]
        acc_jurisdiction = acc_class["jurisdiction"]

        # Parse position value
        val_base = float(item.get("value_base") or item.get("current_value") or 0.0)
        if val_base <= 0:
            try:
                shares = float(str(item.get("shares", 0)).replace(",", ""))
                price = float(str(item.get("purchase_price", 0)).replace("$", "").replace(",", ""))
                val_base = shares * price
            except Exception:
                val_base = 0.0

        if val_base <= 0:
            continue

        info = ticker_info_map.get(sym, {})
        asset_tax = _get_asset_tax_characteristics(sym, info)
        pos_eval = _evaluate_position_location(
            sym, acc_name, acc_type, asset_tax, val_base, jurisdiction=acc_jurisdiction
        )

        evaluations.append(pos_eval)
        total_val_base += val_base
        weighted_score_sum += pos_eval["score"] * val_base

        if pos_eval["issues"]:
            for issue in pos_eval["issues"]:
                tax_leakages.append({
                    "symbol": sym,
                    "account": acc_name,
                    "issue": issue,
                    "value_base": round(val_base, 2),
                })
        for note in pos_eval["notes"]:
            skipped_checks.append({
                "symbol": sym,
                "account": acc_name,
                "reason": note,
                "value_base": round(val_base, 2),
            })

        # Track account breakdown
        if acc_name not in account_stats:
            account_stats[acc_name] = {
                "account_name": acc_name,
                "account_type": acc_type,
                "jurisdiction": acc_jurisdiction,
                "jurisdiction_source": acc_class.get("jurisdiction_source"),
                "jurisdiction_conflict": acc_class.get("jurisdiction_conflict"),
                "shelter": acc_class["shelter"],
                "total_value_base": 0.0,
                "weighted_score_sum": 0.0,
                "positions_count": 0,
            }
        account_stats[acc_name]["total_value_base"] += val_base
        account_stats[acc_name]["weighted_score_sum"] += pos_eval["score"] * val_base
        account_stats[acc_name]["positions_count"] += 1

    if total_val_base <= 0:
        return {"note": "No valid position values found to calculate asset location scores."}

    overall_score = round(weighted_score_sum / total_val_base, 1)

    # Determine Rating
    if overall_score >= 90:
        rating = "🟢 OPTIMAL — High Tax Placement Efficiency"
    elif overall_score >= 75:
        rating = "🟡 GOOD — Minor Tax Drag Optimizations Available"
    elif overall_score >= 60:
        rating = "🟠 SUBOPTIMAL — Moderate Tax Drag Leakage"
    else:
        rating = "🔴 POOR — Significant Tax Placement Inefficiency"

    # Build Account Summaries
    account_summary_list = []
    for acc_name, stats in account_stats.items():
        acc_val = stats["total_value_base"]
        acc_score = round(stats["weighted_score_sum"] / acc_val, 1) if acc_val > 0 else 100.0
        account_summary_list.append({
            "account": acc_name,
            "account_type": stats["account_type"],
            "jurisdiction": stats["jurisdiction"],
            # WHICH source answered. A jurisdiction the user stated and one this
            # engine guessed from the account's name carry different weight, and
            # publishing only the answer makes them indistinguishable — the same
            # "a number without its basis is an assertion" rule the sector
            # decomposition learned.
            "jurisdiction_source": stats.get("jurisdiction_source"),
            "jurisdiction_conflict": stats.get("jurisdiction_conflict"),
            "shelter": stats["shelter"],
            "value_base": round(acc_val, 2),
            "score": acc_score,
            "positions_count": stats["positions_count"],
        })

    # Generate recommended placement swaps. A swap is only proposed WITHIN one
    # jurisdiction: moving a holding between accounts governed by different tax
    # systems is not an asset-location decision, it is a cross-border transfer
    # with its own consequences this engine does not model.
    recommended_swaps = []
    leaking_tax_free = [
        e for e in evaluations
        if e["account_type"] == "TAX_FREE" and e["is_us_listed"]
        and any("withholding" in i for i in e["issues"])
    ]
    deferred_growth = [
        e for e in evaluations
        if e["account_type"] == "TAX_DEFERRED" and e["asset_type"] == "EQUITY" and not e["issues"]
    ]

    for leaking_pos in leaking_tax_free[:2]:
        policy = _US_DIVIDEND_WITHHOLDING.get(leaking_pos["jurisdiction"] or "") or {}
        same_jurisdiction = [
            g for g in deferred_growth if g["jurisdiction"] == leaking_pos["jurisdiction"]
        ]
        for growth_pos in same_jurisdiction[:2]:
            recommended_swaps.append({
                "action": "ASSET_SWAP",
                "jurisdiction": leaking_pos["jurisdiction"],
                "move_out_of": {
                    "symbol": leaking_pos["symbol"],
                    "account": leaking_pos["account"],
                    "account_type": "TAX_FREE",
                },
                "move_into": {
                    "symbol": growth_pos["symbol"],
                    "account": growth_pos["account"],
                    "account_type": "TAX_DEFERRED",
                },
                "rationale": (
                    f"Move US dividend payer {leaking_pos['symbol']} from {leaking_pos['account']} "
                    f"({policy.get('tax_free_label', 'tax-free')}) into {growth_pos['account']}, where the "
                    f"{policy.get('rate', 0.15) * 100:.0f}% US withholding does not apply, and move "
                    f"high-growth {growth_pos['symbol']} the other way so the tax-free room shelters "
                    f"capital gains instead."
                ),
            })

    jurisdictions_seen = sorted({
        a["jurisdiction"] for a in account_summary_list if a["jurisdiction"]
    })
    uncovered_accounts = sorted({
        a["account"] for a in account_summary_list
        if not a["jurisdiction"] and a["account_type"] != "TAXABLE"
    })
    # How the covered ones were answered. An engine reporting 100% coverage on
    # names alone has a weaker claim than the same figure stated by the user, and
    # the score cannot show the difference.
    jurisdiction_sources: dict[str, int] = {}
    for a in account_summary_list:
        source = a.get("jurisdiction_source") or "unknown"
        jurisdiction_sources[source] = jurisdiction_sources.get(source, 0) + 1
    jurisdiction_conflicts = sorted({
        a["account"] for a in account_summary_list if a.get("jurisdiction_conflict")
    })

    return {
        "overall_score": overall_score,
        "rating": rating,
        "total_value_base": round(total_val_base, 2),
        "account_breakdown": account_summary_list,
        "tax_leakages": tax_leakages,
        "recommended_swaps": recommended_swaps,
        "position_evaluations": evaluations,
        # Coverage, reported beside the score rather than inferred from it. A
        # high score over accounts whose jurisdiction was never resolved means
        # "not checked", not "efficient" — and only these fields can tell them
        # apart.
        "tax_policy_version": TAX_POLICY_VERSION,
        "jurisdictions_covered": jurisdictions_seen,
        "uncovered_accounts": uncovered_accounts,
        "jurisdiction_sources": jurisdiction_sources,
        "jurisdiction_conflicts": jurisdiction_conflicts,
        "skipped_checks": skipped_checks,
        "note": (
            "Asset location places foreign-dividend and high-income assets in whichever shelter "
            "does not tax them, and capital-gains growth in the tax-free wrapper. Which shelter "
            "that is depends on the account's tax jurisdiction, taken from what you stated at "
            "Context › Account Jurisdictions and otherwise inferred from the account name "
            "(TFSA/RRSP → Canada, Roth/IRA/401(k) → US, ISA/SIPP → UK); `jurisdiction_source` "
            "on each account says which answered. "
            "Jurisdiction-specific checks are SKIPPED, not guessed, for accounts with no "
            "country on file — see `uncovered_accounts` and `skipped_checks`. "
            "General information, not tax advice; confirm with a tax professional before acting."
        ),
    }


@log_exceptions()
def portfolio_account_jurisdictions() -> dict[str, Any]:
    """Every account in the portfolio, and how its jurisdiction resolves today.

    This is what the entry screen renders, and it exists because a store keyed by
    free text can silently fail to match the thing it describes. The user picks
    from the accounts their holdings actually name rather than typing a name from
    memory — a stored jurisdiction that matches no account is a filled-in store
    that changes nothing, which is this codebase's most repeated failure.

    Returns ``{accounts: [...], stated_unmatched: [...], ...}`` where each account
    carries the name as held, the stated code, the code inferred from the name,
    the effective one and its source. Never raises.
    """
    from tools.memory import (
        JURISDICTION_UNKNOWN,
        get_account_jurisdictions_record,
        normalize_account_key,
    )

    stated = stored_account_jurisdictions()
    record = get_account_jurisdictions_record() or {}

    holdings = load_portfolio()
    names: list[str] = []
    if isinstance(holdings, list):
        for item in holdings:
            name = str(item.get("account") or "").strip()
            if name and name not in names:
                names.append(name)

    accounts = []
    matched_keys = set()
    for name in sorted(names):
        key = normalize_account_key(name)
        matched_keys.add(key)
        resolved = classify_account(name, jurisdictions=stated)
        accounts.append({
            "account": name,
            "key": key,
            "stated": stated.get(key),
            "inferred_from_name": resolved.get("jurisdiction_inferred"),
            "jurisdiction": resolved.get("jurisdiction"),
            "source": resolved.get("jurisdiction_source"),
            "conflict": resolved.get("jurisdiction_conflict"),
            "shelter": resolved.get("shelter"),
            "tax_class": resolved.get("tax_class"),
            # A TAXABLE account needs no country: income taxed at the marginal
            # rate has no shelter rule to get wrong. Saying so is what stops the
            # screen reporting a question that does not exist.
            "jurisdiction_needed": resolved.get("tax_class") != "TAXABLE",
        })

    # Stated jurisdictions that match no account the portfolio holds. Renamed
    # account, a typo, or a closed account — reported rather than swept up,
    # because from inside the store these are indistinguishable from working
    # entries and they are the way this design fails quietly.
    stated_unmatched = sorted(k for k in stated if k not in matched_keys)

    needing = [a for a in accounts if a["jurisdiction_needed"]]
    return {
        "accounts": accounts,
        "stated_unmatched": stated_unmatched,
        "set_at": record.get("set_at"),
        "note": record.get("note") or "",
        "counts": {
            "accounts": len(accounts),
            "need_jurisdiction": len(needing),
            "stated": sum(1 for a in needing if a["source"] == "stated"),
            "declared_unknown": sum(1 for a in needing
                                    if a["source"] == "declared_unknown"),
            "inferred_from_name": sum(1 for a in needing
                                      if a["source"] == "inferred_from_name"),
            "unanswered": sum(1 for a in needing if a["source"] == "unknown"),
            "conflicts": sum(1 for a in needing if a["conflict"]),
        },
        "unknown_value": JURISDICTION_UNKNOWN,
        "tax_policy_version": TAX_POLICY_VERSION,
    }


if __name__ == "__main__":
    import json
    res = analyze_asset_location()
    print(json.dumps(res, indent=2))
