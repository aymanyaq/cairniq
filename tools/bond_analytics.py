"""4.8 — modified duration, convexity, and what a rate move actually costs.

The live curve was already wired (5.7 supplies the real CAD leg instead of a
policy-rate proxy), and `construct_bond_ladder` already prices five rungs off it.
What was missing is the only thing a curve is *for*: a number saying how much a
holding moves when the curve moves. This module is that number.

**Four choices in here are load-bearing, and three of them are refusals.**

  * **Exact repricing is the answer; duration is the explanation.** Every shock
    row carries all three — the exact reprice, the duration-only estimate, and
    the duration+convexity estimate — because the gap between them IS the
    convexity, and quoting only the approximation is how a -100bp estimate
    silently understates the gain. `approximation_error_pct` is reported rather
    than hidden, so a caller can see where the linear number stops being usable
    (it is small at 25bp and material at 200bp, which is the whole reason the
    second-order term exists).

  * **A GIC is not a bond, and this module will not pretend it is.**
    `construct_bond_ladder` builds GIC rungs by default, and a non-redeemable GIC
    has no secondary market: its "price" does not move when rates do, because
    there is no price. Duration on that instrument is real but it measures
    OPPORTUNITY COST — the return given up by being locked in — not a mark you
    could realise. Every payload states `marked_to_market` and the shock table on
    a non-marketable rung is labelled accordingly, instead of reporting a paper
    loss the holder can never take.

  * **Zero bond holdings is a MEASUREMENT, and it must not read like silence.**
    An all-equity portfolio holds no bond at all, so `portfolio_rate_sensitivity()`
    returns no sensitivity figure for as long as that stays true. It therefore
    separates three states that a single "0" would
    conflate: holdings read and classified as non-fixed-income, holdings that
    could NOT be classified, and an unreadable portfolio. A zero standing on
    three unclassified holdings is not a zero — this codebase has shipped that
    exact conflation in Market Pulse, in 5.4's tone verdict and in 5.9's
    `Unknown` rows, and it is the reason `unclassified` rides on the payload.

  * **YTM is an input, never a guess.** Pricing a bond needs its yield, and
    nothing in this app stores one per holding. Where a yield is not supplied the
    module says `yield_unknown` and declines, rather than substituting the
    curve's nearest tenor and presenting the result as that holding's duration.
"""

from __future__ import annotations

from typing import Any

from tools.exception_logger import log_exceptions

# AUTHORED CONSTANTS (2.7). The shock ladder the ±100bp table is built around.
# Symmetric on purpose: convexity is the asymmetry, and it is only visible when
# the same magnitude is shown in both directions.
SHOCKS_BP: tuple[int, ...] = (-200, -100, -50, -25, 25, 50, 100, 200)

# Coupon frequency by instrument wording. A GIC accrues and pays at maturity,
# which is a zero-coupon shape; a Treasury or corporate bond pays semi-annually.
_DEFAULT_FREQUENCY = {
    "GIC": 0,          # 0 = zero-coupon / pay-at-maturity
    "TREASURY": 2,
    "CORPORATE BOND": 2,
    "CORPORATE": 2,
    "BOND": 2,
}

# Instruments with no secondary market. Duration is still meaningful for them as
# opportunity cost; a mark-to-market price change is not.
_NON_MARKETABLE = {"GIC"}

# Symbols whose fixed-income nature is known without a network call. Deliberately
# short and deliberately not exhaustive — it exists so the offline path can
# classify the common cases, NOT so an absent symbol can be called an equity.
# Anything not in here and not resolvable goes to `unclassified`, never to zero.
_KNOWN_BOND_FUNDS: frozenset[str] = frozenset({
    # US aggregate / treasury / credit
    "AGG", "BND", "BNDX", "BIV", "BSV", "BLV", "GOVT", "IEF", "IEI", "SHY", "TLT",
    "TLH", "SHV", "BIL", "SGOV", "VGSH", "VGIT", "VGLT", "VCSH", "VCIT", "VCLT",
    "LQD", "HYG", "JNK", "USHY", "SJNK", "MUB", "TIP", "VTIP", "SCHZ", "SPAB",
    "SPTL", "SPTS", "STIP", "EMB", "PCY", "FLOT", "USFR", "TFLO",
    # Canadian listings
    "XBB.TO", "ZAG.TO", "VAB.TO", "XSB.TO", "ZCS.TO", "XLB.TO", "ZFL.TO",
    "XCB.TO", "ZCM.TO", "XHY.TO", "ZHY.TO", "VSB.TO", "VSC.TO", "CBO.TO",
    "CLF.TO", "XSH.TO", "ZDB.TO", "HFR.TO",
})

# yfinance's own vocabulary for the same fact. Matched case-insensitively as a
# substring of `category` / `quoteType` / `longBusinessSummary` is NOT done — a
# substring match on free text is how "ISA" matched "Visa" in 4.7. These are
# compared against the `category` field exactly, lowercased.
_BOND_CATEGORY_TOKENS: frozenset[str] = frozenset({
    "bond", "treasury", "government", "corporate", "municipal", "fixed income",
    "high yield", "intermediate", "short-term", "long-term", "ultrashort",
})


# ---------------------------------------------------------------------------
# Core pricing
# ---------------------------------------------------------------------------
def _cashflows(face: float, coupon_rate: float, years: float,
               frequency: int) -> list[tuple[float, float]]:
    """`(time_in_years, cashflow)` pairs, earliest first.

    A frequency of 0 means pay-at-maturity (a GIC, a strip, a zero): one flow,
    principal plus accrued, at `years`. Compounded annually, which is how a
    Canadian posted GIC rate is quoted.
    """
    if frequency <= 0:
        return [(years, face * (1.0 + coupon_rate) ** years)]

    periods = max(1, int(round(years * frequency)))
    coupon = face * coupon_rate / frequency
    flows = [((t / frequency), coupon) for t in range(1, periods + 1)]
    flows[-1] = (flows[-1][0], flows[-1][1] + face)
    return flows


def _present_value(flows: list[tuple[float, float]], ytm: float,
                   frequency: int) -> float:
    """Discount `flows` at `ytm`, compounding at `frequency` (annually if 0)."""
    m = frequency if frequency > 0 else 1
    rate = ytm / m
    if rate <= -1.0:
        # A discount factor of zero or negative is not a price, it is a domain
        # error. Returning 0.0 here would render as "this bond is worthless".
        raise ValueError(f"yield {ytm:.4f} is below the -100%/period floor")
    return sum(cf / (1.0 + rate) ** (t * m) for t, cf in flows)


@log_exceptions()
def bond_metrics(coupon_rate: float, ytm: float, years: float,
                 face: float = 100.0, frequency: int = 2) -> dict[str, Any]:
    """Price, Macaulay/modified duration, convexity and DV01 for one instrument.

    Rates are DECIMALS (0.045, not 4.5). Every duration figure is in years and
    every convexity figure is in years-squared, which is what makes the
    second-order term dimensionally correct in `shock_table`.

    Returns an `error` key rather than raising, and never returns a partial set
    of numbers: a duration without the price it was computed from is not
    checkable by the caller.
    """
    try:
        years = float(years)
        face = float(face)
        coupon_rate = float(coupon_rate)
        ytm = float(ytm)
        frequency = int(frequency)
    except (TypeError, ValueError):
        return {"error": "coupon_rate, ytm, years, face and frequency must be numeric"}

    if years <= 0:
        return {"error": "years must be positive — a matured instrument has no duration"}
    if face <= 0:
        return {"error": "face must be positive"}

    m = frequency if frequency > 0 else 1
    try:
        flows = _cashflows(face, coupon_rate, years, frequency)
        price = _present_value(flows, ytm, frequency)
    except ValueError as e:
        return {"error": str(e)}

    if price <= 0:
        return {"error": "discounted price is not positive — check the yield"}

    rate = ytm / m
    weighted_time = 0.0
    convexity_num = 0.0
    for t, cf in flows:
        periods = t * m
        pv = cf / (1.0 + rate) ** periods
        weighted_time += t * pv
        # Σ n(n+1)·CF/(1+r)^(n+2) / m², the standard periodic convexity numerator.
        convexity_num += periods * (periods + 1) * cf / (1.0 + rate) ** (periods + 2)

    macaulay = weighted_time / price
    modified = macaulay / (1.0 + rate)
    convexity = convexity_num / (price * m * m)

    return {
        "price": round(price, 6),
        "face": face,
        "coupon_rate_pct": round(coupon_rate * 100, 4),
        "ytm_pct": round(ytm * 100, 4),
        "years_to_maturity": years,
        "frequency": frequency,
        "macaulay_duration": round(macaulay, 6),
        "modified_duration": round(modified, 6),
        "convexity": round(convexity, 6),
        # The price move for one basis point, in currency units of `face`. The
        # figure a desk actually trades off.
        "dv01": round(modified * price * 0.0001, 6),
        "zero_coupon": frequency <= 0 or coupon_rate == 0,
    }


# ---------------------------------------------------------------------------
# The ±100bp shock table
# ---------------------------------------------------------------------------
@log_exceptions()
def shock_table(coupon_rate: float, ytm: float, years: float,
                face: float = 100.0, frequency: int = 2,
                shocks_bp: tuple[int, ...] = SHOCKS_BP,
                marked_to_market: bool = True) -> dict[str, Any]:
    """What each parallel curve shift does to this instrument's price.

    Three numbers per row, and the reason all three are here rather than one:

      * `exact_pct` re-prices the cashflows at the shocked yield. This is the
        answer.
      * `duration_only_pct` is `-D·Δy`, the linear estimate. Symmetric by
        construction, which is precisely what a bond is not.
      * `duration_convexity_pct` adds `½·C·Δy²`. The gap between this and
        `duration_only_pct` IS the convexity, in the units the user reads.

    `approximation_error_pct` is `exact − duration_convexity`, and it is on the
    payload because a second-order estimate is still an estimate: at ±25bp it is
    noise and at ±200bp it is visible. A table that showed only the approximation
    would make that invisible at exactly the shock size where it matters.

    `marked_to_market=False` (a GIC, a held-to-maturity private placement) keeps
    every figure but relabels what it means: there is no secondary market, so the
    move is the opportunity cost of being locked in, not a loss anyone can take.
    """
    base = bond_metrics(coupon_rate, ytm, years, face, frequency)
    if "error" in base:
        return base

    modified = base["modified_duration"]
    convexity = base["convexity"]
    price = base["price"]
    flows = _cashflows(face, coupon_rate, years, frequency)

    rows: list[dict[str, Any]] = []
    for bp in shocks_bp:
        dy = bp / 10000.0
        try:
            shocked = _present_value(flows, ytm + dy, frequency)
        except ValueError:
            rows.append({
                "shock_bp": bp,
                "error": "shocked yield falls below the -100%/period floor",
            })
            continue

        exact_pct = (shocked / price - 1.0) * 100.0
        dur_pct = -modified * dy * 100.0
        dur_cvx_pct = (-modified * dy + 0.5 * convexity * dy * dy) * 100.0

        rows.append({
            "shock_bp": bp,
            "shocked_ytm_pct": round((ytm + dy) * 100, 4),
            "price": round(shocked, 6),
            "exact_pct": round(exact_pct, 4),
            "duration_only_pct": round(dur_pct, 4),
            "duration_convexity_pct": round(dur_cvx_pct, 4),
            # Positive means duration+convexity UNDERSTATED the true move.
            "approximation_error_pct": round(exact_pct - dur_cvx_pct, 4),
            "value_change": round(shocked - price, 6),
        })

    return {
        **base,
        "marked_to_market": marked_to_market,
        "shocks": rows,
        "basis": "computed",
        "basis_note": (
            "Prices are discounted cashflows at the stated yield — computed, not "
            "an authored scenario constant. What IS assumed is the shape of the "
            "move: every row is a PARALLEL shift of the whole curve. A real "
            "steepening or flattening moves the front and the back by different "
            "amounts and this table does not model that."
            if marked_to_market else
            "Prices are discounted cashflows at the stated yield. This instrument "
            "has NO SECONDARY MARKET, so these are not marks you could realise — "
            "they are the opportunity cost of being locked in at the original "
            "rate while the curve moved. A non-redeemable GIC still pays its "
            "stated rate to maturity no matter what this table says."
        ),
    }


# ---------------------------------------------------------------------------
# Ladder-level sensitivity
# ---------------------------------------------------------------------------
def _frequency_for(investment_type: str) -> int:
    return _DEFAULT_FREQUENCY.get(str(investment_type or "").strip().upper(), 2)


def _is_marketable(investment_type: str) -> bool:
    return str(investment_type or "").strip().upper() not in _NON_MARKETABLE


@log_exceptions()
def ladder_rate_sensitivity(amount: float = 100000.0,
                            investment_type: str = "GIC",
                            currency: str = "CAD") -> dict[str, Any]:
    """Duration and convexity of the 5-year ladder `construct_bond_ladder` builds.

    The ladder tool has always priced its rungs off the live curve and never said
    what a curve move does to them. This is that half.

    A ladder's duration is the market-value-weighted average of its rungs'
    durations; its convexity is the market-value-weighted average of theirs. Both
    weights are MARKET value, not principal — equal principal per rung does not
    mean equal value per rung once the rungs are discounted at different yields.
    """
    from tools.fixed_income import _fetch_current_rates

    try:
        amount = float(amount)
    except (TypeError, ValueError):
        return {"error": "amount must be numeric"}
    if amount <= 0:
        return {"error": "amount must be positive"}

    rates, data_note = _fetch_current_rates(investment_type, currency)
    frequency = _frequency_for(investment_type)
    marketable = _is_marketable(investment_type)
    rung_principal = amount / 5.0

    rungs: list[dict[str, Any]] = []
    total_value = 0.0
    for year in range(1, 6):
        rate_pct = rates.get(year)
        if rate_pct is None:
            rungs.append({"maturity_years": year, "error": "no rate for this tenor"})
            continue
        rate = float(rate_pct) / 100.0
        # A ladder rung is bought AT the prevailing rate, so it prices at par and
        # its coupon equals its yield. That is an assumption about how the rung
        # was acquired, and it is stated rather than buried: a rung bought last
        # year at a different rate has a different duration.
        m = bond_metrics(coupon_rate=rate, ytm=rate, years=float(year),
                         face=rung_principal, frequency=frequency)
        if "error" in m:
            rungs.append({"maturity_years": year, "error": m["error"]})
            continue
        total_value += m["price"]
        rungs.append({
            "maturity_years": year,
            "rate_pct": round(rate * 100, 4),
            "principal": round(rung_principal, 2),
            "value": m["price"],
            "modified_duration": m["modified_duration"],
            "convexity": m["convexity"],
        })

    priced = [r for r in rungs if "error" not in r]
    if not priced or total_value <= 0:
        return {
            "error": "no rung could be priced",
            "rungs": rungs,
            "data_source": data_note,
        }

    weighted_duration = sum(r["modified_duration"] * r["value"] for r in priced) / total_value
    weighted_convexity = sum(r["convexity"] * r["value"] for r in priced) / total_value

    ladder_shocks: list[dict[str, Any]] = []
    for bp in SHOCKS_BP:
        dy = bp / 10000.0
        pct = (-weighted_duration * dy + 0.5 * weighted_convexity * dy * dy) * 100.0
        ladder_shocks.append({
            "shock_bp": bp,
            "estimated_pct": round(pct, 4),
            "estimated_value_change": round(total_value * pct / 100.0, 2),
        })

    return {
        "strategy": f"5-Year {investment_type} Ladder ({currency})",
        "total_value": round(total_value, 2),
        "modified_duration": round(weighted_duration, 4),
        "convexity": round(weighted_convexity, 4),
        "dv01": round(weighted_duration * total_value * 0.0001, 2),
        "rungs": rungs,
        "rungs_priced": len(priced),
        "shocks": ladder_shocks,
        "marked_to_market": marketable,
        "data_source": data_note,
        "assumption": (
            "Each rung is priced at par — bought today at the prevailing rate for "
            "its tenor, so coupon equals yield. A rung bought earlier at a "
            "different rate carries a different duration than the one shown here."
        ),
        "note": (
            "Ladder duration is the market-value-weighted average of the rungs. "
            + ("These are marks you could realise." if marketable else
               "A non-redeemable GIC has no secondary market: this measures the "
               "opportunity cost of being locked in, not a loss you can take. "
               "The ladder still pays its stated rates to maturity.")
        ),
    }


# ---------------------------------------------------------------------------
# Portfolio-level — and the three states a "0" would hide
# ---------------------------------------------------------------------------
def classify_fixed_income(symbol: str,
                          info: dict[str, Any] | None = None) -> dict[str, Any]:
    """Whether one symbol is fixed income. Three answers, and `None` is one of them.

    Returns `{"is_bond": True|False|None, "basis": ...}`. `None` means *not
    determined* and is never collapsed into `False` upstream: a portfolio scored
    as having no bonds because three symbols failed to resolve is a different
    claim from one that has none, and only the first is a data problem.
    """
    sym = str(symbol or "").upper().strip()
    if not sym:
        return {"is_bond": None, "basis": "no symbol"}
    if sym in _KNOWN_BOND_FUNDS:
        return {"is_bond": True, "basis": "known fixed-income fund"}

    if info is None:
        return {"is_bond": None, "basis": "not in the known-fund table and no metadata supplied"}

    quote_type = str(info.get("quoteType") or "").upper().strip()
    category = str(info.get("category") or "").lower().strip()

    if quote_type == "BOND":
        return {"is_bond": True, "basis": "quoteType=BOND"}
    if category:
        # Exact token containment on the CATEGORY field only. Not a substring
        # sweep over free text — 4.7 shipped "ISA" matching "Visa" that way.
        tokens = {t.strip() for t in category.replace("/", " ").replace("-", " ").split()}
        if "bond" in tokens or category in _BOND_CATEGORY_TOKENS:
            return {"is_bond": True, "basis": f"category={category!r}"}
        return {"is_bond": False, "basis": f"category={category!r}"}
    if quote_type in {"EQUITY", "CRYPTOCURRENCY", "CURRENCY", "FUTURE", "INDEX"}:
        return {"is_bond": False, "basis": f"quoteType={quote_type}"}
    if quote_type == "ETF":
        # An ETF with no category is genuinely undetermined. Bond ETFs and equity
        # ETFs share a quoteType, so this is the case that must NOT default.
        return {"is_bond": None, "basis": "ETF with no category — cannot tell"}
    return {"is_bond": None, "basis": f"quoteType={quote_type or 'unknown'}"}


@log_exceptions()
def portfolio_rate_sensitivity(
    yields: dict[str, float] | None = None,
    maturities: dict[str, float] | None = None,
    info_fn: Any = None,
) -> dict[str, Any]:
    """The portfolio's exposure to a parallel curve shift — or why there is none.

    Read `status` first:
      * ``no_portfolio``     — the book could not be read. NOT a bond-free book.
      * ``no_fixed_income``  — every holding was classified and none is a bond.
                               `unclassified` is 0 in this state, by definition.
      * ``undetermined``     — nothing classified as a bond, but some holdings
                               could not be classified either. This is the state
                               a bare "0 bonds" would have hidden.
      * ``yields_missing``   — bonds were found, and no yield is on file for
                               them, so no duration can be computed. The holdings
                               are listed; the number is withheld.
      * ``measured``         — `modified_duration` is the book's fixed-income
                               duration and `shocks` is what a move costs.

    `yields` and `maturities` are per-symbol overrides in DECIMAL and YEARS. They
    exist because nothing in this app stores a yield-to-maturity or an effective
    duration per holding, and substituting the curve's nearest tenor would
    produce a confident number about a bond nobody measured.

    `info_fn` resolves a symbol's instrument metadata and defaults to a cached
    provider lookup. It has to fetch SOMETHING: without metadata every symbol
    outside the known-fund table is `unclassified`, so the endpoint would return
    `undetermined` forever on any book with no recognised bond ticker in it —
    which is a correct refusal built on a question the engine never asked. Found
    on the first live read; the offline tests exercise `sensitivity_over`
    directly and never reach this.
    """
    from tools.portfolio_csv import load_portfolio

    holdings = load_portfolio()
    if isinstance(holdings, dict) and "error" in holdings:
        return {"status": "no_portfolio", "error": holdings["error"],
                "note": "The portfolio could not be read. This is not a report that "
                        "the book holds no bonds."}
    if not isinstance(holdings, list):
        return {"status": "no_portfolio",
                "note": "The portfolio could not be read. This is not a report that "
                        "the book holds no bonds."}

    rows: list[dict[str, Any]] = []
    for h in holdings:
        if not isinstance(h, dict) or "_sync_errors" in h:
            # `load_portfolio` appends a metadata sentinel to the holdings list;
            # counting it as a position adds one phantom row every single call.
            continue
        symbol = str(h.get("symbol") or "").upper().strip()
        if not symbol:
            continue
        resolve = info_fn if info_fn is not None else _default_info
        try:
            info = resolve(symbol)
        except Exception:  # noqa: BLE001 — one bad lookup must not abort the scan
            info = None
        rows.append({"symbol": symbol, "shares": h.get("shares"),
                     "account": h.get("account"), "currency": h.get("currency"),
                     "private": bool(h.get("is_private_asset")), "info": info})

    return sensitivity_over(rows, yields=yields, maturities=maturities)


def _default_info(symbol: str) -> dict[str, Any] | None:
    """The two metadata fields `classify_fixed_income` reads, cached for a day.

    Only `quoteType` and `category` are kept. A provider payload is ~150 keys and
    caching all of them per holding would put a megabyte of unrelated fundamentals
    in the daily store to answer one boolean.

    `stamp=False` on purpose: the decorator's `_as_of` injection is right for a
    payload someone renders and wrong for one that is read by exact key — a stamp
    landing in a metadata dict is the shape of the bug that took
    `check_portfolio_allocation` down for two days, where `_as_of` arrived as a
    sector name and `float + str` raised.
    """
    from tools.cache import cached

    @cached(key_func=lambda s: f"bond_classify_info_{s}", ttl=86400, stamp=False)
    def _fetch(sym: str) -> dict[str, Any]:
        import yfinance as yf

        info = yf.Ticker(sym).info or {}
        return {"quoteType": info.get("quoteType"), "category": info.get("category")}

    try:
        return _fetch(symbol)
    except Exception:  # noqa: BLE001 — an unreachable provider leaves the symbol
        # UNCLASSIFIED, which is the honest state. It must never become `False`.
        return None


@log_exceptions()
def sensitivity_over(rows: list[dict[str, Any]],
                     yields: dict[str, float] | None = None,
                     maturities: dict[str, float] | None = None) -> dict[str, Any]:
    """The duration engine over an explicit position list.

    Split out from `portfolio_rate_sensitivity` so `simulate_scenario` can run the
    same classification and the same refusals over the symbols it was ASKED
    about, rather than over the whole book. A scenario named for three tickers
    that silently reported the portfolio's duration would be answering a question
    nobody asked.

    Each row is ``{symbol, shares?, account?, currency?, info?}``.
    """
    yields = {str(k).upper(): float(v) for k, v in (yields or {}).items()}
    maturities = {str(k).upper(): float(v) for k, v in (maturities or {}).items()}

    bonds: list[dict[str, Any]] = []
    unclassified: list[dict[str, Any]] = []
    equities = 0
    positions = 0

    for h in rows:
        symbol = str(h.get("symbol") or "").upper().strip()
        if not symbol:
            continue
        positions += 1
        if h.get("private"):
            # A private holding has no ticker to look up, so "the provider did
            # not answer" is the wrong reason to print. It is unclassifiable by
            # nature rather than by a failed fetch, and only the accurate reason
            # tells the reader whether there is anything they can do about it.
            unclassified.append({
                "symbol": symbol,
                "reason": ("private asset — no instrument metadata exists, so "
                           "whether it holds fixed income is unknown to this app"),
            })
            continue
        verdict = classify_fixed_income(symbol, h.get("info"))
        if verdict["is_bond"] is True:
            bonds.append({"symbol": symbol, "basis": verdict["basis"],
                          "shares": h.get("shares"), "account": h.get("account"),
                          "currency": h.get("currency")})
        elif verdict["is_bond"] is False:
            equities += 1
        else:
            unclassified.append({"symbol": symbol, "reason": verdict["basis"]})

    common = {
        "positions_read": positions,
        "bond_holdings": len(bonds),
        "non_bond_holdings": equities,
        "unclassified_holdings": len(unclassified),
        "unclassified": unclassified[:25],
    }

    if not bonds and unclassified:
        return {
            "status": "undetermined",
            **common,
            "note": (
                f"No holding was identified as fixed income, but {len(unclassified)} "
                f"of {positions} could not be classified either — so this is NOT a "
                "measured zero. Supply instrument metadata (or add the symbol to the "
                "known-fund table) before reading it as a bond-free book."
            ),
        }
    if not bonds:
        return {
            "status": "no_fixed_income",
            **common,
            "note": (
                f"All {positions} holding(s) were classified and none is fixed "
                "income. Duration and convexity have nothing to measure here — "
                "this book carries no direct exposure to a parallel curve shift. "
                "Equity duration to rates is a different (and much less "
                "well-defined) quantity and is not what this module computes."
            ),
        }

    missing_inputs = [b["symbol"] for b in bonds
                      if b["symbol"] not in yields or b["symbol"] not in maturities]
    if missing_inputs:
        return {
            "status": "yields_missing",
            **common,
            "bonds": bonds,
            "missing_inputs": missing_inputs,
            "note": (
                f"{len(bonds)} fixed-income holding(s) found, and "
                f"{len(missing_inputs)} of them have no yield-to-maturity and/or "
                "effective maturity on file. Nothing in this app records either "
                "per holding. Duration is withheld rather than computed off the "
                "curve's nearest tenor, which would be a confident number about a "
                "bond nobody measured. Supply `yields` and `maturities` to compute it."
            ),
        }

    # Weight by market value where a price is available; fall back to equal weight
    # and SAY SO, because a duration weighted by the wrong thing is not an
    # approximation of the right one.
    priced: list[dict[str, Any]] = []
    total_value = 0.0
    equal_weighted = False
    for b in bonds:
        sym = b["symbol"]
        ytm = yields[sym]
        years = maturities[sym]
        m = bond_metrics(coupon_rate=ytm, ytm=ytm, years=years, face=100.0, frequency=2)
        if "error" in m:
            b["error"] = m["error"]
            continue
        shares = b.get("shares")
        try:
            value = float(shares) * m["price"] / 100.0 if shares else 0.0
        except (TypeError, ValueError):
            value = 0.0
        priced.append({**b, "modified_duration": m["modified_duration"],
                       "convexity": m["convexity"], "value": round(value, 2)})
        total_value += value

    if not priced:
        return {"status": "yields_missing", **common, "bonds": bonds,
                "note": "No fixed-income holding could be priced from the supplied inputs."}

    if total_value <= 0:
        equal_weighted = True
        weights = [1.0 / len(priced)] * len(priced)
    else:
        weights = [p["value"] / total_value for p in priced]

    duration = sum(p["modified_duration"] * w for p, w in zip(priced, weights))
    convexity = sum(p["convexity"] * w for p, w in zip(priced, weights))

    shocks = []
    for bp in SHOCKS_BP:
        dy = bp / 10000.0
        pct = (-duration * dy + 0.5 * convexity * dy * dy) * 100.0
        shocks.append({
            "shock_bp": bp,
            "estimated_pct": round(pct, 4),
            "estimated_value_change": (round(total_value * pct / 100.0, 2)
                                       if total_value > 0 else None),
        })

    return {
        "status": "measured",
        **common,
        "bonds": priced,
        "fixed_income_value": round(total_value, 2) if total_value > 0 else None,
        "modified_duration": round(duration, 4),
        "convexity": round(convexity, 4),
        "dv01": round(duration * total_value * 0.0001, 4) if total_value > 0 else None,
        "shocks": shocks,
        "equal_weighted": equal_weighted,
        "note": (
            "Duration and convexity cover the FIXED-INCOME SLEEVE only, not the "
            "whole book. "
            + ("Weighted equally: no market value was available for these "
               "holdings, so this is not a value-weighted figure."
               if equal_weighted else
               "Weighted by market value.")
        ),
    }


# ---------------------------------------------------------------------------
# 4.8 → simulate_scenario's rate_hike case
# ---------------------------------------------------------------------------
@log_exceptions()
def rate_hike_duration_leg(shock_bp: int = 100,
                           yields: dict[str, float] | None = None,
                           maturities: dict[str, float] | None = None,
                           info_fn: Any = None,
                           rows: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """The fixed-income half of a rate-hike scenario, for `simulate_scenario`.

    `simulate_scenario`'s rate_hike case is an authored -20% equity constant with
    a beta multiplier, and it has never had a rate leg at all — which for a book
    with bonds in it is the half that a rate hike is actually ABOUT. This supplies
    it, and stamps `basis: "computed"` where the equity leg stamps `basis:
    "authored constant"`, so a reader can see that the two halves of one payload
    do not have the same standing.

    On a book with no bonds it returns `applicable: False` and the reason. That
    is the honest answer for the live portfolio today and it must not read as a
    0% impact, which is what an absent key would have implied.

    `rows` scopes the leg to an explicit position list — the symbols a scenario
    named, with the metadata its caller already fetched. Omit it to read the
    whole book.
    """
    sensitivity = (sensitivity_over(rows, yields=yields, maturities=maturities)
                   if rows is not None else
                   portfolio_rate_sensitivity(yields=yields, maturities=maturities,
                                              info_fn=info_fn))
    status = sensitivity.get("status")

    if status != "measured":
        return {
            "applicable": False,
            "status": status,
            "reason": sensitivity.get("note") or sensitivity.get("error"),
            "shock_bp": shock_bp,
            "basis": "computed",
            "basis_note": (
                "No duration leg is included in this scenario. That is a statement "
                "about the book, not an estimate of zero impact — see `reason`."
            ),
        }

    dy = shock_bp / 10000.0
    duration = sensitivity["modified_duration"]
    convexity = sensitivity["convexity"]
    pct = (-duration * dy + 0.5 * convexity * dy * dy) * 100.0
    value = sensitivity.get("fixed_income_value")

    return {
        "applicable": True,
        "shock_bp": shock_bp,
        "modified_duration": duration,
        "convexity": convexity,
        "fixed_income_value": value,
        "estimated_pct": round(pct, 4),
        "estimated_value_change": round(value * pct / 100.0, 2) if value else None,
        "bond_holdings": sensitivity["bond_holdings"],
        "basis": "computed",
        "basis_note": (
            f"A {shock_bp:+d}bp PARALLEL shift applied to the fixed-income sleeve's "
            f"measured duration and convexity. Computed, unlike the equity leg of "
            f"this scenario, which is an authored constant. The parallel assumption "
            f"is the modelling choice here: a real hike usually flattens the curve, "
            f"which moves the front of the ladder more than the back."
        ),
    }


if __name__ == "__main__":  # pragma: no cover — operator convenience
    import json

    print(json.dumps(shock_table(0.045, 0.045, 10.0), indent=2))
    print(json.dumps(portfolio_rate_sensitivity(), indent=2))
