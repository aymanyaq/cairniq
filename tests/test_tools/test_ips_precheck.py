"""Deterministic IPS compliance pre-check (Advisor Roadmap Theme 2.2)."""
import copy

import pytest

import tools.ips_precheck as ips

_CTX = {
    "base_currency": "USD",
    "total_value_base": 100_000.0,
    "holdings": [
        {"symbol": "AAPL", "account": "TFSA", "value_base": 5_000.0, "current_price": "$200.00"},
        {"symbol": "XIC.TO", "account": "RRSP", "value_base": 20_000.0, "current_price": 40.0},
        {"symbol": "CASH", "account": "Margin", "value_base": 10_000.0, "current_price": 1.0},
    ],
}

_CURRENT_SECTORS = {
    "Technology": 0.20,
    "Financial Services": 0.30,
    "Healthcare": 0.35,
    "Energy": 0.05,
    "Cash": 0.10,
}


def _make_allocation_fake(current_map, cand_maps, calls):
    def fake(symbols, amounts, allow_network):
        calls.append((tuple(symbols), allow_network))
        if len(symbols) == 1 and allow_network:
            spec = cand_maps.get(symbols[0], {
                "map": {"Technology": 1.0}, "source": "API", "details": "Single Stock",
            })
            return {
                "sector_allocation_raw": dict(spec["map"]),
                "holding_details": [{
                    "symbol": symbols[0],
                    "classification_source": spec["source"],
                    "sector_details": spec["details"],
                }],
            }
        return {"sector_allocation_raw": dict(current_map), "holding_details": []}
    return fake


# The caps these tests exercise. They must be stated explicitly, because the
# module has no defaults: a profile that sets nothing is unconstrained, which is
# what test_no_stated_caps_is_inert covers.
_CAPS = {
    "max_position_pct": 10.0,
    "max_fund_position_pct": 25.0,
    "max_sector_pct": 30.0,
    "max_risk_per_trade_pct": 2.0,
}


def _setup(monkeypatch, tmp_path, current_map=None, cand_maps=None, ctx=None, constraints=None):
    calls = []
    monkeypatch.setattr(ips, "_get_decision_context", lambda: copy.deepcopy(ctx or _CTX))
    monkeypatch.setattr(ips, "_is_cash_symbol", lambda s: s == "CASH")
    monkeypatch.setattr(ips, "_get_quote_price", lambda s: (None, ""))
    monkeypatch.setattr(
        ips, "_get_allocation",
        _make_allocation_fake(current_map or _CURRENT_SECTORS, cand_maps or {}, calls),
    )
    risk_constraints = {**_CAPS, **(constraints or {})}
    monkeypatch.setattr(ips, "_load_memory", lambda: {"risk_constraints": copy.deepcopy(risk_constraints)})
    return calls


def _rows_by_check(result, ticker=None):
    rows = result["rows"]
    if ticker:
        rows = [r for r in rows if ticker in r["trade"]]
    return {r["check"]: r for r in rows}


def _assert_said_nothing(result):
    """Nothing was checked and nothing was claimed.

    Asserted field by field rather than against a whole-dict literal: the result
    carries the execution-readiness verdict too, and a literal here would make
    every future field an unrelated test failure — which is what it did.
    """
    assert result["trades"] == []
    assert result["rows"] == []
    assert result["violations"] == []
    assert result["block"] == ""


def test_no_proposed_trades_is_inert(monkeypatch, tmp_path):
    calls = _setup(monkeypatch, tmp_path)

    result = ips.run_ips_precheck("The market looks stretched and VIX is elevated. Keep holding your positions.")

    _assert_said_nothing(result)
    assert calls == []  # no sector/network work on non-trade turns


def test_buy_within_caps_all_pass(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path)

    result = ips.run_ips_precheck("I recommend you buy $5,000 of NVDA here, entry at $200, stop-loss at $180.")

    assert result["violations"] == []
    checks = _rows_by_check(result)
    assert checks["position cap"]["verdict"] == "PASS"
    assert "5.0% post-trade" in checks["position cap"]["computed"]
    assert "cash-funded" in checks["position cap"]["computed"]
    assert checks["sector cap"]["verdict"] == "PASS"
    assert "Technology" in checks["sector cap"]["computed"]
    assert checks["dollar-at-risk"]["verdict"] == "PASS"
    assert "$500" in checks["dollar-at-risk"]["computed"]
    assert "CONFIRM" in result["block"]
    assert "| BUY NVDA $5,000 |" in result["block"]


def test_position_cap_breach_fails(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path)

    result = ips.run_ips_precheck("Add $9,000 to AAPL immediately.")

    checks = _rows_by_check(result)
    assert checks["position cap"]["verdict"] == "FAIL"
    assert "14.0% post-trade" in checks["position cap"]["computed"]
    assert len(result["violations"]) == 1
    assert "10% single name cap" in result["violations"][0]
    assert checks["account location"]["computed"] == "currently held in: TFSA"


def test_fund_cap_applies_to_funds(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path, cand_maps={
        "XIC.TO": {
            "map": {"Financial Services": 0.30, "Energy": 0.15, "Industrials": 0.12,
                    "Basic Materials": 0.10, "Other": 0.33},
            "source": "Fund Decomposition DB", "details": "Decomposed",
        },
    })

    result = ips.run_ips_precheck("Deploy $18,000 into XIC.TO.")

    checks = _rows_by_check(result)
    assert checks["position cap"]["verdict"] == "FAIL"
    assert "fund" in checks["position cap"]["computed"]
    assert "new money" in checks["position cap"]["computed"]  # cash 10k < 18k
    assert any("25% fund cap" in v for v in result["violations"])
    assert checks["sector cap"]["verdict"] == "PASS"  # FS lands exactly at 30.0%


def test_sector_cap_breach_fails(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path, current_map={
        "Technology": 0.28, "Financial Services": 0.30, "Healthcare": 0.27,
        "Energy": 0.05, "Cash": 0.10,
    })

    result = ips.run_ips_precheck("Buy $10,000 of NVDA today.")

    checks = _rows_by_check(result)
    assert checks["position cap"]["verdict"] == "PASS"  # exactly at the 10% cap
    assert checks["sector cap"]["verdict"] == "FAIL"
    assert "38.0% post-trade" in checks["sector cap"]["computed"]
    assert len(result["violations"]) == 1
    assert "30% sector cap" in result["violations"][0]


def test_pct_of_portfolio_size_parses(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path)

    result = ips.run_ips_precheck("Allocate 5% of your portfolio to NVDA.")

    assert result["trades"][0]["size_usd"] == 5000.0
    checks = _rows_by_check(result)
    assert checks["position cap"]["verdict"] == "PASS"
    assert "5% of portfolio" in checks["position cap"]["trade"]


def test_negation_and_third_party_buying_ignored(monkeypatch, tmp_path):
    calls = _setup(monkeypatch, tmp_path)

    result = ips.run_ips_precheck(
        "Do not buy NVDA at these levels. Heavy insider buying in AMD is notable but not actionable."
    )

    assert result["trades"] == []
    assert result["block"] == ""
    assert calls == []


def test_unsized_buy_yields_not_evaluated_rows(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path)

    result = ips.run_ips_precheck("Consider adding NVDA on the next pullback.")

    assert result["violations"] == []
    checks = _rows_by_check(result)
    assert checks["position cap"]["verdict"] == "NOT_EVALUATED"
    assert "headroom to 10% cap ≈ $10,000" in checks["position cap"]["computed"]
    assert checks["sector cap"]["verdict"] == "NOT_EVALUATED"
    assert checks["dollar-at-risk"]["verdict"] == "NOT_EVALUATED"
    assert "<ips_precheck>" in result["block"]


def test_dollar_after_price_cue_is_not_a_size(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path)

    result = ips.run_ips_precheck("Buy NVDA at $200.")

    assert result["trades"][0]["size_usd"] is None
    assert result["violations"] == []
    assert _rows_by_check(result)["position cap"]["verdict"] == "NOT_EVALUATED"


def test_restricted_symbol_fails(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path, constraints={"restricted_symbols": ["NVDA"]})

    result = ips.run_ips_precheck("Buy $1,000 of NVDA.")

    checks = _rows_by_check(result)
    assert checks["restricted list"]["verdict"] == "FAIL"
    assert any("restricted list" in v for v in result["violations"])


def test_constraints_overlay_raises_cap(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path, constraints={"max_position_pct": 20})

    result = ips.run_ips_precheck("Add $9,000 to AAPL immediately.")

    assert _rows_by_check(result)["position cap"]["verdict"] == "PASS"
    assert result["violations"] == []


def test_disabled_constraints_noop(monkeypatch, tmp_path):
    calls = _setup(monkeypatch, tmp_path, constraints={"enabled": False})

    result = ips.run_ips_precheck("Buy $5,000 of NVDA.")

    _assert_said_nothing(result)
    assert calls == []


def test_dollar_at_risk_breach_fails(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path)

    result = ips.run_ips_precheck("Buy $8,000 of NVDA at $200, stop at $100.")

    checks = _rows_by_check(result)
    assert checks["position cap"]["verdict"] == "PASS"
    assert checks["dollar-at-risk"]["verdict"] == "FAIL"
    assert "$4,000 at risk" in checks["dollar-at-risk"]["computed"]
    assert any("max-risk rule" in v for v in result["violations"])


def test_shares_size_converts_via_held_price(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path)

    result = ips.run_ips_precheck("Buy 40 shares of AAPL here.")

    assert result["trades"][0]["size_usd"] == 8000.0  # 40 × $200.00 (string price coerced)
    checks = _rows_by_check(result)
    assert checks["position cap"]["verdict"] == "FAIL"  # 5k + 8k = 13%
    assert "40 shares (≈$8,000)" in checks["position cap"]["trade"]


# The draft shape that exposed the extraction bugs: a per-ticker shopping list
# (short rows, one name each), price ranges, a trigger price restated after a buy
# verb, and an explicit exclusion line. Synthetic names/amounts — fixtures never
# carry real holdings or balances.
_SHOPPING_LIST_DRAFT = """The highest-conviction name to accumulate is NVIDIA (NVDA) (Price: $211.99).
Ranked Shopping List:
NVDA: Entry $200-$210 | Stop $185.40 | Trigger: Start $210, add $200.
AMD: Entry $325-$335 | Stop $309.20 (2x ATR) | Trigger: Start $335, add $325.
Excluded: INTC and MU (broken growth theses).
Cash Plan: Deploy your $10,000.00 liquid cash in two tranches of $5,000.
"""


def test_shopping_list_rows_keep_their_own_levels(monkeypatch, tmp_path):
    """Each row's stop/entry must come from that row, not the one above it."""
    _setup(monkeypatch, tmp_path)

    trades = {
        t["ticker"]: t
        for t in ips.extract_proposed_trades(
            _SHOPPING_LIST_DRAFT, {"NVDA", "AMD", "INTC", "MU"}, 100_000.0
        )
    }

    assert trades["NVDA"]["stop"] == 185.40
    assert trades["NVDA"]["stated_entry"] == 200.0
    assert trades["AMD"]["stop"] == 309.20
    assert trades["AMD"]["stated_entry"] == 325.0


def test_excluded_names_are_not_proposed_trades(monkeypatch, tmp_path):
    """An avoid-list row must not inherit a buy verb from the row above it."""
    _setup(monkeypatch, tmp_path)

    trades = ips.extract_proposed_trades(
        _SHOPPING_LIST_DRAFT, {"NVDA", "AMD", "INTC", "MU"}, 100_000.0
    )

    assert {t["ticker"] for t in trades} == {"NVDA", "AMD"}


def test_price_levels_are_never_read_as_position_sizes(monkeypatch, tmp_path):
    """No per-name dollar size is stated, so every cap check must abstain rather
    than size the trade off a price level."""
    _setup(monkeypatch, tmp_path)

    result = ips.run_ips_precheck(
        _SHOPPING_LIST_DRAFT, {"NVDA", "AMD", "INTC", "MU"}
    )

    assert all(t["size_usd"] is None for t in result["trades"])
    assert result["violations"] == []  # nothing computed => nothing to fail
    for ticker in ("NVDA", "AMD"):
        assert _rows_by_check(result, ticker)["position cap"]["verdict"] == "NOT_EVALUATED"


def test_range_second_endpoint_is_not_a_size(monkeypatch, tmp_path):
    """Only the first endpoint sits behind a price cue; both are prices."""
    _setup(monkeypatch, tmp_path)

    result = ips.run_ips_precheck("NVDA: Entry $200-$210 | Stop $185.40.")

    assert result["trades"] == []  # no buy verb either — nothing to check


def test_proceeds_destination_is_not_a_proposed_buy(monkeypatch, tmp_path):
    """A buy verb pointing at cash or "elsewhere" names where the proceeds GO,
    not a trade in whatever ticker shares the window. Real regression: a pure
    trim instruction — "...reduce redundant exposure to AAPL and MSFT, locking
    in those gains to deploy elsewhere or hold in cash" — was extracted as a
    proposed BUY of MSFT with no size, which the judge reads as a Rule 3
    MAGNITUDE MISS on a trade the draft never proposed.
    """
    _setup(monkeypatch, tmp_path)

    for text in (
        "Trim ITOT and IVV to cut overlap with AAPL and MSFT, locking in those "
        "gains to deploy elsewhere or hold in cash.",
        "Sell the laggards and allocate into cash reserves until NVDA sets up.",
        "Exit the position and redeploy to the sidelines rather than chase AMD.",
    ):
        assert ips.extract_proposed_trades(
            text, {"AAPL", "MSFT", "ITOT", "IVV", "NVDA", "AMD"}, 100_000.0
        ) == [], text


def test_real_buy_with_cash_wording_still_extracts(monkeypatch, tmp_path):
    """The destination guard must not swallow a genuine deployment INTO a name."""
    _setup(monkeypatch, tmp_path)

    trades = ips.extract_proposed_trades(
        "Deploy $5,000 of your idle cash into NVDA.", {"NVDA"}, 100_000.0
    )

    assert [t["ticker"] for t in trades] == ["NVDA"]
    assert trades[0]["size_usd"] == 5000.0


def test_trigger_price_after_buy_verb_is_not_a_size(monkeypatch, tmp_path):
    """'add $200' where $200 is already the stated entry means add AT $200."""
    _setup(monkeypatch, tmp_path)

    trades = ips.extract_proposed_trades("Buy NVDA. Entry $200, stop $180, add $200.", {"NVDA"}, 100_000.0)

    assert trades[0]["size_usd"] is None
    assert trades[0]["stop"] == 180.0


# The draft shape that produced six NOT_EVALUATED rows on a fully-specified
# trade: each pick is a heading, and every number lives in the bullets beneath
# it. A line-anchored evidence window saw only the heading. Synthetic
# names/amounts — fixtures never carry real holdings or balances.
_SIZING_BLOCK_DRAFT = """\
1. AAPL (Apple) - Primary Buy

* Sector: Technology
* Current Price: $200.00
* Sizing & Magnitude Check:
   * Proposed Size: 40 shares.
   * Total Investment: $8,000.00
   * Structural Stop-Loss: $180.00 (2x ATR below the 20-day swing low).

2. AMD (Advanced Micro Devices) - Secondary Buy

* Current Price: $325.00
* Sizing & Magnitude Check:
   * Total Investment: $3,000.00
   * Structural Stop-Loss: $309.20
"""


def test_sizing_block_under_a_heading_is_evaluated(monkeypatch, tmp_path):
    """Numbers in the bullets beneath a pick's heading belong to that pick.

    Every check abstaining on a fully-sized trade is not a harmless abstention:
    a NOT_EVALUATED sizing row reads downstream as a Rule 3 MAGNITUDE MISS, so
    the draft was failed for omitting numbers it had in fact stated.
    """
    _setup(monkeypatch, tmp_path)

    result = ips.run_ips_precheck(_SIZING_BLOCK_DRAFT, {"AAPL", "AMD"})
    trades = {t["ticker"]: t for t in result["trades"]}

    assert trades["AAPL"]["size_usd"] == 8000.0
    assert trades["AAPL"]["stop"] == 180.0
    checks = _rows_by_check(result, "AAPL")
    assert checks["position cap"]["verdict"] == "FAIL"  # 5k held + 8k = 13% > 10%
    assert checks["dollar-at-risk"]["verdict"] == "PASS"  # $800 risk << 2% of 100k


def test_sizing_block_never_absorbs_the_next_tickers_numbers(monkeypatch, tmp_path):
    """The block window must still stop dead at the next candidate ticker."""
    _setup(monkeypatch, tmp_path)

    trades = {
        t["ticker"]: t
        for t in ips.extract_proposed_trades(_SIZING_BLOCK_DRAFT, {"AAPL", "AMD"}, 100_000.0)
    }

    assert trades["AMD"]["size_usd"] == 3000.0
    assert trades["AMD"]["stop"] == 309.20
    assert "$8,000" not in trades["AMD"]["window"]
    assert "$3,000" not in trades["AAPL"]["window"]


def test_labelled_size_is_not_confused_with_a_price(monkeypatch, tmp_path):
    """'Structural Stop-Loss: $180' is a price cue and must beat the size label."""
    _setup(monkeypatch, tmp_path)

    trades = ips.extract_proposed_trades(
        "Buy NVDA.\n   * Target Price: $260.00\n   * Stop-Loss: $180.00\n", {"NVDA"}, 100_000.0
    )

    assert trades[0]["size_usd"] is None
    assert trades[0]["stop"] == 180.0


def test_currency_labelled_size_converts_to_base(monkeypatch, tmp_path):
    """A USD size against a CAD portfolio must be converted before any cap.

    Taking "$12,500.00 USD" at face value against a CAD total understates every
    percent-of-portfolio check by the whole FX rate — a ~40% error that turns a
    cap breach into a computed PASS.
    """
    cad_ctx = {**copy.deepcopy(_CTX), "base_currency": "CAD"}
    _setup(monkeypatch, tmp_path, ctx=cad_ctx)
    monkeypatch.setattr(ips, "_get_fx_rate", lambda frm, to: 1.40 if (frm, to) == ("USD", "CAD") else 0.0)

    result = ips.run_ips_precheck("Buy NVDA.\n   * Total Investment: $9,000.00 USD\n", {"NVDA"})

    assert result["trades"][0]["size_usd"] == 12600.0  # 9,000 USD × 1.40
    row = _rows_by_check(result)["position cap"]
    assert "$12,600 CAD (draft stated $9,000 USD)" in row["trade"]
    # 12.6k of new money is 11.2% post-trade; the same figure read as CAD is a
    # cash-funded 9.0% — i.e. face value silently converts this breach to a PASS.
    assert row["verdict"] == "FAIL"


def test_unconvertible_currency_abstains_rather_than_comparing(monkeypatch, tmp_path):
    """No FX rate means no honest comparison — abstain, never compare as-is."""
    cad_ctx = {**copy.deepcopy(_CTX), "base_currency": "CAD"}
    _setup(monkeypatch, tmp_path, ctx=cad_ctx)
    monkeypatch.setattr(ips, "_get_fx_rate", lambda frm, to: 0.0)

    result = ips.run_ips_precheck("Buy NVDA.\n   * Total Investment: $9,000.00 USD\n", {"NVDA"})

    assert result["trades"][0]["size_usd"] is None
    assert result["violations"] == []
    assert _rows_by_check(result)["position cap"]["verdict"] == "NOT_EVALUATED"


def test_shares_size_converts_quote_currency_to_base(monkeypatch, tmp_path):
    """`shares × price` is in the security's currency, not the profile's.

    A shares-only draft states no currency at all, so nothing in the text marks
    the figure as foreign — the price does. Taken at face value against a CAD
    total, a US-listed buy understates every cap by the whole FX rate.
    """
    cad_ctx = {**copy.deepcopy(_CTX), "base_currency": "CAD"}
    _setup(monkeypatch, tmp_path, ctx=cad_ctx)
    monkeypatch.setattr(ips, "_get_quote_price", lambda s: (118.00, "USD"))
    monkeypatch.setattr(ips, "_get_fx_rate", lambda frm, to: 1.40 if (frm, to) == ("USD", "CAD") else 0.0)

    result = ips.run_ips_precheck("Buy 74 shares of ISRG.", {"ISRG"})

    assert result["trades"][0]["size_usd"] == pytest.approx(12_224.80)  # 74 × $118 USD × 1.40
    row = _rows_by_check(result)["position cap"]
    assert "74 shares ≈$12,225 CAD (at $118.00 USD)" in row["trade"]
    # The flip this guards: $8,732 read as CAD is covered by the $10k cash pile,
    # so it prices as a cash-funded 8.7% — a computed PASS. Converted, it is
    # new money against a larger denominator at 10.9%, over the 10% cap.
    assert row["verdict"] == "FAIL"


def test_shares_size_takes_currency_from_the_holding_without_a_quote(monkeypatch, tmp_path):
    """A held name already carries its currency — no quote round-trip needed."""
    cad_ctx = {**copy.deepcopy(_CTX), "base_currency": "CAD"}
    cad_ctx["holdings"][0]["currency"] = "USD"  # AAPL, held at $200.00 USD
    _setup(monkeypatch, tmp_path, ctx=cad_ctx)
    quote_calls: list[str] = []
    monkeypatch.setattr(ips, "_get_quote_price", lambda s: (quote_calls.append(s), (None, ""))[1])
    monkeypatch.setattr(ips, "_get_fx_rate", lambda frm, to: 1.40 if (frm, to) == ("USD", "CAD") else 0.0)

    result = ips.run_ips_precheck("Buy 40 shares of AAPL here.")

    assert quote_calls == []  # the holding answered it
    assert result["trades"][0]["size_usd"] == pytest.approx(11_200.0)  # 40 × $200 USD × 1.40
    row = _rows_by_check(result)["position cap"]
    assert "40 shares ≈$11,200 CAD (at $200.00 USD)" in row["trade"]
    assert row["verdict"] == "FAIL"  # 5k held + 11.2k new = 14.6% > 10%


def test_shares_size_abstains_when_price_currency_is_unknown(monkeypatch, tmp_path):
    """An unlabelled price against a non-USD base is not comparable — abstain.

    The quote feed can omit `currency`; guessing "USD" there would silently
    reintroduce the very error this converts away.
    """
    cad_ctx = {**copy.deepcopy(_CTX), "base_currency": "CAD"}
    _setup(monkeypatch, tmp_path, ctx=cad_ctx)
    monkeypatch.setattr(ips, "_get_quote_price", lambda s: (118.00, ""))

    result = ips.run_ips_precheck("Buy 74 shares of ISRG.", {"ISRG"})

    assert result["trades"][0]["size_usd"] is None
    assert result["violations"] == []
    assert _rows_by_check(result)["position cap"]["verdict"] == "NOT_EVALUATED"


def test_shares_size_unlabelled_price_still_evaluated_for_a_usd_profile(monkeypatch, tmp_path):
    """USD base is the one case where an unlabelled price needs no conversion.

    Abstaining here would strip sizing from the common path — and a
    NOT_EVALUATED sizing row reads downstream as a Rule 3 MAGNITUDE MISS.
    """
    _setup(monkeypatch, tmp_path)  # _CTX is USD-based, holdings carry no currency
    monkeypatch.setattr(ips, "_get_quote_price", lambda s: (118.00, ""))

    result = ips.run_ips_precheck("Buy 74 shares of ISRG.", {"ISRG"})

    assert result["trades"][0]["size_usd"] == pytest.approx(8_732.0)
    assert _rows_by_check(result)["position cap"]["verdict"] == "PASS"


def test_never_raises_on_context_error(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path)

    def boom():
        raise RuntimeError("portfolio unavailable")
    monkeypatch.setattr(ips, "_get_decision_context", boom)

    result = ips.run_ips_precheck("Buy $5,000 of NVDA.")

    _assert_said_nothing(result)


# ---------------------------------------------------------------------------
# Unstated caps. The old module carried house defaults (2%/10%/25%/30%) that no
# profile ever set, so the judge enforced them and cited them back to the user
# as "your 2% limit". A limit the user never wrote must enforce nothing.
# ---------------------------------------------------------------------------

def _setup_bare(monkeypatch, tmp_path, risk_constraints, ctx=None):
    """Like _setup but with NO cap defaults layered in — only what is passed."""
    calls = []
    monkeypatch.setattr(ips, "_get_decision_context", lambda: copy.deepcopy(ctx or _CTX))
    monkeypatch.setattr(ips, "_is_cash_symbol", lambda s: s == "CASH")
    monkeypatch.setattr(ips, "_get_quote_price", lambda s: (None, ""))
    monkeypatch.setattr(ips, "_get_allocation", _make_allocation_fake(_CURRENT_SECTORS, {}, calls))
    monkeypatch.setattr(ips, "_load_memory", lambda: {"risk_constraints": copy.deepcopy(risk_constraints)})
    return calls


@pytest.mark.parametrize("memory", [{}, {"risk_constraints": {}}, {"risk_constraints": None}])
def test_no_stated_caps_enforces_nothing_and_does_no_work(monkeypatch, tmp_path, memory):
    """No stated limit anywhere -> no rows, no violations, and no work done.

    An empty violation list matters as much as ever: _format_block announces
    itself as "the profile's IPS constraints" and tells the judge a
    NOT_EVALUATED row IS a Rule 3 MAGNITUDE MISS, so emitting a vacuous table
    would put the phantom limit straight back into the verdict. Nothing about
    the readiness note changes that — it computes no rows, applies no cap, and
    still bails before touching the portfolio.
    """
    calls = []
    monkeypatch.setattr(ips, "_get_decision_context", lambda: copy.deepcopy(_CTX))
    monkeypatch.setattr(ips, "_get_allocation", _make_allocation_fake(_CURRENT_SECTORS, {}, calls))
    monkeypatch.setattr(ips, "_load_memory", lambda: copy.deepcopy(memory))

    result = ips.run_ips_precheck("Buy $5,000 of NVDA here, entry at $200, stop-loss at $180.")

    assert result["trades"] == []
    assert result["rows"] == []
    assert result["violations"] == []
    assert "<ips_precheck>" not in result["block"]  # no table, vacuous or otherwise
    assert calls == []  # bailed before any portfolio/sector work


# ---------------------------------------------------------------------------
# Execution readiness — the second question an empty block could not answer
# ---------------------------------------------------------------------------

def test_an_unasked_profile_says_the_proposal_is_not_execution_ready(monkeypatch, tmp_path):
    """The defect this closes: a sized buy on a profile with no caps used to
    pass through in total silence, and silence reads as a clean check."""
    _setup_bare(monkeypatch, tmp_path, {})

    result = ips.run_ips_precheck("Buy $5,000 of NVDA here, entry at $200, stop-loss at $180.")

    assert result["execution_ready"] is False
    assert "NOT EXECUTION-READY" in result["block"]
    # A finding about the PROFILE, not a fault in the draft: routing it through
    # the violation gate would cap the score of advice that did nothing wrong.
    assert result["violations"] == []


def test_a_profile_that_chose_no_limits_is_left_alone(monkeypatch, tmp_path):
    """Confirmed unlimited is a finished profile. Nagging it would make the
    confirmation worth nothing, which is the whole reason it exists."""
    _setup_bare(monkeypatch, tmp_path, {
        "unconstrained_ack": {"acknowledged_at": "2026-07-28T09:00:00", "axes": list(ips._CONSTRAINT_KEYS)},
    })

    result = ips.run_ips_precheck("Buy $5,000 of NVDA here, entry at $200, stop-loss at $180.")

    assert result["execution_ready"] is True
    assert result["block"] == ""


def test_a_turn_that_proposes_nothing_stays_silent_even_when_unready(monkeypatch, tmp_path):
    """The readiness note rides on a proposal. A market-commentary turn has none,
    and putting the profile's state into every judge prompt would be noise."""
    calls = _setup_bare(monkeypatch, tmp_path, {})

    result = ips.run_ips_precheck("The market looks stretched and VIX is elevated. Keep holding.")

    assert result["execution_ready"] is False  # the profile state is still reported
    assert result["block"] == ""               # ...but nothing is said about it
    assert calls == []


def test_a_partly_capped_profile_still_flags_the_axes_nobody_answered(monkeypatch, tmp_path):
    """The most misleading case: a table of PASS rows on a profile that stated
    two of four axes reads as a clean bill of health for the whole trade."""
    _setup_bare(monkeypatch, tmp_path, {"max_position_pct": 10.0})

    result = ips.run_ips_precheck("Buy $5,000 of NVDA here, entry at $200, stop-loss at $180.")

    assert _rows_by_check(result)["position cap"]["verdict"] == "PASS"
    assert result["execution_ready"] is False
    assert "NOT EXECUTION-READY" in result["block"]


def test_the_readiness_note_forbids_the_judge_inventing_the_missing_limit(monkeypatch, tmp_path):
    """Rule 8 SOURCE FRAUD territory. The block names the axis, never a number,
    and says outright that it is not a breach and not to be scored."""
    _setup_bare(monkeypatch, tmp_path, {})

    block = ips.run_ips_precheck("Buy $5,000 of NVDA at $200, stop at $180.")["block"]

    assert "not a fault in the advice" in block
    assert "do not score it" in block
    assert "no figure exists" in block
    assert "%" not in block


def test_unreadable_memory_grants_no_caps(monkeypatch, tmp_path):
    """A broken profile read must not resurrect defaults — it means no limits."""
    def boom():
        raise OSError("profile unreadable")
    monkeypatch.setattr(ips, "_load_memory", boom)

    constraints = ips.load_ips_constraints()

    assert ips.stated_caps(constraints) == {}
    assert all(constraints[key] is None for key in ips._CONSTRAINT_KEYS)


def test_only_stated_caps_are_checked(monkeypatch, tmp_path):
    """A profile capping position size says nothing about risk or sector."""
    _setup_bare(monkeypatch, tmp_path, {"max_position_pct": 10.0})

    result = ips.run_ips_precheck("Buy $5,000 of NVDA here, entry at $200, stop-loss at $180.")

    checks = _rows_by_check(result)
    assert checks["position cap"]["verdict"] == "PASS"
    assert "dollar-at-risk" not in checks
    assert "sector cap" not in checks
    assert "2%" not in result["block"]


def test_unsized_trade_raises_no_magnitude_row_without_caps(monkeypatch, tmp_path):
    """The NOT_EVALUATED rows are themselves the Rule 3 trigger — suppress them."""
    _setup_bare(monkeypatch, tmp_path, {"restricted_symbols": ["TSLA"]})

    result = ips.run_ips_precheck("Start accumulating NVDA on weakness.")

    assert result["rows"] == [] or all(r["verdict"] == "INFO" for r in result["rows"])
    assert result["violations"] == []


def test_restricted_list_alone_still_enforced(monkeypatch, tmp_path):
    """A no-buy list is a stated rule even when no numeric cap is."""
    _setup_bare(monkeypatch, tmp_path, {"restricted_symbols": ["NVDA"]})

    result = ips.run_ips_precheck("Buy $5,000 of NVDA here, entry at $200, stop-loss at $180.")

    checks = _rows_by_check(result)
    assert checks["restricted list"]["verdict"] == "FAIL"
    assert "dollar-at-risk" not in checks  # still no invented risk limit


@pytest.mark.parametrize("bad", ["", "abc", None, 0, -5, [], {}])
def test_malformed_cap_reads_as_unstated_never_as_default(monkeypatch, bad):
    """Junk must not coerce into a number — least of all back into 2.0."""
    monkeypatch.setattr(ips, "_load_memory", lambda: {"risk_constraints": {"max_risk_per_trade_pct": bad}})

    constraints = ips.load_ips_constraints()

    assert constraints["max_risk_per_trade_pct"] is None
    assert "max_risk_per_trade_pct" not in ips.stated_caps(constraints)


def test_stated_caps_reports_only_what_user_set(monkeypatch):
    monkeypatch.setattr(ips, "_load_memory", lambda: {
        "risk_constraints": {"max_risk_per_trade_pct": 1.5, "max_sector_pct": "bogus"}
    })

    assert ips.stated_caps() == {"max_risk_per_trade_pct": 1.5}


# ---------------------------------------------------------------------------
# A percentage describing what the user ALREADY holds is not a proposed size.
# ---------------------------------------------------------------------------

# Heading-plus-bullets layout: the block window joins the heading to the
# bullets, so a percentage in the heading competes with the real size below.
_DESCRIPTIVE_PCT_DRAFT = """✅ **Position Maintained:** NVDA still earns its slot beside your 40% tech-heavy portfolio.
- **Portfolio Total:** $100,000.00 USD.
- **NVDA Trade Sizing:** Allocate Tranche 1 ($3,000 USD).
- **Execution:** Buy 3 shares at $200 USD.
- **Structural Stop:** $180 USD (anchored to 40-week MA support)."""


def test_current_weight_is_not_read_as_a_proposed_size():
    """A trailing "portfolio" used to qualify any nearby %, so a heading
    describing the book's CURRENT tech weight sized the trade at that weight
    (here 40% = $40,000) instead of the 3 shares the draft actually proposed.
    The judge then reported position-cap and max-risk breaches against the
    phantom figure and scored the revision below the draft it was fixing."""
    trades = ips.extract_proposed_trades(_DESCRIPTIVE_PCT_DRAFT, {"NVDA"}, 100_000.0, "USD")

    assert len(trades) == 1
    assert trades[0]["size_usd"] is None          # not 40_000.0
    assert trades[0]["size_label"] == "3 shares"  # the size actually proposed
    assert trades[0]["stop"] == 180.0


def test_no_phantom_cap_breach_from_a_descriptive_percentage(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path, ctx={
        "base_currency": "USD",
        "total_value_base": 100_000.0,
        "holdings": [{"symbol": "CASH", "account": "Margin", "value_base": 5_000.0, "current_price": 1.0}],
    })
    monkeypatch.setattr(ips, "_get_quote_price", lambda s: (200.0, "USD"))

    result = ips.run_ips_precheck(_DESCRIPTIVE_PCT_DRAFT)

    assert result["violations"] == []
    checks = _rows_by_check(result)
    assert checks["position cap"]["verdict"] == "PASS"
    assert "40,000" not in result["block"]
    assert "40%" not in result["block"]


@pytest.mark.parametrize("draft,expected", [
    ("Allocate 5% of your portfolio to NVDA.", "5% of portfolio (≈$5,000)"),
    ("Start a 3% position in NVDA.", "3% of portfolio (≈$3,000)"),
    ("Allocate 4% to NVDA here.", "4% of portfolio (≈$4,000)"),
    ("Buy 2% of the book in NVDA.", "2% of portfolio (≈$2,000)"),
])
def test_real_percentage_sizes_still_parse(draft, expected):
    """The guard must not cost the module its legitimate percent sizing."""
    trades = ips.extract_proposed_trades(draft, {"NVDA"}, 100_000.0, "USD")

    assert trades[0]["size_label"] == expected


@pytest.mark.parametrize("draft", [
    "NVDA adds to your already 30% technology exposure in the portfolio.",
    "Buy NVDA — it diversifies your 45% US-heavy portfolio.",
    "Adding NVDA against a portfolio that is 40% tech is a concentration risk.",
])
def test_descriptive_percentages_never_become_sizes(draft):
    trades = ips.extract_proposed_trades(draft, {"NVDA"}, 100_000.0, "USD")

    assert all(t["size_usd"] is None for t in trades)


# --- Verb-as-label sizing + benchmark mentions -----------------------------
# Both regressions were observed together on one live market_dip draft: the
# judge scored a fully-sized revision 3/10 for a "Magnitude Miss", reporting
# that per-ticker sizing was "hidden", when every figure was in plain sight.

_DIP_TRANCHE_DRAFT = """Macro Risk: 3.5% CPI and 4.2% unemployment justify caution. We are deploying only a fractional Tranche 1 ($3,400 CAD) and holding $5,100 CAD cash for a deeper >5% SPY correction.
QRS (Not Held): Current: $0 CAD (0%). Buy: $1,500 CAD (0.30% of $500,000.00 CAD portfolio). Stop: $30.00. Risk: $90 CAD (0.02%).
DEF.TO (Not Held): Current: $0 CAD (0%). Buy: $1,500 CAD (0.30%). Stop: $6.00 CAD. Risk: $90 CAD (0.02%).
LMN (Held): Current: $3,500.00 CAD (0.70%). Add: $1,500 CAD (0.30%). Stop: $12.00. Risk: $120 CAD (0.02%)."""


@pytest.mark.parametrize("verb", ["Buy", "Add", "Deploy", "Allocate", "Invest"])
def test_size_verb_used_as_a_label_parses(verb):
    """"Buy: $1,500" — the verb as a LABEL, separated from the figure by a colon.

    _SIZE_VERB_BEFORE_RE required strict adjacency, so the colon broke the cue;
    the figure matched neither the verb form nor _SIZE_LABEL_BEFORE_RE (a noun
    allowlist), and a fully sized trade read as "no size stated".
    """
    trades = ips.extract_proposed_trades(
        f"NVDA (Not Held): {verb}: $1,500 USD. Stop: $30.00.", {"NVDA"}, 100_000.0, "USD"
    )

    assert trades[0]["size_usd"] == 1500.0
    assert trades[0]["stop"] == 30.00


@pytest.mark.parametrize("draft", [
    "Buy NVDA at $200.",
    "Add NVDA near $200.",
    "Buy NVDA — entry $200.",
])
def test_verb_label_widening_still_rejects_prices(draft):
    """The colon is optional, but adjacency is not: a stated ENTRY stays a price."""
    trades = ips.extract_proposed_trades(draft, {"NVDA"}, 100_000.0, "USD")

    assert all(t["size_usd"] is None for t in trades)


def test_every_ticker_in_a_sizing_block_reports_its_size():
    """The live draft that scored 3/10 for a Magnitude Miss it did not commit."""
    trades = {
        t["ticker"]: t
        for t in ips.extract_proposed_trades(
            _DIP_TRANCHE_DRAFT, {"QRS", "DEF.TO", "LMN", "SPY"}, 500_000.00, "CAD"
        )
    }

    assert {"QRS", "DEF.TO", "LMN"} <= set(trades)
    assert all(trades[t]["size_usd"] == 1500.0 for t in ("QRS", "DEF.TO", "LMN"))
    assert trades["QRS"]["stop"] == 30.00
    assert trades["LMN"]["stop"] == 12.00


@pytest.mark.parametrize("draft", [
    "We are deploying Tranche 1 ($3,400 CAD) if we get a deeper >5% SPY correction.",
    "Add to the position on any SPY drawdown past -5%.",
    "Only High Conviction beat SPY, at +4.3pp of sector alpha.",
    "Deploy the second tranche if SPY drops another 5%.",
])
def test_benchmark_mentions_are_not_proposed_trades(draft):
    """An index proxy used as a market yardstick shares a block with real buy
    verbs constantly in a dip plan; it was extracted as a sizeless phantom
    trade, which then dragged the sizing audit down."""
    trades = ips.extract_proposed_trades(draft, None, 100_000.0, "USD")

    assert "SPY" not in {t["ticker"] for t in trades}


def test_a_genuine_benchmark_ticker_buy_still_extracts():
    """The guard is per-MENTION: naming SPY as a yardstick must not immunise it
    against a real buy stated elsewhere in the same draft."""
    draft = (
        "Hold cash through a deeper >5% SPY correction.\n"
        "SPY (Not Held): Buy: $5,000 USD. Stop: $580.00."
    )
    trades = {t["ticker"]: t for t in ips.extract_proposed_trades(draft, {"SPY"}, 100_000.0, "USD")}

    assert trades["SPY"]["size_usd"] == 5000.0
    assert trades["SPY"]["stop"] == 580.0


@pytest.mark.parametrize("tag", ["(Not Held)", "(not currently held)", "(No Position)", "(Held)"])
def test_position_status_tag_is_not_a_buy_negation(tag):
    """"QRS (Not Held): Buy: $1,500" — the status tag the advisor writes on every
    new-name sizing line. The bare \\bnot\\b negation cue read it as "do not buy"
    and dropped the trade from the audit altogether, which is worse than a size
    miss: an unaudited trade is indistinguishable from no trade proposed."""
    trades = ips.extract_proposed_trades(
        f"NVDA {tag}: Buy: $1,500 USD. Stop: $30.00.", {"NVDA"}, 100_000.0, "USD"
    )

    assert len(trades) == 1
    assert trades[0]["size_usd"] == 1500.0


@pytest.mark.parametrize("draft", [
    "Do not buy NVDA here.",
    "Avoid adding NVDA at these levels.",
    "Excluded: NVDA — buy thesis is broken.",
    "NVDA (Not Held): we are not buying it here.",
])
def test_real_buy_negations_still_suppress_the_trade(draft):
    """Blanking the status tag must not cost the negation guard its actual job."""
    trades = ips.extract_proposed_trades(draft, {"NVDA"}, 100_000.0, "USD")

    assert trades == []
