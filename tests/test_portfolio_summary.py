import pytest

from tools import portfolio_csv


def test_compute_portfolio_summary_keeps_values_when_sync_errors_exist(monkeypatch):
    sample_holdings = [
        {
            "symbol": "VOO",
            "shares": 10.0,
            "purchase_price": 20.0,
            "current_price": 25.0,
            "market_value": 250.0,
            "account": "Manual Brokerage",
            "currency": "USD",
            "source": "Manual",
            "is_private_asset": False,
        },
        {
            "symbol": "PSA.TO",
            "shares": 50.0,
            "purchase_price": 10.0,
            "current_price": 10.0,
            "market_value": 500.0,
            "account": "Manual Brokerage",
            "currency": "CAD",
            "source": "Manual",
            "is_private_asset": False,
        },
        {"_sync_errors": ["Questrade Global Error: token refresh failed"]},
    ]

    def fake_get_cached(key, ttl_seconds=None):
        if key == "usd_cad_rate":
            return 1.4
        return None

    monkeypatch.setattr(portfolio_csv, "load_portfolio", lambda: sample_holdings)
    monkeypatch.setattr(portfolio_csv, "get_cached", fake_get_cached)
    monkeypatch.setattr(portfolio_csv, "set_cached", lambda *args, **kwargs: None)

    summary = portfolio_csv._compute_portfolio_summary()

    assert [item["symbol"] for item in summary["holdings"]] == ["VOO", "PSA.TO"]
    assert summary["total_value_usd"] == pytest.approx(607.1428571428571)
    assert summary["total_value_cad"] == pytest.approx(850.0)
    assert summary["total_invested_usd"] == pytest.approx(557.1428571428571)
    assert summary["total_gain_loss_cad"] == pytest.approx(70.0)
    assert "Questrade Global Error" in summary["sync_errors"][0]
    assert "last_sync_time" in summary


def test_load_portfolio_replaces_synced_questrade_csv_rows(monkeypatch, tmp_path):
    csv_path = tmp_path / "portfolio.csv"
    csv_path.write_text(
        "\n".join(
            [
                "symbol,shares,purchase_price,account,currency",
                "AAPL,10,150.0,TFSA Questrade,USD",
                "MSFT,5,250.0,Manual Brokerage,USD",
            ]
        )
    )

    class LiveQuestradeAPI:
        def get_all_holdings(self):
            return {
                "holdings": [
                    {
                        "symbol": "AAPL",
                        "shares": 7.0,
                        "purchase_price": 140.0,
                        "current_price": 200.0,
                        "market_value": 1400.0,
                        "account": "TFSA Questrade",
                        "currency": "USD",
                    }
                ],
                "errors": [],
            }

    monkeypatch.setattr("tools.questrade.QuestradeAPI", LiveQuestradeAPI)

    holdings = portfolio_csv.load_portfolio(str(csv_path))

    aapl_rows = [item for item in holdings if item.get("symbol") == "AAPL"]
    assert len(aapl_rows) == 1
    assert aapl_rows[0]["shares"] == 7.0
    assert aapl_rows[0]["source"] == "API"
    assert any(item.get("symbol") == "MSFT" for item in holdings)


def test_demo_mode_loads_profile_scoped_my_portfolio(monkeypatch, tmp_path):
    base_dir = tmp_path
    profile_dir = base_dir / "user_data" / "profiles" / "demo"
    profile_dir.mkdir(parents=True)
    csv_path = profile_dir / "my_portfolio.csv"
    csv_path.write_text(
        "Symbol,Shares,Purchase Price,Account,Currency\n"
        "DEMO,3,10,Demo Brokerage,USD\n"
    )

    monkeypatch.setattr("tools.user_profile.os.path.dirname", lambda x: str(base_dir))
    monkeypatch.setenv("DEMO_MODE", "true")

    holdings = portfolio_csv.load_portfolio()

    assert holdings == [
        {
            "symbol": "DEMO",
            "shares": 3.0,
            "purchase_price": 10.0,
            "account": "Demo Brokerage",
            "currency": "USD",
            "return_pct": None,
            "source": "Manual",
            "is_private_asset": False,
        }
    ]


def test_demo_seed_portfolio_is_around_250k_and_uses_seeded_prices(monkeypatch, tmp_path):
    from tools.user_profile import ensure_demo_profile

    base_dir = tmp_path
    monkeypatch.setattr("tools.user_profile.os.path.dirname", lambda x: str(base_dir))
    monkeypatch.setenv("DEMO_MODE", "true")
    monkeypatch.setattr(portfolio_csv, "get_cached", lambda key, ttl_seconds=None: 1.44 if key == "usd_cad_rate" else None)
    monkeypatch.setattr(portfolio_csv, "set_cached", lambda *args, **kwargs: None)

    ensure_demo_profile(reset=True)
    holdings = portfolio_csv.load_portfolio()

    assert all(
        item.get("current_price") is not None
        for item in holdings
        if item.get("symbol") not in {"PENSION-DEMO"}
    )

    summary = portfolio_csv._compute_portfolio_summary()

    # USD leg 74,175 @ 1.44 = 106,812, CAD equities 78,200, and the pension at
    # 59,000 units x 1.055 = 62,245. The previous 246,962.0 expectation valued the
    # pension at 61,950: totals were rebuilt from the 2dp display string, so its
    # 1.055 unit price truncated to 1.05 and lost 295.00.
    assert summary["total_value_cad"] == pytest.approx(247257.0)
    assert 240000 <= summary["total_value_cad"] <= 260000
    assert len(summary["holdings"]) == 11


def test_unit_priced_pension_applies_manual_return_without_private_asset_type(monkeypatch):
    """A pension row lacking Asset Type=Private must still honour its return_pct.

    Legacy CSVs (and rows added through the portfolio editor, which defaults to
    Public) carry no Asset Type, so the row was priced as a synthetic unit asset
    but returned as a $1.00 equity — reporting a flat 0% and dropping the return.
    """
    monkeypatch.setattr(
        portfolio_csv,
        "load_portfolio",
        lambda: [
            {
                "symbol": "PENSION-DEMO",
                "shares": 59000.0,
                "purchase_price": 1.0,
                "account": "Demo Pension",
                "currency": "CAD",
                "return_pct": 5.5,
                "source": "Manual",
                "is_private_asset": False,
            }
        ],
    )
    monkeypatch.setattr(portfolio_csv, "get_cached", lambda key, ttl_seconds=None: None)
    monkeypatch.setattr(portfolio_csv, "set_cached", lambda *args, **kwargs: None)
    monkeypatch.setattr(portfolio_csv, "get_profile_base_currency", lambda: "CAD")
    monkeypatch.setattr(portfolio_csv, "get_exchange_rate", lambda f, t: 1.0)

    holding = portfolio_csv._compute_portfolio_summary()["holdings"][0]

    assert holding["gain_loss"] == "+5.5%"
    assert holding["gain_loss_pct"] == pytest.approx(5.5)
    assert holding["is_cash_or_pension"] is True
    # Full precision: not the 61,950 the rounded 1.05 display price would give.
    assert holding["value_base"] == pytest.approx(62245.0)


def _stated_total_row(**overrides):
    """A group-pension row as its holder can actually describe it.

    A group-pension statement gives units, a personal rate of return and a total. It does
    not give a unit cost, so the row carries no entry price and the engine has to
    work backwards from the total rather than assume a $1.00 unit.
    """
    row = {
        "symbol": "GRPPEN",
        "shares": 1240.5678,
        "purchase_price": 0.0,
        "account": "Workplace Pension",
        "currency": "CAD",
        "return_pct": 8.4,
        "market_value": 45678.90,
        "source": "Manual",
        "is_private_asset": False,
    }
    row.update(overrides)
    return row


def _summary_for(monkeypatch, row):
    monkeypatch.setattr(portfolio_csv, "load_portfolio", lambda: [row])
    monkeypatch.setattr(portfolio_csv, "get_cached", lambda key, ttl_seconds=None: None)
    monkeypatch.setattr(portfolio_csv, "set_cached", lambda *a, **k: None)
    monkeypatch.setattr(portfolio_csv, "get_profile_base_currency", lambda: "CAD")
    monkeypatch.setattr(portfolio_csv, "get_exchange_rate", lambda f, t: 1.0)
    return portfolio_csv._compute_portfolio_summary()["holdings"][0]


def test_stated_total_is_the_mark_and_cost_basis_is_derived_from_it(monkeypatch):
    """Units + return + total, no entry price — the whole of what a statement gives.

    Before this, the only way to enter a pension was a $1.00 synthetic unit, which
    valued 1,240 units at $1,240 and contradicted the statement in the holder's hand.
    """
    holding = _summary_for(monkeypatch, _stated_total_row())

    # The typed total survives to the penny — not units x a 2dp-rounded unit price.
    assert holding["value_base"] == pytest.approx(45678.90)
    assert holding["gain_loss_pct"] == pytest.approx(8.4)
    assert holding["is_cash_or_pension"] is True
    # Entry price is worked backwards out of the total and the return, so the
    # position's cost basis is right without the holder ever knowing a unit cost.
    cost_basis = holding["purchase_price_raw"] * holding["shares"]
    assert cost_basis == pytest.approx(45678.90 / 1.084)
    assert holding["current_price_raw"] == pytest.approx(45678.90 / 1240.5678)


def test_stated_total_holds_without_the_private_asset_flag(monkeypatch):
    """The editor defaults to Public, so the fix cannot depend on Asset Type."""
    public = _summary_for(monkeypatch, _stated_total_row(is_private_asset=False))
    private = _summary_for(monkeypatch, _stated_total_row(is_private_asset=True))
    assert public["value_base"] == pytest.approx(private["value_base"])
    assert public["gain_loss_pct"] == pytest.approx(private["gain_loss_pct"])


def test_negative_return_derives_a_cost_basis_above_the_total(monkeypatch):
    holding = _summary_for(monkeypatch, _stated_total_row(return_pct=-6.5, market_value=41000.0))

    assert holding["value_base"] == pytest.approx(41000.0)
    assert holding["gain_loss_pct"] == pytest.approx(-6.5)
    # A losing year means it cost MORE than it is worth; a cost basis at or below
    # the total would report the loss as a gain.
    assert holding["purchase_price_raw"] * holding["shares"] == pytest.approx(41000.0 / 0.935)


def test_stated_total_without_a_return_computes_it_rather_than_reporting_flat(monkeypatch):
    """A typed entry price plus a typed total is enough. 0% would be a fabricated flat."""
    holding = _summary_for(monkeypatch, _stated_total_row(purchase_price=30.0, return_pct=None))

    assert holding["value_base"] == pytest.approx(45678.90)
    assert holding["gain_loss_pct"] == pytest.approx(
        (45678.90 - 30.0 * 1240.5678) / (30.0 * 1240.5678) * 100
    )


def test_a_brokers_market_value_is_not_read_as_a_stated_total(monkeypatch):
    """market_value carries two meanings; only the manual one is a statement total.

    Every broker sets market_value as a "don't re-fetch" marker on a row that has a
    genuine quote. Reading that as hand-typed re-derived the broker's price and
    classified a live equity as a pension.
    """
    holding = _summary_for(monkeypatch, {
        "symbol": "AAPL",
        "shares": 10.0,
        "purchase_price": 150.0,
        "current_price": 110.0,
        "market_value": 1100.0,
        "account": "Questrade TFSA",
        "currency": "USD",
        "source": "API",
        "is_private_asset": False,
    })

    assert holding["is_cash_or_pension"] is False
    assert holding["current_price_raw"] == pytest.approx(110.0)
    assert holding["stated_total"] is None


def _underspecified_row(**overrides):
    """The row the stated-total path cannot save: units and a return, nothing else.

    Its holder supplied neither of the two things that could value it — no entry
    price to multiply the units by, and no statement total to work backwards from —
    so it is genuinely underspecified rather than merely awkward to price.
    """
    row = {
        "symbol": "GRPPEN",
        "shares": 1240.0,
        "purchase_price": 0.0,
        "account": "Workplace Pension",
        "currency": "CAD",
        "return_pct": 8.4,
        "source": "Manual",
        "is_private_asset": False,
    }
    row.update(overrides)
    return row


def _summary_of(monkeypatch, rows, quote=None):
    """Full summary for a list of rows, with the quote lookup pinned.

    Defaults to a failed lookup, which is the honest default for these rows: a group
    pension has no ticker for anyone to quote. Pinning it also keeps the test off the
    network, where it would otherwise be measuring the network.
    """
    import tools.market_data as market_data

    monkeypatch.setattr(portfolio_csv, "load_portfolio", lambda: list(rows))
    monkeypatch.setattr(portfolio_csv, "get_cached", lambda key, ttl_seconds=None: None)
    monkeypatch.setattr(portfolio_csv, "set_cached", lambda *a, **k: None)
    monkeypatch.setattr(portfolio_csv, "get_profile_base_currency", lambda: "CAD")
    monkeypatch.setattr(portfolio_csv, "get_exchange_rate", lambda f, t: 1.0)
    monkeypatch.setattr(
        market_data, "get_stock_data",
        lambda sym, *a, **k: dict(quote) if quote else {"current_price": "N/A"},
    )
    return portfolio_csv._compute_portfolio_summary()


def test_units_with_no_entry_price_and_no_total_are_unvalued_not_zero(monkeypatch):
    """The row that reported $0.00 and a confident +0.0%.

    Both halves were fabrications, and they compounded: the zero dropped a real
    position out of every allocation and risk figure, and the flat return dressed the
    missing entry price up as a year of no movement.
    """
    holding = _summary_of(monkeypatch, [_underspecified_row()])["holdings"][0]

    assert holding["is_unvalued"] is True
    # None, not 0.0. A zero is a claim that the position is worth nothing, which is a
    # different statement from "we have no basis to say what it is worth".
    assert holding["value_base"] is None
    assert holding["value_native"] is None
    assert holding["value_usd"] is None
    assert holding["value_cad"] is None
    # Nor a flat return. The typed 8.4% is not echoed here either: a percentage of an
    # unknown amount is still unknown.
    assert holding["gain_loss"] == "—"
    assert holding["gain_loss_pct"] is None
    assert holding["purchase_price"] == "—"
    assert holding["current_price"] == "—"
    assert "Unvalued" in holding["status"]


def test_an_unvalued_row_is_excluded_from_the_totals_rather_than_counted_as_zero(monkeypatch):
    """A zero sums silently; the whole point is that this position does not."""
    summary = _summary_of(monkeypatch, [
        _underspecified_row(),
        _stated_total_row(symbol="PENSION-OK", shares=1000.0, market_value=20000.0,
                          return_pct=5.0),
    ])

    # The total is the sum of what could be valued, and says so — not 45,678.90 + 0.
    assert summary["total_value_base"] == pytest.approx(20000.0)
    assert summary["total_value_cad"] == pytest.approx(20000.0)
    assert summary["summary"]["number_of_positions"] == 2

    # The position is still listed. Excluded from the arithmetic is not the same as
    # hidden — a holder who cannot see the row cannot fix it.
    assert [h["symbol"] for h in summary["holdings"]] == ["GRPPEN", "PENSION-OK"]


def test_the_excluded_position_is_named_in_a_notice(monkeypatch):
    """"1 holding excluded" says something is wrong without saying which row."""
    summary = _summary_of(monkeypatch, [_underspecified_row()])

    assert summary["unvalued_holdings"] == [{
        "symbol": "GRPPEN",
        "account": "Workplace Pension",
        "shares": 1240.0,
        "reason": "no entry price and no stated total value",
    }]
    notice = summary["unvalued_notice"]
    assert "GRPPEN" in notice and "Workplace Pension" in notice
    # Both repairs are offered, because either one alone is enough.
    assert "entry price" in notice and "total value" in notice

    # Kept off integration_notices, which means one specific other thing — a broker
    # nobody asked. Consumers label that channel "not synced (never asked)", which
    # would misdescribe a holding that simply cannot be priced.
    assert summary["integration_notices"] == []


def test_a_complete_portfolio_reports_no_unvalued_holdings(monkeypatch):
    """The notice must stay empty when there is nothing to notice."""
    summary = _summary_of(monkeypatch, [_stated_total_row()])

    assert summary["unvalued_holdings"] == []
    assert summary["unvalued_notice"] == ""
    assert summary["holdings"][0]["is_unvalued"] is False
    # The sibling stated-total path is untouched by any of this.
    assert summary["holdings"][0]["value_base"] == pytest.approx(45678.90)


def test_a_stated_total_rescues_the_row(monkeypatch):
    """One of the two repairs the notice offers. The total is the mark directly."""
    holding = _summary_of(monkeypatch, [_underspecified_row(market_value=45678.90)])["holdings"][0]

    assert holding["is_unvalued"] is False
    assert holding["value_base"] == pytest.approx(45678.90)
    assert holding["gain_loss_pct"] == pytest.approx(8.4)


def test_an_entry_price_rescues_the_row_by_valuing_it_at_cost(monkeypatch):
    """The other repair. With no quote, units × entry price is the only mark there is.

    The typed 8.4% is NOT applied on this path — the row reports +0.0% and is valued
    at cost. That is a separate defect of the same family (a manual row with an entry
    price, a typed return and no quote still prints a flat return), deliberately left
    alone here: it has a real basis to value against, so it is not the unvalued case,
    and changing it would move the totals of rows nobody reported a problem with.
    This test pins the current behaviour so that change is a visible one when it comes.
    """
    holding = _summary_of(monkeypatch, [_underspecified_row(purchase_price=30.0)])["holdings"][0]

    assert holding["is_unvalued"] is False
    assert holding["value_base"] == pytest.approx(1240.0 * 30.0)
    assert holding["gain_loss_pct"] == pytest.approx(0.0)


def test_a_live_quote_still_reverse_engineers_the_entry_price(monkeypatch):
    """The existing fallback must keep working — unvalued is the last resort, not the first.

    A missing entry price is only fatal when nothing else can price the row. Where a
    quote exists, the cost basis is still worked backwards out of it and the typed
    return, exactly as before.
    """
    holding = _summary_of(
        monkeypatch, [_underspecified_row(symbol="AAPL")],
        quote={"current_price": "$120.00"},
    )["holdings"][0]

    assert holding["is_unvalued"] is False
    assert holding["current_price_raw"] == pytest.approx(120.0)
    assert holding["purchase_price_raw"] == pytest.approx(120.0 / 1.084)
    assert holding["gain_loss_pct"] == pytest.approx(8.4)


@pytest.mark.parametrize("row,label", [
    (_underspecified_row(purchase_price=1.0), "unit-priced pension"),
    (_underspecified_row(is_private_asset=True), "private asset"),
    (_underspecified_row(symbol="CASH", purchase_price=0.0), "cash balance"),
    (_underspecified_row(shares=0.0), "no units to value"),
])
def test_rows_that_have_a_basis_are_not_swept_up_as_unvalued(monkeypatch, row, label):
    """Every synthetic-unit row already had an answer; none of them may change.

    Over-triggering here would be worse than the original bug: it would empty the
    totals of pensions and cash that were being valued correctly.
    """
    holding = _summary_of(monkeypatch, [row])["holdings"][0]

    assert holding["is_unvalued"] is False, label
    assert holding["value_base"] is not None, label


def test_winners_and_losers_survive_a_row_with_no_return_to_rank(monkeypatch):
    """gain_loss is "—" on an unvalued row, and the ranking parses it as a float.

    Leaving it in the tradable list raised ValueError and took the whole summary
    down — a portfolio that would not load at all.
    """
    summary = _summary_of(monkeypatch, [
        _underspecified_row(),
        {"symbol": "AAPL", "shares": 10.0, "purchase_price": 100.0,
         "current_price": 150.0, "market_value": 1500.0, "account": "TFSA",
         "currency": "CAD", "source": "API", "is_private_asset": False},
    ])

    assert summary["top_winners"] == ["AAPL: +50.0%"]
    assert summary["top_losers"] == []
    assert "GRPPEN" not in " ".join(summary["top_winners"] + summary["top_losers"])


def test_the_account_is_still_listed_but_its_total_omits_the_unvalued_row(monkeypatch):
    """The custody node is real even when one position in it cannot be priced."""
    summary = _summary_of(monkeypatch, [
        _underspecified_row(),
        _stated_total_row(symbol="PENSION-OK", shares=1000.0, market_value=20000.0,
                          return_pct=5.0, account="Workplace Pension"),
    ])

    accounts = {a["account"]: a for a in summary["accounts"]}
    assert "Workplace Pension" in accounts
    assert accounts["Workplace Pension"]["total_value"] == "$20,000.00 CAD"
    # A zero would have dragged this return toward 0% as though the pension had a
    # flat sleeve in it.
    assert accounts["Workplace Pension"]["return"] == "+5.0%"


def test_an_account_holding_only_unvalued_rows_reports_zero_without_raising(monkeypatch):
    """The degenerate case: nothing in the account can be priced.

    Every account-level figure is a sum over value_base, which is None here, so this
    is where an unguarded aggregate raises.
    """
    summary = _summary_of(monkeypatch, [_underspecified_row()])

    accounts = {a["account"]: a for a in summary["accounts"]}
    assert accounts["Workplace Pension"]["total_value"] == "$0.00 CAD"
    assert summary["total_value_base"] == pytest.approx(0.0)
    # A pension account: the liquidity split buckets it, and that bucket sums too.
    assert summary["liquidity"]["locked_pension_value"] == "$0.00 CAD"


def test_the_decision_context_carries_the_absence_rather_than_coercing_it_to_zero(monkeypatch):
    """This is what reaches the agents, and _coerce_number turns None into 0.0.

    Left alone, the fix would hold everywhere except the one payload an advisor
    actually reads — where the position would reappear as a verified $0.00.
    """
    summary = _summary_of(monkeypatch, [_underspecified_row()])
    monkeypatch.setattr(portfolio_csv, "get_portfolio_summary", lambda force=False: summary)

    ctx = portfolio_csv.get_portfolio_decision_context()
    holding = ctx["holdings"][0]

    assert holding["is_unvalued"] is True
    assert holding["value_base"] is None
    assert holding["value_cad"] is None
    assert holding["value_usd"] is None
    assert holding["allocation_pct"] is None
    # The symbol is still owned — "we cannot value it" is not "you do not hold it".
    assert ctx["owned_symbols"] == ["GRPPEN"]
    assert "GRPPEN" in ctx["unvalued_notice"]


def test_get_tradeable_symbols_filters_cash_pensions_and_metadata(monkeypatch):
    monkeypatch.setattr(
        portfolio_csv,
        "load_portfolio",
        lambda: [
            {"symbol": "AAPL", "purchase_price": 150.0},
            {"symbol": "CASH", "purchase_price": 1.0},
            {"symbol": "PRIVATE_FUND_1", "purchase_price": 20.0, "is_private_asset": True},
            {"symbol": "PRIVATE_FUND_2", "purchase_price": 20.0, "is_private_asset": True},
            {"symbol": "VTI.TO", "purchase_price": 75.0},
            {"_sync_errors": ["ignored"]},
        ],
    )

    assert sorted(portfolio_csv.get_tradeable_symbols()) == ["AAPL", "VTI.TO"]


def test_portfolio_decision_context_includes_allocations_and_verification(monkeypatch):
    monkeypatch.setattr(
        portfolio_csv,
        "get_portfolio_summary",
        lambda force=False: {
            "last_sync_time": "2026-05-14T10:00:00",
            "total_value_cad": 10000.0,
            "total_value_usd": 7000.0,
            "sync_errors": [],
            "summary": {"current_value": "$10,000.00 CAD"},
            "holdings": [
                {
                    "symbol": "T",
                    "account": "IRA",
                    "source": "Manual",
                    "shares": 100,
                    "current_price": "$5.00",
                    "purchase_price": "$4.00",
                    "gain_loss": "+25.0%",
                    "currency": "USD",
                    "value_cad": 720.0,
                    "value_usd": 500.0,
                },
                {
                    "symbol": "CASH",
                    "account": "IRA",
                    "source": "Manual",
                    "shares": 9280,
                    "current_price": "$1.00",
                    "purchase_price": "$1.00",
                    "gain_loss": "+0.0%",
                    "currency": "CAD",
                    "value_cad": 9280.0,
                    "value_usd": 6500.0,
                    "is_cash_or_pension": True,
                },
            ],
        },
    )

    context = portfolio_csv.get_portfolio_decision_context(symbols=["T", "DAL"])

    assert context["owned_symbols"] == ["CASH", "T"]
    assert context["holdings"][0]["allocation_pct"] == pytest.approx(7.2)
    assert context["requested_symbols"][0]["owned"] is True
    assert context["requested_symbols"][1]["symbol"] == "DAL"
    assert context["requested_symbols"][1]["owned"] is False
