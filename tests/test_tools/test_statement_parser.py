"""Pasted-statement extraction — the guards, not the model.

This module lets a language model read a financial document and propose rows for
the user's own ledger, which makes it the highest-consequence extraction in the
app: a fabricated position is indistinguishable from a real one once it is in the
table, and every allocation, risk and return figure downstream reads that table.

The tests below pin the three guards that make that acceptable — citation or
discard, numbers echoed rather than computed, and cost basis kept apart from
market price — plus the boundary the whole design rests on: this module drafts,
it never writes. Nothing here reaches a model or a file.
"""
import json
import types

import pytest

import tools.statement_parser as sp


@pytest.fixture
def llm(monkeypatch):
    """A stand-in model. Set `llm.reply` to the JSON the parse should receive."""
    import agent.utils as utils

    holder = types.SimpleNamespace(reply='{"holdings": []}', calls=0, raises=None)

    def _safe_invoke(_client, _messages, **_kwargs):
        holder.calls += 1
        if holder.raises:
            raise holder.raises
        return types.SimpleNamespace(content=holder.reply)

    monkeypatch.setattr(utils, "llm_ready", lambda: (True, ""))
    monkeypatch.setattr(utils, "get_fast_llm", lambda *a, **k: object())
    monkeypatch.setattr(utils, "safe_invoke", _safe_invoke)
    return holder


def _reply(*holdings):
    return json.dumps({"holdings": list(holdings)})


STATEMENT = (
    "Holdings as at 30 June 2026\n"
    "AAPL   Apple Inc            30       170.25     5,850.00\n"
    "TD.TO  Toronto-Dominion     100      85.40      9,120.00\n"
    "Total portfolio value                           14,970.00\n"
)


# ---------------------------------------------------------------------------
# The gates — and the difference between "found nothing" and "could not look"
# ---------------------------------------------------------------------------

def test_empty_input_never_reaches_the_model(llm):
    report = sp.parse_statement_text("   ")

    assert report["ok"] is False
    assert report["rows"] == []
    assert "paste" in report["reason"].lower()
    assert llm.calls == 0


def test_oversized_input_is_refused_whole_rather_than_truncated(llm):
    report = sp.parse_statement_text("AAPL 30 170.25\n" * 5000)

    assert report["ok"] is False
    assert llm.calls == 0
    # The user must be able to tell that nothing was read, because a silent
    # truncation drops the positions at the bottom of the statement invisibly.
    assert "nothing was read" in report["reason"].lower()


def test_an_unconfigured_provider_says_so_instead_of_finding_nothing(monkeypatch, llm):
    import agent.utils as utils
    monkeypatch.setattr(utils, "llm_ready", lambda: (False, "LLM_PROVIDER=vertexai but key not set"))

    report = sp.parse_statement_text(STATEMENT)

    assert report["ok"] is False
    assert llm.calls == 0
    assert "no language model is configured" in report["reason"].lower()
    assert "key not set" in report["reason"]


def test_a_model_that_cannot_be_built_is_reported_not_swallowed(llm):
    # llm_ready() clearing is not the same as a client that builds.
    llm.raises = RuntimeError("no model id for role=fast")

    report = sp.parse_statement_text(STATEMENT)

    assert report["ok"] is False
    assert "could not be reached" in report["reason"]
    assert "no model id" in report["reason"]


def test_no_holdings_found_reads_differently_from_everything_discarded(llm):
    llm.reply = _reply()
    empty = sp.parse_statement_text(STATEMENT)

    llm.reply = _reply({"symbol": "NVDA", "shares": 5, "source_text": "NVDA 5 400.00"})
    discarded = sp.parse_statement_text(STATEMENT)

    assert empty["ok"] and discarded["ok"]
    assert empty["rows"] == [] and discarded["rows"] == []
    # Both are empty; they are not the same claim about the user's text.
    assert "no positions were found" in empty["reason"]
    assert "nothing could be verified" in discarded["reason"].lower()
    assert discarded["dropped"], "a discard must name what it discarded"


# ---------------------------------------------------------------------------
# Guard one: citation or discard
# ---------------------------------------------------------------------------

def test_a_verified_row_survives_with_its_quote(llm):
    llm.reply = _reply({
        "symbol": "AAPL", "name": "Apple Inc", "shares": 30, "purchase_price": 170.25,
        "account": "RRSP", "currency": "USD",
        "source_text": "AAPL   Apple Inc            30       170.25     5,850.00",
    })

    report = sp.parse_statement_text(STATEMENT)

    assert report["ok"] is True
    assert len(report["rows"]) == 1
    row = report["rows"][0]
    assert row["symbol"] == "AAPL"
    assert row["shares"] == 30
    assert row["purchase_price"] == 170.25
    assert row["account"] == "RRSP"
    assert "AAPL" in row["source_text"]


def test_a_row_quoting_text_the_user_never_pasted_is_dropped(llm):
    llm.reply = _reply(
        {"symbol": "AAPL", "shares": 30, "purchase_price": 170.25,
         "source_text": "AAPL   Apple Inc            30       170.25     5,850.00"},
        # Plausible, well-formed, and about a company the user does not hold.
        {"symbol": "NVDA", "shares": 12, "purchase_price": 400.00,
         "source_text": "NVDA   NVIDIA Corp          12       400.00     4,800.00"},
    )

    report = sp.parse_statement_text(STATEMENT)

    assert [r["symbol"] for r in report["rows"]] == ["AAPL"]
    assert any("NVDA" in d for d in report["dropped"])


def test_whitespace_is_the_only_difference_a_quote_may_have(llm):
    # A clipboard mangles column alignment; it does not invent content.
    llm.reply = _reply({
        "symbol": "TD.TO", "shares": 100, "purchase_price": 85.40,
        "source_text": "TD.TO Toronto-Dominion 100 85.40 9,120.00",
    })

    report = sp.parse_statement_text(STATEMENT)

    assert len(report["rows"]) == 1
    assert report["rows"][0]["symbol"] == "TD.TO"


def test_a_ticker_absent_from_its_own_quoted_line_is_dropped(llm):
    # The quote is real; the symbol attached to it is not what that line says.
    llm.reply = _reply({
        "symbol": "MSFT", "shares": 30, "purchase_price": 170.25,
        "source_text": "AAPL   Apple Inc            30       170.25     5,850.00",
    })

    report = sp.parse_statement_text(STATEMENT)

    assert report["rows"] == []
    assert any("MSFT" in d for d in report["dropped"])


def test_a_row_with_no_ticker_is_dropped_rather_than_guessed(llm):
    llm.reply = _reply({
        "symbol": None, "name": "Apple Inc", "shares": 30,
        "source_text": "AAPL   Apple Inc            30       170.25     5,850.00",
    })

    report = sp.parse_statement_text(STATEMENT)

    assert report["rows"] == []
    assert any("Apple Inc" in d for d in report["dropped"])


def test_a_symbol_that_is_really_a_description_is_dropped(llm):
    llm.reply = _reply({
        "symbol": "Apple Inc Common Stock", "shares": 30,
        "source_text": "AAPL   Apple Inc            30       170.25     5,850.00",
    })

    report = sp.parse_statement_text(STATEMENT)

    assert report["rows"] == []
    assert any("not a usable ticker" in d for d in report["dropped"])


# ---------------------------------------------------------------------------
# Guard two: numbers are echoed, never computed
# ---------------------------------------------------------------------------

def test_a_quantity_not_in_the_quote_is_blanked_and_flagged_not_kept(llm):
    # 34.36 is 5,850.00 / 170.25 — arithmetic on two real numbers, and a share
    # count the statement never printed.
    llm.reply = _reply({
        "symbol": "AAPL", "shares": 34.36, "purchase_price": 170.25,
        "source_text": "AAPL   Apple Inc            30       170.25     5,850.00",
    })

    report = sp.parse_statement_text(STATEMENT)

    assert len(report["rows"]) == 1, "the position is real; only the number is not"
    row = report["rows"][0]
    assert row["shares"] is None
    assert any("34.36" in w for w in row["warnings"])


def test_a_cost_basis_not_in_the_quote_is_blanked_and_flagged(llm):
    llm.reply = _reply({
        "symbol": "AAPL", "shares": 30, "purchase_price": 195.00,
        "source_text": "AAPL   Apple Inc            30       170.25     5,850.00",
    })

    report = sp.parse_statement_text(STATEMENT)

    row = report["rows"][0]
    assert row["shares"] == 30
    assert row["purchase_price"] is None
    assert any("195" in w for w in row["warnings"])


def test_thousands_separators_do_not_make_a_real_number_look_invented(llm):
    text = "CASH   Cash Balance   4,200.00   1.00   4,200.00"
    llm.reply = _reply({
        "symbol": "CASH", "shares": 4200.00, "purchase_price": 1.00,
        "source_text": text,
    })

    report = sp.parse_statement_text(f"Holdings\n{text}\n")

    row = report["rows"][0]
    assert row["shares"] == 4200.0
    assert row["purchase_price"] == 1.0
    assert not any("not in the quoted line" in w for w in row["warnings"])


def test_a_missing_cost_basis_is_stated_rather_than_left_blank_and_silent(llm):
    llm.reply = _reply({
        "symbol": "AAPL", "shares": 30, "purchase_price": None,
        "source_text": "AAPL   Apple Inc            30       170.25     5,850.00",
    })

    report = sp.parse_statement_text(STATEMENT)

    row = report["rows"][0]
    assert row["purchase_price"] is None
    assert any("cost basis" in w.lower() for w in row["warnings"])


def test_a_nan_quantity_never_becomes_a_number(llm):
    # float("nan") does not raise, and json.dumps emits it as a bare NaN token
    # that has taken this app's data off the air before.
    llm.reply = '{"holdings": [{"symbol": "AAPL", "shares": NaN, "purchase_price": 170.25, ' \
                '"source_text": "AAPL   Apple Inc            30       170.25     5,850.00"}]}'

    report = sp.parse_statement_text(STATEMENT)

    # json.loads accepts the bare NaN token, so the row does arrive — the screen
    # that matters is the one that stops it becoming a quantity.
    assert len(report["rows"]) == 1
    assert report["rows"][0]["shares"] is None


# ---------------------------------------------------------------------------
# Fields that are the user's answer, not the model's
# ---------------------------------------------------------------------------

def test_defaults_fill_only_what_the_text_does_not_state_and_say_so(llm):
    llm.reply = _reply({
        "symbol": "AAPL", "shares": 30, "purchase_price": 170.25,
        "account": None, "currency": None,
        "source_text": "AAPL   Apple Inc            30       170.25     5,850.00",
    })

    report = sp.parse_statement_text(STATEMENT, default_account="Pension", default_currency="cad")

    row = report["rows"][0]
    assert row["account"] == "Pension"
    assert row["currency"] == "CAD"
    assert any("defaulted to CAD" in w for w in row["warnings"])


def test_a_stated_currency_beats_the_default(llm):
    llm.reply = _reply({
        "symbol": "AAPL", "shares": 30, "purchase_price": 170.25, "currency": "USD",
        "source_text": "AAPL   Apple Inc            30       170.25     5,850.00",
    })

    report = sp.parse_statement_text(STATEMENT, default_currency="CAD")

    assert report["rows"][0]["currency"] == "USD"


def test_a_currency_this_app_cannot_handle_is_refused_not_carried(llm):
    llm.reply = _reply({
        "symbol": "AAPL", "shares": 30, "purchase_price": 170.25, "currency": "XBT",
        "source_text": "AAPL   Apple Inc            30       170.25     5,850.00",
    })

    report = sp.parse_statement_text(STATEMENT)

    row = report["rows"][0]
    assert row["currency"] == ""
    assert any("XBT" in w for w in row["warnings"])


def test_an_account_naming_a_synced_broker_is_warned_about(llm):
    # load_portfolio() drops CSV rows whose account matches a synced Questrade
    # account, so this row would vanish on the next load with no error anywhere.
    llm.reply = _reply({
        "symbol": "AAPL", "shares": 30, "purchase_price": 170.25, "account": "Questrade TFSA",
        "source_text": "AAPL   Apple Inc            30       170.25     5,850.00",
    })

    report = sp.parse_statement_text(STATEMENT)

    assert any("syncs directly" in w for w in report["rows"][0]["warnings"])


# ---------------------------------------------------------------------------
# Shape and bounds
# ---------------------------------------------------------------------------

def test_the_same_position_listed_twice_is_kept_once(llm):
    quote = "AAPL   Apple Inc            30       170.25     5,850.00"
    llm.reply = _reply(
        {"symbol": "AAPL", "shares": 30, "purchase_price": 170.25, "account": "RRSP", "source_text": quote},
        {"symbol": "AAPL", "shares": 30, "purchase_price": 170.25, "account": "RRSP", "source_text": quote},
    )

    report = sp.parse_statement_text(STATEMENT)

    assert len(report["rows"]) == 1
    assert any("more than once" in d for d in report["dropped"])


def test_the_same_ticker_in_two_accounts_is_two_positions(llm):
    quote = "AAPL   Apple Inc            30       170.25     5,850.00"
    llm.reply = _reply(
        {"symbol": "AAPL", "shares": 30, "purchase_price": 170.25, "account": "RRSP", "source_text": quote},
        {"symbol": "AAPL", "shares": 30, "purchase_price": 170.25, "account": "TFSA", "source_text": quote},
    )

    report = sp.parse_statement_text(STATEMENT)

    assert len(report["rows"]) == 2


def test_row_count_is_capped_and_the_cap_is_reported(llm, monkeypatch):
    monkeypatch.setattr(sp, "MAX_ROWS", 2)
    quote = "AAPL   Apple Inc            30       170.25     5,850.00"
    llm.reply = _reply(*[
        {"symbol": "AAPL", "shares": 30, "purchase_price": 170.25, "account": f"ACC{i}", "source_text": quote}
        for i in range(5)
    ])

    report = sp.parse_statement_text(STATEMENT)

    assert len(report["rows"]) == 2
    assert any("paste the rest separately" in d.lower() for d in report["dropped"])


def test_an_unreadable_response_is_a_reader_failure_not_an_empty_statement(llm):
    # A long paste whose response hit the output cap lands here. Reporting it as
    # ok:True with no rows would tell the user their own text was empty.
    llm.reply = "I could not find any holdings in that text, sorry!"

    report = sp.parse_statement_text(STATEMENT)

    assert report["ok"] is False
    assert report["rows"] == []
    assert "could not be read back" in report["reason"]


def test_a_fenced_json_block_is_still_read(llm):
    llm.reply = "```json\n" + _reply({
        "symbol": "AAPL", "shares": 30, "purchase_price": 170.25,
        "source_text": "AAPL   Apple Inc            30       170.25     5,850.00",
    }) + "\n```"

    report = sp.parse_statement_text(STATEMENT)

    assert len(report["rows"]) == 1


# ---------------------------------------------------------------------------
# The shape a web portal actually pastes in
# ---------------------------------------------------------------------------
# A copied portal table arrives as ONE VALUE PER LINE with no row boundaries:
# the headings first, then every cell of each position in that same order. The
# summary block above it is full of large dollar figures that are not holdings,
# the accounts are listed somewhere the holdings table cannot see, and the cost
# basis sits in the column next to the market price.
#
# Invented throughout — tickers, quantities, prices, account numbers and totals
# are synthetic and do not reconcile to anything. This repo is public and a
# fixture built from a real statement has leaked a real book here before.

PORTAL_PASTE = """All accounts

TFSA - 10000001
Self-directed

RSP - 10000002
Self-directed

+ New account
Total equity (Combined in CAD)
$61,000.00
+4.10% past 6 months
Net deposits
$50,000.00
Total P&L
$11,000.00
Market value
$60,000.00
Cash
$1,000.00
Balanced Portfolio
Portfolio and market activity

Symbol
Asset type
% of portfolio
QTY*
Avg price
Symbol price
Market value
Currency
ZZZA.TO
FICTIONAL ASSET MANAGEMENT LTD BROAD INDEX ETF
Can. equity
40.00%
500.1234
$40.1111
$48.00
$24,006.00
CAD
ZZZB
FICTIONAL TRUST GLOBAL EX-NORTH-AMERICA ETF
Intl. equity
60.00%
300.5000
$90.2222
$119.80
$36,000.00
USD
View disclosures
"""

# The two positions, quoted the way a reader that understood the column order
# would quote them: the whole run of lines belonging to each.
_ZZZA_QUOTE = (
    "ZZZA.TO\nFICTIONAL ASSET MANAGEMENT LTD BROAD INDEX ETF\nCan. equity\n40.00%\n"
    "500.1234\n$40.1111\n$48.00\n$24,006.00\nCAD"
)
_ZZZB_QUOTE = (
    "ZZZB\nFICTIONAL TRUST GLOBAL EX-NORTH-AMERICA ETF\nIntl. equity\n60.00%\n"
    "300.5000\n$90.2222\n$119.80\n$36,000.00\nUSD"
)


def test_a_position_spanning_nine_lines_verifies_against_the_paste(llm):
    """The record is nine lines; the quote is all nine, and it must still check out.

    Only whitespace may differ between a quote and the text, which is what makes
    a multi-line record quotable at all — but it is worth pinning, because if
    this check were line-oriented the entire portal format would be discarded
    row by row with a message blaming the user's paste.
    """
    llm.reply = _reply(
        {"symbol": "ZZZA.TO", "name": "Fictional Broad Index ETF", "shares": 500.1234,
         "purchase_price": 40.1111, "account": None, "currency": "CAD",
         "source_text": _ZZZA_QUOTE},
        {"symbol": "ZZZB", "name": "Fictional Global ETF", "shares": 300.5,
         "purchase_price": 90.2222, "account": None, "currency": "USD",
         "source_text": _ZZZB_QUOTE},
    )

    report = sp.parse_statement_text(PORTAL_PASTE, default_account="Combined")

    assert [r["symbol"] for r in report["rows"]] == ["ZZZA.TO", "ZZZB"]
    assert report["rows"][0]["shares"] == 500.1234
    assert report["rows"][0]["purchase_price"] == 40.1111
    assert report["rows"][1]["currency"] == "USD"


def test_four_decimal_quantities_are_not_mistaken_for_invented_numbers(llm):
    """500.1234 and 90.2222 are what portals actually print, and %g rounds them.

    The digit check has to accept the number as printed. If it did not, the
    quantity on every fractional-share position would be blanked as unverifiable
    — the guard firing on exactly the rows it was built to let through.
    """
    llm.reply = _reply({
        "symbol": "ZZZA.TO", "shares": 500.1234, "purchase_price": 40.1111,
        "currency": "CAD", "source_text": _ZZZA_QUOTE,
    })

    row = sp.parse_statement_text(PORTAL_PASTE)["rows"][0]

    assert row["shares"] == 500.1234
    assert row["purchase_price"] == 40.1111
    assert not [w for w in row["warnings"] if "quoted line" in w]


def test_the_market_price_column_is_not_accepted_as_a_cost_basis(llm):
    """The one mistake in this format that is invisible afterwards.

    'Avg price' and 'Symbol price' are adjacent columns. Taking the second pins
    the position at 0% return for as long as it is held, and it reads as a fact
    rather than as an error. Both numbers are in the quote, so the digit check
    cannot separate them — the prompt must, and this test states the value the
    parser is required to end up with.
    """
    llm.reply = _reply({
        "symbol": "ZZZA.TO", "shares": 500.1234, "purchase_price": 40.1111,
        "currency": "CAD", "source_text": _ZZZA_QUOTE,
    })

    row = sp.parse_statement_text(PORTAL_PASTE)["rows"][0]

    assert row["purchase_price"] == 40.1111, "Avg price is the cost basis"
    assert row["purchase_price"] != 48.00, "Symbol price is the market price"


def test_a_combined_view_leaves_the_account_to_the_user(llm):
    """The accounts are listed where the holdings table cannot see them.

    A merged table says nothing about which account each position sits in, so
    the extraction must not attach one. The user's own answer fills it, and the
    row says out loud that it was filled rather than read.
    """
    llm.reply = _reply({
        "symbol": "ZZZB", "shares": 300.5, "purchase_price": 90.2222,
        "account": None, "currency": "USD", "source_text": _ZZZB_QUOTE,
    })

    named = sp.parse_statement_text(PORTAL_PASTE, default_account="Combined RSP")
    unnamed = sp.parse_statement_text(PORTAL_PASTE)

    assert named["rows"][0]["account"] == "Combined RSP"
    assert unnamed["rows"][0]["account"] == ""
    assert any("no account named" in w.lower() for w in unnamed["rows"][0]["warnings"])


def test_a_column_heading_is_not_a_position(llm):
    """'Symbol' and 'Currency' are shaped exactly like tickers.

    The heading row arrives in the same one-value-per-line stream as the data, so
    a row built from it quotes text that really is in the paste and passes every
    other check here.
    """
    llm.reply = _reply(
        {"symbol": "SYMBOL", "shares": 500.1234, "purchase_price": 40.1111,
         "source_text": "Symbol\nAsset type\n% of portfolio\nQTY*\nAvg price"},
        {"symbol": "ZZZA.TO", "shares": 500.1234, "purchase_price": 40.1111,
         "currency": "CAD", "source_text": _ZZZA_QUOTE},
    )

    report = sp.parse_statement_text(PORTAL_PASTE)

    assert [r["symbol"] for r in report["rows"]] == ["ZZZA.TO"]
    assert any("column heading" in d for d in report["dropped"])


def test_a_portfolio_total_dressed_as_a_holding_is_discarded(llm):
    """The summary block is full of large dollar figures that are not positions.

    'Total equity' quotes real pasted text and carries a real number, so the
    citation check and the digit check both pass it. What stops it is that a
    total has no ticker — and the fallback for a missing ticker is to discard
    the row, never to invent one.
    """
    llm.reply = _reply(
        {"symbol": None, "name": "Total equity", "shares": 61000.00,
         "source_text": "Total equity (Combined in CAD)\n$61,000.00"},
        {"symbol": "ZZZB", "shares": 300.5, "purchase_price": 90.2222,
         "currency": "USD", "source_text": _ZZZB_QUOTE},
    )

    report = sp.parse_statement_text(PORTAL_PASTE)

    assert [r["symbol"] for r in report["rows"]] == ["ZZZB"]
    assert any("Total equity" in d for d in report["dropped"])


def test_the_sector_column_does_not_become_the_asset_type(llm):
    """This format's 'Asset type' means 'Can. equity', not 'Private'.

    Safe by construction — anything that is not exactly Private is Public — but
    the two fields share a name across the seam, so the collapse is pinned here
    rather than left to be rediscovered.
    """
    llm.reply = _reply({
        "symbol": "ZZZA.TO", "shares": 500.1234, "purchase_price": 40.1111,
        "asset_type": "Can. equity", "currency": "CAD", "source_text": _ZZZA_QUOTE,
    })

    assert sp.parse_statement_text(PORTAL_PASTE)["rows"][0]["asset_type"] == "Public"


def test_a_cash_balance_among_the_holdings_is_a_position(llm):
    """Cash is held, unlike every other figure in the summary block.

    The rule that discards totals must not also discard the cash line, or a real
    balance disappears from the portfolio with nothing said about it.
    """
    llm.reply = _reply({
        "symbol": "CASH", "shares": 1000.00, "purchase_price": 1,
        "currency": "CAD", "source_text": "Cash\n$1,000.00",
    })

    report = sp.parse_statement_text(PORTAL_PASTE)

    assert len(report["rows"]) == 1
    assert report["rows"][0]["shares"] == 1000.0


def test_the_parser_writes_nothing(llm):
    """The boundary the whole design rests on, asserted on the source.

    A convention that this module only drafts is worth what a convention is
    worth. The portfolio CSV is the input to every other engine in this app, and
    the one thing a model must not be able to do is put a row in it.
    """
    import inspect

    source = inspect.getsource(sp)
    for forbidden in ("save_portfolio", "open(", "get_data_path", "writer", "clear_cache"):
        assert forbidden not in source, f"statement_parser must not {forbidden!r}"
