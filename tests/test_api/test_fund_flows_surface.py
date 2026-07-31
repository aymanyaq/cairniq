"""
The 5.5 fund-flows READ SURFACE (`templates/monitor.html`).

Executed rather than eyeballed, for the same reason 4.10a's panel is: the whole
value of this surface is the numbers it REFUSES to print, and a refusal is
invisible from the server. ``tools/fund_flows.get_flow_series`` already withholds
``wow`` in three of its four states; the only place that withholding can be
undone is the inline renderer, where no Python test would otherwise reach.

Two hazards, and the second is the expensive one.

``accruing`` is not zero flow. A fund recorded once cannot be measured, and
rendering that as ``0`` or "flat" states the opposite of what is known — the same
silence-read-as-continuity failure 4.10a exists to prevent.

``source_change`` is worse, because the fabricated number would be large and
plausible. FMP and Yahoo disagree about SPY's shares outstanding by roughly 15%;
differencing one against the other reports a definitional gap as a creation event
worth billions. A future edit that "fills in the blank" for these rows would put
a fabricated inflow on screen and break no other test in this suite, so the
numeric cells of non-ready rows are asserted to hold an em dash directly.
"""
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from server import app

TEMPLATE = Path(__file__).resolve().parents[2] / "templates" / "monitor.html"
# The panel renderers call esc/engineNotice/loadEnginePanel/fmtShares, which
# moved to a <script src> when the read panels were split across two pages.
# The extracted inline block alone no longer runs, so the harness rebuilds
# what the browser loads: shared file first, then the page's own block.
SHARED_JS = Path(__file__).resolve().parents[2] / "static" / "js" / "engine_panel.js"

node = shutil.which("node")
requires_node = pytest.mark.skipif(
    node is None, reason="node is required to execute the inline renderer"
)

_SCRIPT_RE = re.compile(r"<script(?![^>]*\bsrc=)[^>]*>(.*?)</script>", re.S | re.I)

# Words that assert a flow of zero. None may appear on a row that has no number.
_FLATNESS_WORDS = ("flat", "unchanged", "no flow", "no change", "steady")

_HARNESS = """
const __els = {};
const __el = (id) => (__els[id] = __els[id] || { id, innerHTML: '' });
globalThis.document = {
  getElementById: __el,
  addEventListener: () => {},
  querySelector: () => null,
  querySelectorAll: () => [],
};
globalThis.window = { location: { reload: () => {}, href: '' } };
globalThis.fetch = async () => { throw new Error('no network in this test'); };

%(script)s

renderFundFlows(JSON.parse(process.argv[2]), __el('flows-body'));
console.log(JSON.stringify({
  body: __el('flows-body').innerHTML,
  coverage: __el('flows-coverage').innerHTML,
}));
"""


def _render(payload: dict) -> dict:
    """Run the template's own renderer over one API payload."""
    script = (SHARED_JS.read_text(encoding="utf-8") + "\n"
              + _SCRIPT_RE.search(TEMPLATE.read_text(encoding="utf-8")).group(1))
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "render.mjs"
        path.write_text(_HARNESS % {"script": script}, encoding="utf-8")
        out = subprocess.run(
            [node, str(path), json.dumps(payload)],
            capture_output=True, text=True, timeout=60,
        )
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout)


def _row(body: str, symbol: str) -> str:
    """The one <tr> whose first cell is `symbol`."""
    rows = body.split("<tbody>")[1].split("</tbody>")[0].split("<tr")
    for r in rows:
        if f">{symbol}</td>" in r:
            return r
    raise AssertionError(f"no row for {symbol} in:\n{body}")


def _cells(row: str) -> list[str]:
    return [c.split("</td>")[0] for c in row.split("<td")[1:]]


def _universe(funds, unresolved=()):
    return {"funds": list(funds), "non_funds": 3,
            "unresolved": list(unresolved), "profiles_read": 1}


def _ready(symbol, change, pct, gap=7, stale=False, source="fmp"):
    return {
        "symbol": symbol, "status": "ready", "days_recorded": 9,
        "first_date": "2026-07-20", "latest_date": "2026-07-29",
        "latest_shares": 1_000_000_000.0, "source": source, "points": [],
        "wow": {"from_date": "2026-07-22", "to_date": "2026-07-29", "gap_days": gap,
                "shares_change": change, "percent_change": pct,
                "direction": "creations" if change > 0 else "redemptions",
                "stale_window": stale},
        "note": f"Flow measured over {gap} days from a single source ({source}).",
    }


def _accruing(symbol, days_recorded=1, days_until_ready=6):
    return {
        "symbol": symbol, "status": "accruing", "days_recorded": days_recorded,
        "first_date": "2026-07-29", "latest_date": "2026-07-29",
        "latest_shares": 500_000_000.0, "source": "fmp", "points": [],
        "wow": None, "days_until_ready": days_until_ready,
        "note": "Recording is live, but a week-over-week flow needs two points "
                "at least 6 days apart. No flow number is being reported.",
    }


def _source_change(symbol):
    return {
        "symbol": symbol, "status": "source_change", "days_recorded": 12,
        "first_date": "2026-07-10", "latest_date": "2026-07-29",
        "latest_shares": 900_000_000.0, "source": "yahoo", "points": [],
        "wow": None,
        "note": "The two points come from different sources (fmp → yahoo).",
    }


def _no_data(symbol):
    return {
        "symbol": symbol, "status": "no_data", "days_recorded": 0,
        "note": f"No shares-outstanding rows recorded for {symbol}. "
                "This is NOT a zero-flow reading.",
    }


def test_the_panel_is_wired_into_the_monitor_page():
    res = TestClient(app).get("/monitor")
    assert res.status_code == 200
    assert 'id="flows-body"' in res.text
    assert "/api/portfolio/fund-flows" in res.text


@requires_node
def test_accruing_shows_no_number_and_does_not_claim_flatness():
    out = _render({"universe": _universe(["VOO"]),
                   "fund_series": {"VOO": _accruing("VOO")}})
    body = out["body"]
    cells = _cells(_row(body, "VOO"))

    # Share change and percent columns — an em dash, never a 0.
    assert cells[4].endswith(">—"), cells[4]
    assert cells[5].endswith(">—"), cells[5]
    assert "Accruing" in cells[1]

    # Scoped to the ROW, for the reason 4.10a's panel scopes its causal-word
    # check to the tbody: the explanatory notice is allowed to say "flat"
    # because it says it in order to deny it ("not a set of flat ones").
    row = _row(body, "VOO").lower()
    for word in _FLATNESS_WORDS:
        assert word not in row, f"an unmeasurable fund is described as {word!r}"
    # It must say what it is waiting for, not just that it is waiting.
    assert "6 more days" in body


@requires_node
def test_a_source_change_never_renders_a_delta():
    """The expensive fabrication: ~15% provider disagreement as a creation event."""
    out = _render({"universe": _universe(["SPY"]),
                   "fund_series": {"SPY": _source_change("SPY")}})
    body = out["body"]
    cells = _cells(_row(body, "SPY"))

    assert cells[4].endswith(">—"), cells[4]
    assert cells[5].endswith(">—"), cells[5]
    assert "Source changed" in cells[1]
    # And the reason gets stated below the table, where it cannot be mistaken
    # for a rendering gap that someone should go and fill in.
    assert "a source change is not a flow" in body.lower()
    assert "15%" in body


@requires_node
def test_no_data_is_not_a_zero_reading():
    out = _render({"universe": _universe(["ZAG"]),
                   "fund_series": {"ZAG": _no_data("ZAG")}})
    body = out["body"]
    cells = _cells(_row(body, "ZAG"))
    assert cells[4].endswith(">—")
    assert "Not recorded" in cells[1]
    assert "no shares-outstanding rows recorded" in body.lower()


@requires_node
def test_a_measured_fund_shows_its_flow_and_its_window():
    out = _render({"universe": _universe(["QQQ"]),
                   "fund_series": {"QQQ": _ready("QQQ", 12_500_000.0, 1.42)}})
    body = out["body"]
    cells = _cells(_row(QQQ_BODY := body, "QQQ"))
    assert "+12,500,000" in cells[4]
    assert "+1.42%" in cells[5]
    assert "2026-07-22" in cells[6] and "2026-07-29" in cells[6]
    assert "Measured" in cells[1]
    # A single measurable fund must not draw the panel-wide "none yet" banner.
    assert "no fund has a week-over-week flow yet" not in QQQ_BODY.lower()


@requires_node
def test_a_redemption_renders_as_negative_not_as_absent():
    out = _render({"universe": _universe(["XIU"]),
                   "fund_series": {"XIU": _ready("XIU", -8_000_000.0, -0.91)}})
    cells = _cells(_row(out["body"], "XIU"))
    assert "8,000,000" in cells[4]
    assert "—" not in cells[4], "a measured redemption was rendered as unmeasurable"
    assert "0.91%" in cells[5]


@requires_node
def test_a_window_wider_than_a_week_says_so():
    out = _render({"universe": _universe(["VTI"]),
                   "fund_series": {"VTI": _ready("VTI", 3_000_000.0, 0.2,
                                                 gap=19, stale=True)}})
    cells = _cells(_row(out["body"], "VTI"))
    assert "19d" in cells[6]
    assert "wider than a week" in cells[6]


@requires_node
def test_the_all_accruing_case_says_it_once_for_the_whole_panel():
    out = _render({"universe": _universe(["VOO", "VEA"]),
                   "fund_series": {"VOO": _accruing("VOO"),
                                   "VEA": _accruing("VEA", days_until_ready=3)}})
    body = out["body"].lower()
    assert "no fund has a week-over-week flow yet" in body
    assert "accruing series, not a set of flat ones" in body
    assert "2 fund(s) are recording" in body
    assert "0 of 2" in out["coverage"]
    assert "2 withheld, not zero" in out["coverage"]


@requires_node
def test_a_fund_with_nothing_on_file_is_not_described_as_recording():
    """Measured against the live demo profile, where 4 of 5 funds are `no_data`.

    The banner originally said "All 5 fund(s) are recording", which is a
    reassuring wait message wrapped around a feed that has never produced a row
    for those symbols. `accruing` is a claim about a recorder that IS running;
    `no_data` is a claim that it is not.
    """
    out = _render({
        "universe": _universe(["PSA.TO", "SPY", "VTI"]),
        "fund_series": {"PSA.TO": _accruing("PSA.TO", days_until_ready=5),
                        "SPY": _no_data("SPY"),
                        "VTI": _no_data("VTI")},
    })
    body = out["body"].lower()
    assert "1 fund(s) are recording" in body
    assert "2 have nothing on file at all" in body
    assert "all 3 fund(s) are recording" not in body


@requires_node
def test_coverage_counts_measurable_funds_not_recorded_ones():
    """A fund can be recording every day and still be unmeasurable — the header
    must count what can be READ, not what has been written."""
    out = _render({"universe": _universe(["QQQ", "SPY", "VOO"]),
                   "fund_series": {"QQQ": _ready("QQQ", 1.0, 0.1),
                                   "SPY": _source_change("SPY"),
                                   "VOO": _accruing("VOO")}})
    coverage = out["coverage"]
    assert "1 of 3" in coverage
    assert "2 withheld, not zero" in coverage


@requires_node
def test_an_unresolved_holding_is_not_reported_as_having_no_funds():
    """Empty-store, two causes: a portfolio without funds versus a classifier
    that could not answer. The second one is hiding holdings."""
    out = _render({"universe": _universe([], unresolved=["BRK.B", "XYZ.TO"]),
                   "fund_series": {}})
    body = out["body"].lower()
    assert "no fund could be classified" in body
    assert "brk.b" in body and "xyz.to" in body
    assert "not a portfolio without funds" in body
    assert "no fund holdings" not in body


@requires_node
def test_a_genuine_absence_of_funds_says_so_plainly():
    out = _render({"universe": _universe([]), "fund_series": {}})
    body = out["body"].lower()
    assert "no fund holdings" in body
    assert "no fund could be classified" not in body


@requires_node
def test_unresolved_holdings_are_named_even_when_other_funds_resolved():
    """The classifier's own gap must not be swallowed by a table that happens to
    have rows in it."""
    out = _render({"universe": _universe(["QQQ"], unresolved=["ACME.XX"]),
                   "fund_series": {"QQQ": _ready("QQQ", 1.0, 0.1)}})
    body = out["body"]
    assert "ACME.XX" in body
    assert "unresolved and excluded" in body.lower()
