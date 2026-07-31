"""
The 4.10a reconciliation READ SURFACE (`templates/portfolio_editor.html`).

Why this is executed rather than eyeballed: the panel's entire job is to hold
three claims apart — the recorder has never run (``no_data``), exactly one
snapshot exists so far (``accruing``), and the portfolio genuinely did not move
(``ready`` with an empty change list). Collapsing any of them into the others is
the precise failure `tools/portfolio_reconciliation` was written to prevent, and
it is invisible from the server: the distinction lives entirely inside an inline
renderer that no Python test would otherwise touch. So the renderer itself is
run here, in node, once per status.

The second guard is vocabulary, and it is the one most likely to erode. Every
change the engine emits carries ``cause: "unclassified"`` — a quantity delta is
equally consistent with a trade, a deposit, a transfer in kind, a reinvested
dividend, a fee, an FX conversion, a split or a corporate action. A future edit
relabelling "Quantity up" as "Bought" would pass every other test in this suite
while putting a fabrication on screen, so the change rows are checked for causal
words directly. The explanatory note is deliberately exempt: it names those
causes in order to deny them.
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

TEMPLATE = Path(__file__).resolve().parents[2] / "templates" / "portfolio_editor.html"
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

# Words that assert a CAUSE. None of them may appear in a change row.
_CAUSAL_WORDS = ("bought", "sold", "buy", "sell", "purchase", "deposit",
                 "withdrawal", "withdrew", "trade", "dividend", "transfer")

# A DOM small enough to be obviously correct: the renderer only ever reads
# getElementById().innerHTML. addEventListener is a no-op so the page's own
# DOMContentLoaded work — and the fetch it would start — never fires here.
# A DOM small enough to be obviously correct. Elements carry querySelectorAll
# because the cause cell wires change handlers after render; it returns nothing,
# so no listener is attached and the panel's own fetch never fires here.
_HARNESS = """
const __els = {};
const __el = (id) => (__els[id] = __els[id] || {
  id, innerHTML: '', querySelectorAll: () => [],
});
globalThis.document = {
  getElementById: __el,
  addEventListener: () => {},
  querySelector: () => null,
  querySelectorAll: () => [],
};
globalThis.window = { location: { reload: () => {}, href: '' } };
globalThis.fetch = async () => { throw new Error('no network in this test'); };

%(script)s

// The cause options normally arrive from /api/portfolio/classification-options
// before the first render. Injected here so both paths are testable: with them
// the cell is a SELECT (an entry point), without them a read-only badge.
const __opts = JSON.parse(process.argv[3] || 'null');
if (__opts) { CAUSE_OPTIONS = __opts; }

renderReconciliation(JSON.parse(process.argv[2]), __el('recon-body'));
console.log(JSON.stringify({
  body: __el('recon-body').innerHTML,
  coverage: __el('recon-coverage').innerHTML,
}));
"""

# Mirrors tools/portfolio_classification.CAUSES. The contract between them is
# asserted for real in test_classification_options_match_the_engine below.
_OPTIONS = [
    {"value": "external_inflow", "label": "Money in", "description": "d", "is_external_flow": True},
    {"value": "external_outflow", "label": "Money out", "description": "d", "is_external_flow": True},
    {"value": "trade", "label": "Trade", "description": "d", "is_external_flow": False},
    {"value": "drip", "label": "Dividend reinvested", "description": "d", "is_external_flow": False},
]


def _render(payload: dict, options: list | None = None) -> dict:
    """Run the template's own renderer over one API payload."""
    script = (SHARED_JS.read_text(encoding="utf-8") + "\n"
              + _SCRIPT_RE.search(TEMPLATE.read_text(encoding="utf-8")).group(1))
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "render.mjs"
        path.write_text(_HARNESS % {"script": script}, encoding="utf-8")
        out = subprocess.run(
            [node, str(path), json.dumps(payload), json.dumps(options)],
            capture_output=True, text=True, timeout=60,
        )
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout)


_OPTION_RE = re.compile(r"<option\b.*?</option>", re.S | re.I)


def _rows(body: str) -> str:
    """Just the change rows — the note names causes in order to deny them.

    ``<option>`` elements are stripped, and that exemption is narrow and
    deliberate. Since the classification step shipped, the cause cell is a
    SELECT: the panel offers "Trade", "Money in", "Dividend reinvested" as
    choices a human may make. An offered choice is not an assertion, and the
    guard below is about assertions — that the panel never states a cause the
    engine did not get from a person. What the row actually CLAIMS is the
    selected option, which `_selected` reads and the tests check separately.
    """
    rows = body.split("<tbody>")[1].split("</tbody>")[0]
    return _OPTION_RE.sub("", rows).lower()


def _rows_raw(body: str) -> str:
    """The change rows with their options intact."""
    return body.split("<tbody>")[1].split("</tbody>")[0]


def _selected(row_html: str) -> str:
    """The option a row is actually asserting — i.e. what the user sees chosen."""
    m = re.search(r'<option value="([^"]+)"[^>]*\bselected\b', row_html, re.I)
    return m.group(1) if m else ""


_UNSTATED_CHANGE = {
    "kind": "quantity_increase", "account": "TFSA", "symbol": "VOO",
    "currency": "CAD", "is_cash": False, "prior_shares": 100.0,
    "current_shares": 112.5, "delta": 12.5, "prior_date": "2026-07-28",
    "current_date": "2026-07-29", "cause": "unclassified", "classified": False,
    "is_external_flow": False, "spans_gap": False, "gap_days": 1,
}


def _CLASSIFICATION(complete, unclassified, seen, inflows=0, outflows=0,
                    priced=True, unpriced=0, net=None):
    """The shape tools/portfolio_classification.flow_summary returns.

    `priced` / `unpriced_flow_count` / `flow_amount_base` were added 2026-07-30
    and this fixture is kept in step deliberately. A fixture that lags the payload
    it claims to mirror is how the percent-vs-fraction yield bug stayed green for
    months in `test_asset_location.py` — the mock was written from what the reader
    expected rather than from what the producer sends, so it could only ever
    confirm the reader.
    """
    return {
        "complete": complete, "changes_seen": seen,
        "classified_count": seen - unclassified, "unclassified_count": unclassified,
        "external_inflows": inflows, "external_outflows": outflows,
        "inflow_units": 0.0, "outflow_units": 0.0,
        "priced": priced, "unpriced_flow_count": unpriced, "flow_amount_base": net,
        "note": ("Every observed change has a stated cause; external flows are "
                 "complete for this window." if complete else
                 f"{unclassified} of {seen} observed change(s) have no stated cause. "
                 "Any of them could be money in or out, so the flows below are a "
                 "LOWER BOUND and a time-weighted return computed on them would be "
                 "wrong by an unknown amount in an unknown direction."),
    }


_EMPTY_COVERAGE = {"observed_days": 0, "calendar_days": 0, "missing_days": 0,
                   "first_date": "", "latest_date": "", "gaps": []}

# The live shape measured on cairniq 2026-07-29: a span of 89 days holding only
# 82 observations. The span figure alone cannot see the other 7.
_HOLED_COVERAGE = {"observed_days": 82, "calendar_days": 89, "missing_days": 7,
                   "first_date": "2026-05-02", "latest_date": "2026-07-29",
                   "gaps": [{"after": "2026-06-01", "before": "2026-06-05", "missing_days": 3},
                            {"after": "2026-07-01", "before": "2026-07-05", "missing_days": 3}]}


def test_the_panel_is_wired_into_the_portfolio_page():
    res = TestClient(app).get("/portfolio")
    assert res.status_code == 200
    assert 'id="recon-body"' in res.text
    assert "/api/portfolio/reconciliation" in res.text


@requires_node
def test_no_data_is_never_rendered_as_an_unchanged_portfolio():
    out = _render({
        "status": "no_data", "coverage": _EMPTY_COVERAGE, "changes": [],
        "note": "No position snapshots have been recorded yet.",
    })
    body = out["body"].lower()
    assert "recorder has not run" in body
    # The claim the engine forbids: silence read as continuity.
    assert "no changes observed" not in body
    assert "<tbody>" not in body


@requires_node
def test_accruing_reports_the_recorder_and_withholds_a_change_list():
    out = _render({
        "status": "accruing", "coverage": dict(_EMPTY_COVERAGE, observed_days=1, calendar_days=1),
        "changes": [], "snapshots": 1, "days_until_ready": 1,
        "note": "Recording is live (first snapshot 2026-07-29), but a reconciliation "
                "needs two snapshots to compare.",
    })
    body = out["body"].lower()
    assert "accruing" in body
    assert "no changes observed" not in body
    assert "<tbody>" not in body


@requires_node
def test_only_two_real_snapshots_may_report_no_changes():
    out = _render({
        "status": "ready", "coverage": _HOLED_COVERAGE,
        "prior_date": "2026-07-28", "current_date": "2026-07-29",
        "spans_gap": False, "changes": [], "change_count": 0,
        "truncated": False, "material_cash_min": 1.0, "note": "0 change(s) observed.",
    })
    assert "no changes observed" in out["body"].lower()


@requires_node
def test_change_rows_are_unclassified_and_name_no_cause():
    payload = {
        "status": "ready", "coverage": _HOLED_COVERAGE,
        "prior_date": "2026-07-28", "current_date": "2026-07-29", "spans_gap": False,
        "changes": [
            {"kind": "quantity_increase", "account": "TFSA", "symbol": "VOO", "currency": "CAD",
             "is_cash": False, "prior_shares": 100.0, "current_shares": 112.5, "delta": 12.5,
             "prior_date": "2026-07-28", "current_date": "2026-07-29",
             "cause": "unclassified", "spans_gap": False, "gap_days": 1},
            {"kind": "cash_decrease", "account": "RRSP", "symbol": "CAD", "currency": "CAD",
             "is_cash": True, "prior_shares": 5000.0, "current_shares": 1200.0, "delta": -3800.0,
             "prior_date": "2026-07-28", "current_date": "2026-07-29",
             "cause": "unclassified", "spans_gap": False, "gap_days": 1},
            {"kind": "position_closed", "account": "Margin", "symbol": "KO", "currency": "USD",
             "is_cash": False, "prior_shares": 40.0, "current_shares": None, "delta": -40.0,
             "prior_date": "2026-07-28", "current_date": "2026-07-29",
             "cause": "unclassified", "spans_gap": False, "gap_days": 1},
        ],
        "change_count": 3, "truncated": False, "material_cash_min": 1.0,
        "note": "3 change(s) observed. Every change is UNCLASSIFIED: a quantity delta is "
                "equally consistent with a trade, a deposit, a transfer, a reinvested "
                "dividend, a fee or a corporate action.",
    }
    body = out = _render(payload)["body"]
    rows = _rows(body)

    assert rows.count("unclassified") == 3
    for word in _CAUSAL_WORDS:
        assert word not in rows, f"change rows name a cause: {word!r}"

    # Accounts are part of the identity — the same ticker in two shelters is two
    # positions, and collapsing them would net a transfer to zero.
    for account in ("TFSA", "RRSP", "Margin"):
        assert account in body
    # A closed position shows an em dash for "current", not a zero: recording it
    # as 0 would manufacture a disposal.
    assert "—" in body
    # The disclaimer that makes the table honest must survive.
    assert "unclassified" in out.lower()


@requires_node
def test_coverage_reports_observed_days_not_the_span():
    out = _render({
        "status": "ready", "coverage": _HOLED_COVERAGE,
        "prior_date": "2026-07-28", "current_date": "2026-07-29",
        "spans_gap": False, "changes": [], "change_count": 0,
        "truncated": False, "material_cash_min": 1.0, "note": "",
    })
    coverage = out["coverage"]
    # 82 of 89 — never the bare 89-day span, which is blind to the 7 holes.
    assert "82 of 89" in coverage
    assert "7 missing in 2 gaps" in coverage


@requires_node
def test_a_change_across_a_gap_is_not_attributed_to_a_date():
    out = _render({
        "status": "ready", "coverage": _HOLED_COVERAGE,
        "prior_date": "2026-07-24", "current_date": "2026-07-29", "spans_gap": True,
        "changes": [
            {"kind": "quantity_increase", "account": "TFSA", "symbol": "VOO", "currency": "CAD",
             "is_cash": False, "prior_shares": 100.0, "current_shares": 112.5, "delta": 12.5,
             "prior_date": "2026-07-24", "current_date": "2026-07-29",
             "cause": "unclassified", "spans_gap": True, "gap_days": 5},
        ],
        "change_count": 1, "truncated": False, "material_cash_min": 1.0, "note": "",
    })
    body = out["body"].lower()
    assert "not consecutive" in body
    assert "5 days apart" in body
    assert "no change below can be attributed to a single date" in body


@requires_node
def test_the_cause_cell_is_an_entry_point_not_a_label():
    """The lesson from `risk_constraints`: a store filed as blocked-on-the-user
    while having no reachable writer. The cause column must be a control."""
    out = _render({
        "status": "ready", "coverage": _HOLED_COVERAGE,
        "prior_date": "2026-07-28", "current_date": "2026-07-29", "spans_gap": False,
        "changes": [dict(_UNSTATED_CHANGE)], "change_count": 1, "truncated": False,
        "material_cash_min": 1.0, "note": "",
        "classification": _CLASSIFICATION(complete=False, unclassified=1, seen=1),
    }, options=_OPTIONS)
    body = out["body"]
    assert "<select" in body
    assert 'class="cause-select"' in body or "cause-select" in body
    # Every engine cause is offered, so no cause is unreachable from the screen.
    for value in ("external_inflow", "external_outflow", "trade", "drip"):
        assert f'value="{value}"' in body
    # And an unstated change asserts exactly that.
    assert _selected(_rows_raw(body)) == "unclassified"
    assert "Not stated" in body


@requires_node
def test_without_options_the_cell_degrades_to_a_badge_not_an_empty_control():
    """An empty <select> is a control that silently cannot be used."""
    out = _render({
        "status": "ready", "coverage": _HOLED_COVERAGE,
        "prior_date": "2026-07-28", "current_date": "2026-07-29", "spans_gap": False,
        "changes": [dict(_UNSTATED_CHANGE)], "change_count": 1, "truncated": False,
        "material_cash_min": 1.0, "note": "",
        "classification": _CLASSIFICATION(complete=False, unclassified=1, seen=1),
    })  # no options injected
    body = out["body"]
    assert "<select" not in body
    assert "Unclassified" in body


@requires_node
def test_a_stated_cause_shows_as_selected_and_flags_a_flow():
    out = _render({
        "status": "ready", "coverage": _HOLED_COVERAGE,
        "prior_date": "2026-07-28", "current_date": "2026-07-29", "spans_gap": False,
        "changes": [dict(_UNSTATED_CHANGE, cause="external_inflow", classified=True,
                         is_external_flow=True, cause_label="Money in")],
        "change_count": 1, "truncated": False, "material_cash_min": 1.0, "note": "",
        "classification": _CLASSIFICATION(complete=True, unclassified=0, seen=1,
                                          inflows=1),
    }, options=_OPTIONS)
    body = out["body"]
    assert _selected(_rows_raw(body)) == "external_inflow"
    # The TWR-relevant fact gets its own mark: this one leaves the capital base.
    assert "Flow" in body
    assert "every change has a stated cause" in body.lower()
    assert "time-weighted return (4.10) can use this window" in body.lower()


@requires_node
def test_an_incomplete_window_reads_as_blocked_not_as_progress():
    """A TWR over three known deposits while a fourth is unstated is not 75%
    right — it is wrong by an unknown amount in an unknown direction."""
    out = _render({
        "status": "ready", "coverage": _HOLED_COVERAGE,
        "prior_date": "2026-07-28", "current_date": "2026-07-29", "spans_gap": False,
        "changes": [dict(_UNSTATED_CHANGE)], "change_count": 4, "truncated": False,
        "material_cash_min": 1.0, "note": "",
        "classification": _CLASSIFICATION(complete=False, unclassified=1, seen=4,
                                          inflows=3),
    }, options=_OPTIONS)
    body = out["body"].lower()
    assert "1 change(s) have no stated cause" in body
    assert "lower bound" in body
    # The completed-progress framing must not appear.
    assert "can use this window" not in body


@requires_node
def test_a_stale_classification_is_explained_rather_than_silently_reverted():
    """Reverting to a blank control without saying why looks like a failed save."""
    out = _render({
        "status": "ready", "coverage": _HOLED_COVERAGE,
        "prior_date": "2026-07-28", "current_date": "2026-07-29", "spans_gap": False,
        "changes": [dict(_UNSTATED_CHANGE, stale_classification={
            "cause": "external_inflow", "classified_at": "2026-07-29T09:00:00",
            "against_delta": 12.5, "now_delta": 400.0,
            "note": "This change was classified, then the underlying snapshot "
                    "changed. The earlier answer is not being applied to different "
                    "numbers — it needs restating.",
        })],
        "change_count": 1, "truncated": False, "material_cash_min": 1.0, "note": "",
        "classification": _CLASSIFICATION(complete=False, unclassified=1, seen=1),
    }, options=_OPTIONS)
    body = out["body"]
    assert "Previously stated as" in body
    assert "external_inflow" in body
    assert "12.5" in body and "400" in body
    assert "needs restating" in body
    # It is still unstated, and the control says so.
    assert _selected(_rows_raw(body)) == "unclassified"


@requires_node
def test_a_truncated_list_says_how_many_it_is_hiding():
    changes = [
        {"kind": "quantity_increase", "account": "TFSA", "symbol": f"SYM{i}", "currency": "CAD",
         "is_cash": False, "prior_shares": 1.0, "current_shares": 2.0, "delta": 1.0,
         "prior_date": "2026-07-28", "current_date": "2026-07-29",
         "cause": "unclassified", "spans_gap": False, "gap_days": 1}
        for i in range(50)
    ]
    out = _render({
        "status": "ready", "coverage": _HOLED_COVERAGE,
        "prior_date": "2026-07-28", "current_date": "2026-07-29", "spans_gap": False,
        "changes": changes, "change_count": 137, "truncated": True,
        "material_cash_min": 1.0, "note": "",
    })
    assert "showing 50 of 137" in out["body"].lower()


# ---------------------------------------------------------------------------
# The amount cell — 4.10's second axis (2026-07-30)
# ---------------------------------------------------------------------------
# `amount_base` reached `classify_change` and the API that morning with no writer
# in the UI, so 4.10 could only ever answer `flows_incomplete → unpriced_flows`.
# That is indistinguishable, from the outside, from a user who declined to
# answer — which is the exact failure this file's cause-cell tests were written
# about, re-introduced hours after being cited. These tests are its guard.
def _flow_change(cause="external_inflow", **kw):
    base = dict(_UNSTATED_CHANGE, symbol="CASH", is_cash=True,
                prior_shares=1000.0, current_shares=6000.0, delta=5000.0,
                kind="cash_increase", cause=cause, classified=True,
                is_external_flow=True, cause_label="Money in")
    base.update(kw)
    return base


def _ready(changes, classification, base_currency="CAD"):
    return {
        "status": "ready", "coverage": _HOLED_COVERAGE, "base_currency": base_currency,
        "prior_date": "2026-07-28", "current_date": "2026-07-29", "spans_gap": False,
        "changes": changes, "change_count": len(changes), "truncated": False,
        "material_cash_min": 1.0, "note": "", "classification": classification,
    }


@requires_node
def test_a_flow_gets_an_amount_box_and_the_header_names_the_currency():
    """The box is the writer. Without it the store has no reachable author and
    `unpriced_flows` reads as the user's silence rather than ours."""
    out = _render(_ready([_flow_change()],
                         _CLASSIFICATION(complete=True, unclassified=0, seen=1,
                                         inflows=1, priced=False, unpriced=1)),
                  options=_OPTIONS)
    body = out["body"]
    assert "flow-amount" in body
    assert 'type="number"' in body
    # The amount is in BASE currency and the row beside it says CAD only by
    # coincidence — an unlabelled money box is the units trap this repo keeps
    # finding, so the column header must name the currency.
    assert "Amount (CAD)" in body


@requires_node
def test_a_non_flow_cause_gets_no_box_and_says_why_rather_than_going_blank():
    """A blank cell reads as "you have not filled this in". Only external flows
    move the capital base, so nothing is being asked of a trade or a DRIP."""
    out = _render(_ready([_flow_change(cause="drip", is_external_flow=False,
                                       cause_label="Dividend reinvested")],
                         _CLASSIFICATION(complete=True, unclassified=0, seen=1)),
                  options=_OPTIONS)
    body = out["body"]
    assert "flow-amount" not in body
    assert "no amount needed" in body.lower()


@requires_node
def test_an_unstated_change_asks_for_the_cause_first():
    out = _render(_ready([dict(_UNSTATED_CHANGE)],
                         _CLASSIFICATION(complete=False, unclassified=1, seen=1)),
                  options=_OPTIONS)
    body = out["body"]
    assert "flow-amount" not in body
    assert "set a cause first" in body.lower()


@requires_node
def test_a_stated_cause_with_no_amount_still_blocks_4_10_on_screen():
    """THE test for this cell. `complete` and `priced` are two axes, and the
    banner read only the first until today — telling the user the window was
    finished while the engine went on refusing to compute a return."""
    out = _render(_ready([_flow_change()],
                         _CLASSIFICATION(complete=True, unclassified=0, seen=1,
                                         inflows=1, priced=False, unpriced=1)),
                  options=_OPTIONS)
    body = out["body"].lower()
    assert "1 flow(s) have a cause but no amount" in body
    # The completed framing must not appear while an amount is missing.
    assert "can use this window" not in body


@requires_node
def test_both_axes_satisfied_is_what_unblocks_the_window():
    out = _render(_ready([_flow_change(amount_base=5000.0)],
                         _CLASSIFICATION(complete=True, unclassified=0, seen=1,
                                         inflows=1, priced=True, net=5000.0)),
                  options=_OPTIONS)
    body = out["body"].lower()
    assert "stated cause and amount" in body
    assert "time-weighted return (4.10) can use this window" in body


@requires_node
def test_a_recorded_amount_prefills_as_a_magnitude_with_no_minus_sign():
    """The sign is derived from the cause server-side, so showing a stored
    -5000 back in the box would invite the user to re-type it negative and make
    the sign a data-entry convention again."""
    out = _render(_ready([_flow_change(cause="external_outflow",
                                       cause_label="Money out",
                                       amount_base=-5000.0)],
                         _CLASSIFICATION(complete=True, unclassified=0, seen=1,
                                         outflows=1, priced=True, net=-5000.0)),
                  options=_OPTIONS)
    body = out["body"]
    assert 'value="5000"' in body
    assert 'value="-5000"' not in body
    # And the direction is stated in words instead.
    assert ">out<" in body


@requires_node
def test_the_amount_column_never_asserts_a_cause():
    """The vocabulary guard this file is built around, extended to the new cell."""
    out = _render(_ready([_flow_change(amount_base=5000.0)],
                         _CLASSIFICATION(complete=True, unclassified=0, seen=1,
                                         inflows=1, priced=True, net=5000.0)),
                  options=_OPTIONS)
    rows = _rows(out["body"])
    for word in ("bought", "sold", "purchase", "withdrew"):
        assert word not in rows, word
