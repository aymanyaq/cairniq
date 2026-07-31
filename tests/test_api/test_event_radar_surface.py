"""
The 3.5b event-radar READ SURFACE (`templates/monitor.html`).

Executed in node for the reason the 4.10a and 5.5 suites give: the guarantee
lives in the renderer, and the server cannot see it.

The guarantee here is `unknown`. ``tools/event_radar.build_event_radar`` collects
the symbols whose date could not be established and returns them separately
because — in its own words — "no earnings coming" and "the provider did not tell
us" are different facts and only one of them is safe to act on. A panel that
draws an empty calendar while `unknown` is non-empty reports silence about names
nobody actually checked, and a holder reading "nothing ahead" would be reading a
provider outage. So the empty state is asserted to branch on `unknown` first.

The second guarantee is about delivery. This panel reads the CACHED radar (6h
TTL) while the alert path calls ``build_event_radar()`` directly, precisely
because a stale countdown changes the answer rather than aging it. The panel may
therefore mark a row as sitting inside the T-3/T-1 window, but must never say an
alert was or will be sent — it does not read the alert store. Delivery-claiming
words are asserted absent.
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

# Words that assert DELIVERY. The panel reports a calendar, not an outbox.
_DELIVERY_WORDS = ("was sent", "we sent", "alert sent", "notified", "delivered",
                   "you were told", "we alerted", "alert fired")

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

renderEventRadar(JSON.parse(process.argv[2]), __el('radar-body'));
console.log(JSON.stringify({
  body: __el('radar-body').innerHTML,
  coverage: __el('radar-coverage').innerHTML,
}));
"""


def _render(payload: dict) -> dict:
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


def _row(body: str, name: str) -> str:
    rows = body.split("<tbody>")[1].split("</tbody>")[0].split("<tr")
    for r in rows:
        if f">{name}</td>" in r:
            return r
    raise AssertionError(f"no row for {name} in:\n{body}")


def _event(kind, symbol, date, days, label):
    return {"kind": kind, "symbol": symbol, "date": date,
            "days_until": days, "label": label}


_AS_OF = "2026-07-29T20:00:00"


def test_the_panel_is_wired_into_the_monitor_page():
    res = TestClient(app).get("/monitor")
    assert res.status_code == 200
    assert 'id="radar-body"' in res.text
    assert "/api/portfolio/event-radar" in res.text


@requires_node
def test_an_empty_calendar_with_unknowns_is_not_reported_as_clear():
    """The failure this engine exists to prevent: a provider outage read as calm."""
    out = _render({"as_of": _AS_OF, "checked": 4, "events": [],
                   "unknown": ["SHOP.TO", "ATD.TO", "BN"]})
    body = out["body"].lower()
    assert "3 holding(s) returned no earnings date" in body
    assert "not the same as having nothing scheduled" in body
    # The all-clear must not appear.
    assert "no dated events ahead" not in body
    assert "clear calendar" not in body


@requires_node
def test_a_genuinely_clear_calendar_says_so_and_says_it_is_complete():
    out = _render({"as_of": _AS_OF, "checked": 6, "events": [], "unknown": []})
    body = out["body"].lower()
    assert "no dated events ahead" in body
    assert "all 6 held name(s) answered" in body
    assert "this is a clear calendar, not a missing one" in body


@requires_node
def test_events_render_nearest_first_with_their_countdown():
    out = _render({
        "as_of": _AS_OF, "checked": 3, "unknown": [],
        "events": [
            _event("earnings", "AAPL", "2026-07-30", 1, "AAPL reports earnings"),
            _event("ex_dividend", "VOO", "2026-08-05", 7, "VOO goes ex-dividend"),
            _event("fomc", None, "2026-09-16", 49, "FOMC rate decision"),
        ],
    })
    body = out["body"]
    assert "Tomorrow" in _row(body, "AAPL")
    assert "7 days" in _row(body, "VOO")
    # A macro event has no symbol and must still get a name in the table.
    assert "49 days" in _row(body, "Markets")
    assert "FOMC rate decision" in body


@requires_node
def test_the_alert_window_is_marked_without_claiming_an_alert_was_sent():
    out = _render({
        "as_of": _AS_OF, "checked": 2, "unknown": [],
        "events": [
            _event("earnings", "MSFT", "2026-08-01", 3, "MSFT reports earnings"),
            _event("earnings", "NVDA", "2026-08-08", 10, "NVDA reports earnings"),
        ],
    })
    body = out["body"]
    assert "In alert window" in _row(body, "MSFT")   # T-3 is an earnings offset
    assert "In alert window" not in _row(body, "NVDA")

    low = body.lower()
    for word in _DELIVERY_WORDS:
        assert word not in low, f"the panel claims delivery: {word!r}"
    assert "not what has been sent" in low


@requires_node
def test_ex_dividend_uses_its_own_single_offset():
    """EX_DIV_OFFSETS is (1,) — a three-day dividend warning is noise, and the
    panel must not mark T-3 on a row that will never fire at T-3."""
    out = _render({
        "as_of": _AS_OF, "checked": 2, "unknown": [],
        "events": [
            _event("ex_dividend", "XIU", "2026-08-01", 3, "XIU goes ex-dividend"),
            _event("ex_dividend", "ZAG", "2026-07-30", 1, "ZAG goes ex-dividend"),
        ],
    })
    body = out["body"]
    assert "In alert window" not in _row(body, "XIU")
    assert "In alert window" in _row(body, "ZAG")


@requires_node
def test_coverage_reports_names_answered_not_names_swept():
    """`checked` counts names the sweep VISITED. Quoting it alone credits the
    radar with coverage of the names that produced nothing."""
    out = _render({"as_of": _AS_OF, "checked": 10,
                   "unknown": ["A", "B", "C"],
                   "events": [_event("earnings", "AAPL", "2026-08-10", 12,
                                     "AAPL reports earnings")]})
    coverage = out["coverage"]
    assert "7 of 10" in coverage
    assert "3 with no date available" in coverage


@requires_node
def test_unknowns_are_named_above_a_table_that_does_have_rows():
    """A non-empty table is the easiest place for a coverage gap to disappear."""
    out = _render({"as_of": _AS_OF, "checked": 3, "unknown": ["BN.TO"],
                   "events": [_event("earnings", "AAPL", "2026-08-10", 12,
                                     "AAPL reports earnings")]})
    body = out["body"]
    assert "BN.TO" in body
    assert "returned no earnings date" in body
    assert "<tbody>" in body


@requires_node
def test_a_name_can_be_unknown_and_still_have_a_row():
    """Found on the demo book's first live read, and the reason the notice was
    reworded. build_event_radar appends to `unknown` when the EARNINGS date is
    unusable, then checks ex-dividend regardless — so MSFT sat in `unknown`
    while owning an ex-dividend row, under a notice claiming no row covered it.
    """
    out = _render({
        "as_of": _AS_OF, "checked": 3, "unknown": ["MSFT", "VTI"],
        "events": [_event("ex_dividend", "MSFT", "2026-08-19", 21,
                          "MSFT goes ex-dividend")],
    })
    body = out["body"]
    assert "no row below covers them" not in body.lower()
    assert "MSFT appears below for a different event kind" in body
    assert "says nothing about their earnings" in body
    # VTI has no row, so it must not be described as having one.
    assert "VTI appears below" not in body


@requires_node
def test_the_header_reports_how_old_the_cached_answer_is():
    """as_of is the BUILD time (5.8), so a cached radar reports its true age
    instead of restarting the clock on read."""
    from datetime import datetime, timedelta
    stamp = (datetime.now() - timedelta(hours=5)).isoformat(timespec="seconds")
    out = _render({"as_of": stamp, "checked": 1, "unknown": [],
                   "events": [_event("earnings", "AAPL", "2026-08-10", 12,
                                     "AAPL reports earnings")]})
    assert "built 5h ago" in out["coverage"]


@requires_node
def test_today_is_rendered_as_today_not_as_zero_days():
    out = _render({"as_of": _AS_OF, "checked": 1, "unknown": [],
                   "events": [_event("earnings", "TSLA", "2026-07-29", 0,
                                     "TSLA reports earnings")]})
    row = _row(out["body"], "TSLA")
    assert "Today" in row
    assert "0 days" not in row
