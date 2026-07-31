"""
6.4 — the cross-specialist findings ledger.

Specialists in this graph hand structured facts to each other by printing them
into a message and having the next node dig them back out with a regex. The
canonical instance, and the one this module was written to delete, is
DeepReasoning's risk pre-screen:

    if 'top_picks' in msg_content and 'score' in msg_content:
        for match in re.findall(r"'symbol':\\s*'([A-Z]{1,5})'", msg_content):

That is a Python ``repr`` being parsed. It fails in four ways, and every one of
them fails SILENTLY — no tickers found means the headwind pre-screen simply does
not run, and nothing says so:

  1. ``[A-Z]{1,5}`` cannot match ``SHOP.TO`` or ``BRK.B``, so the whole Canadian
     and share-class side of the book is invisible to it.
  2. It depends on the repr using single quotes. Serialize that payload as JSON
     anywhere upstream and the pattern matches nothing, forever.
  3. The gate is a substring test on prose — ``'top_picks' in msg_content`` is
     true of a message that merely mentions the phrase.
  4. It reads the message TEXT, so a fact the producer knows precisely gets
     round-tripped through formatting and recovered approximately.

The fix is not a better regex. A producer that already holds the structured
object should publish the object.

**Turn-stamping is not optional here.** ``data_context`` has no state reducer, so
an entry written on an earlier turn is indistinguishable by content from one
written on this one — the same trap 2.3's evidence union hit, where the deep
path's publication kept losing to a stale key. Every finding carries the turn
that produced it, and `read_findings` filters to the current turn by default. A
reader that wants history has to ask for it in as many words.

**An empty ledger is a real answer and a different one from a missing ledger.**
`read_findings` returns a list; `findings_status` says which of the two you have.
A consumer that treats "no findings" as "nothing found" when the truth is "no
producer ran" is the failure this project keeps rediscovering, and the regex
above is an instance of it.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

_KEY = "findings"

# The kinds a finding may carry. A closed set, because an open one becomes a
# second regex problem: consumers would go back to substring-matching `kind`.
KINDS = (
    "candidate",       # a name a producer is putting forward (scanner picks)
    "risk_flag",       # a named hazard attached to symbols
    "verification",    # a holdings/data check with a pass/fail
    "constraint",      # an IPS or profile limit that bears on the turn
    "observation",     # a fact worth carrying that is none of the above
)

# How many findings one turn may hold. A cap rather than unbounded growth: this
# rides in graph state, gets checkpointed, and a scan that published per-symbol
# findings unbounded would bloat every later turn's payload.
MAX_FINDINGS = 200


def make_finding(kind: str, source: str, summary: str,
                 symbols: list[str] | None = None,
                 payload: dict[str, Any] | None = None) -> dict[str, Any]:
    """One structured fact from one producer.

    `summary` is for a human or an LLM reading the ledger; `payload` is for code.
    Both are required to be present rather than one derived from the other,
    because deriving the summary from the payload is how a consumer ends up
    parsing the summary again.
    """
    kind = str(kind or "observation").strip().lower()
    if kind not in KINDS:
        kind = "observation"
    return {
        "kind": kind,
        "source": str(source or "unknown").strip(),
        "summary": str(summary or "").strip(),
        # Preserved verbatim — NOT uppercased or suffix-stripped. A producer that
        # knows the symbol as "SHOP.TO" knows something the normalizer would
        # throw away, and the consumer can normalize for its own comparison.
        "symbols": [str(s).strip() for s in (symbols or []) if str(s).strip()],
        "payload": payload if isinstance(payload, dict) else {},
        "at": datetime.now().isoformat(timespec="seconds"),
    }


def publish_findings(data_context: dict[str, Any] | None,
                     findings: list[dict[str, Any]],
                     turn_key: str) -> dict[str, Any]:
    """Return a COPY of `data_context` with `findings` appended for this turn.

    A copy, because `data_context` has no reducer: mutating the incoming dict
    would write into state the graph may reuse, and returning a fresh dict is how
    every other publisher in this codebase does it.

    Findings from OTHER turns are kept — a consumer filters by turn, and dropping
    them here would make the ledger unable to answer anything about the
    conversation. The cap trims oldest-first.
    """
    ctx = {**(data_context or {})}
    existing = list(ctx.get(_KEY) or [])
    stamped = [{**f, "turn_key": turn_key} for f in findings if isinstance(f, dict)]
    merged = existing + stamped
    if len(merged) > MAX_FINDINGS:
        merged = merged[-MAX_FINDINGS:]
    ctx[_KEY] = merged
    return ctx


def read_findings(state_or_context: dict[str, Any] | None,
                  turn_key: str | None = None,
                  kind: str | None = None,
                  source: str | None = None,
                  all_turns: bool = False) -> list[dict[str, Any]]:
    """Findings from THIS turn, optionally filtered by kind and source.

    Accepts either the graph state or a bare `data_context`, because callers hold
    one or the other and making them unwrap it is how a reader ends up looking in
    the wrong place and concluding there is nothing there.

    `turn_key` of "" never matches anything. That is deliberate and matches
    `current_turn_key`'s own contract: an unidentifiable turn falls back to the
    conservative path, which here means reading no findings rather than reading
    someone else's.
    """
    ctx = state_or_context or {}
    # A state dict carries data_context; a data_context carries findings directly.
    if _KEY not in ctx and isinstance(ctx.get("data_context"), dict):
        ctx = ctx["data_context"]
    rows = [r for r in (ctx.get(_KEY) or []) if isinstance(r, dict)]

    if not all_turns:
        rows = [r for r in rows if r.get("turn_key") == turn_key] if turn_key else []
    if kind:
        rows = [r for r in rows if r.get("kind") == kind]
    if source:
        rows = [r for r in rows if r.get("source") == source]
    return rows


def findings_symbols(findings: list[dict[str, Any]], limit: int | None = None) -> list[str]:
    """De-duplicated symbols across findings, in first-seen order.

    Order is preserved because producers rank: the scanner's first pick is its
    best one, and a set would throw that away right before a caller takes the
    top five.
    """
    seen: list[str] = []
    for f in findings:
        for symbol in f.get("symbols") or []:
            if symbol not in seen:
                seen.append(symbol)
    return seen[:limit] if limit else seen


# ---------------------------------------------------------------------------
# Producers
# ---------------------------------------------------------------------------
# Declared in one place so "who publishes findings" is answerable by reading a
# dict rather than by grepping the node files. Each entry maps a tool name to a
# function taking that tool's RAW return value — the dict, before it is
# stringified into a ToolMessage, which is the whole point.
def _from_opportunity_scan(result: dict[str, Any]) -> list[dict[str, Any]]:
    """Scanner picks, with the symbols the scanner actually chose.

    Replaces `re.findall(r"'symbol':\\s*'([A-Z]{1,5})'", ...)` over the repr.
    Note what the regex could not do and this does: `SHOP.TO` and `BRK.B`
    survive, `conviction` and `score` come across as numbers rather than being
    dropped, and a scan that genuinely picked nothing is distinguishable from a
    scan whose output could not be parsed.
    """
    picks = result.get("top_picks")
    if not isinstance(picks, list):
        return []

    out = []
    for p in picks:
        if not isinstance(p, dict) or not p.get("symbol"):
            continue
        out.append(make_finding(
            kind="candidate",
            source="OpportunityScanner",
            summary=(f"{p.get('symbol')} — score {p.get('score')}, "
                     f"conviction {p.get('conviction') or 'n/a'}"
                     + (f", theme {p['theme']}" if p.get("theme") else "")),
            symbols=[p["symbol"]],
            payload={k: p.get(k) for k in
                     ("symbol", "price", "score", "conviction", "theme",
                      "theme_cycle_stage", "entry_stage", "risk_flag", "stop")
                     if k in p},
        ))

    # A risk_flag the scanner already surfaced is a finding in its own right —
    # DeepReasoning is instructed to address every one of them, and making it
    # re-read the pick list to find them is how one gets missed.
    for p in picks:
        if isinstance(p, dict) and p.get("risk_flag") and p.get("symbol"):
            out.append(make_finding(
                kind="risk_flag", source="OpportunityScanner",
                summary=f"{p['symbol']}: {p['risk_flag']}",
                symbols=[p["symbol"]], payload={"risk_flag": p["risk_flag"]},
            ))
    return out


def _from_holdings_verification(result: dict[str, Any]) -> list[dict[str, Any]]:
    """Which named tickers are actually held. A held/not-held answer is the fact
    most often re-derived from prose downstream, and getting it wrong is the
    hallucination class this graph has the most history with."""
    held = result.get("held") or result.get("verified_holdings") or []
    missing = result.get("not_held") or result.get("missing") or []
    out = []
    if isinstance(held, list) and held:
        out.append(make_finding(
            "verification", "PortfolioVerification",
            f"{len(held)} ticker(s) verified as held",
            symbols=[str(h.get("symbol") if isinstance(h, dict) else h) for h in held],
            payload={"held": True},
        ))
    if isinstance(missing, list) and missing:
        out.append(make_finding(
            "verification", "PortfolioVerification",
            f"{len(missing)} ticker(s) NOT held",
            symbols=[str(m.get("symbol") if isinstance(m, dict) else m) for m in missing],
            payload={"held": False},
        ))
    return out


TOOL_EXTRACTORS = {
    "scan_opportunities": _from_opportunity_scan,
    "scan_sector_opportunities": _from_opportunity_scan,
    "scan_guru_picks": _from_opportunity_scan,
    "verify_portfolio_holdings": _from_holdings_verification,
}


def extract_tool_findings(tool_name: str, result: Any) -> list[dict[str, Any]]:
    """Structured findings from one tool's RAW result. Never raises.

    Never raises because this runs inside the tool-recording path: a producer
    that threw while publishing a finding would take down the tool result it was
    describing, which is a strictly worse outcome than the regex it replaces.
    """
    extractor = TOOL_EXTRACTORS.get(str(tool_name or "").strip())
    if not extractor or not isinstance(result, dict):
        return []
    try:
        return extractor(result)
    except Exception:  # noqa: BLE001 — see docstring
        return []


def findings_status(state_or_context: dict[str, Any] | None,
                    turn_key: str | None = None) -> dict[str, Any]:
    """Whether the ledger is empty, and which kind of empty.

    `no_producer` and `empty` are the distinction this exists for. The first says
    nothing published on this turn — a specialist did not run, or ran and did not
    reach its publish point. The second says a producer ran and had nothing to
    say. A consumer that renders both as "no risks found" is asserting a clean
    bill of health it never received.
    """
    all_rows = read_findings(state_or_context, all_turns=True)
    turn_rows = read_findings(state_or_context, turn_key=turn_key)
    if turn_rows:
        return {"status": "ready", "count": len(turn_rows),
                "sources": sorted({r.get("source", "") for r in turn_rows})}
    return {
        "status": "empty" if all_rows else "no_producer",
        "count": 0,
        "sources": [],
        "note": ("Findings exist from other turns but none from this one."
                 if all_rows else
                 "No specialist has published a finding. This is not a statement "
                 "that nothing was found."),
    }
