"""
Deterministic IPS compliance pre-check (Advisor Roadmap Theme 2.2).

Numerically checks trades PROPOSED IN A DRAFT against the profile's IPS
constraints — position caps, sector caps, dollar-at-risk — so the Risk Judge
confirms a computed pass/fail table instead of estimating concentration and
sizing itself. This is the mandatory gate of arc #2 ("the advisor's call"):
no decision proposal (3.8) ships without clearing it, and 4.7's
superficial-loss / asset-location rule layer slots in here when it lands.

Constraints: read from the profile's own `user_memory.json` under
`risk_constraints` — see load_ips_constraints for why there are no house
defaults. `execution_readiness` is the companion to that decision and the seam
every proposal surface reports through (this gate, 4.4's optimizer, and 3.8's
decision proposals when they land): no cap is still no cap, but an axis nobody
has been ASKED about is now distinguishable from one the user deliberately left
open, and only the second makes a sized proposal execution-ready. It blocks
nothing and authors nothing — it reports.

All stated limits are percent-of-portfolio in the BASE currency. A
stated dollar size that carries an explicit currency code ("$12,500.00 USD"
for a CAD profile) is converted to base before any cap is applied — comparing
a foreign-currency figure against a base-currency cap silently mis-states
every check by the FX rate, which for a CAD/USD profile is ~40%. An unlabeled
figure is still assumed to be base-currency.

Extraction is deliberately NARROW (same philosophy as the grounding
pre-audits in agent/nodes/risk_manager.py): only an explicit buy-side verb
near a ticker, with negation ("do not buy") and third-party ("insider
buying") guards, counts as a proposed trade; only an explicit size ("$5,000",
"$5k", "3% of your portfolio", "40 shares", "Total Investment: $12,500")
makes it numerically checkable. Evidence is read from the ticker's own line
PLUS the lines structurally subordinate to it (see _evidence_window) — drafts
state a pick as a heading with its sizing block indented underneath, so a
line-only window abstains on trades that are in fact fully specified.
Anything else becomes a NOT_EVALUATED row — which the judge maps onto Rule 3
(MAGNITUDE MISS) for tactical trades — never a computed FAIL. Sell/trim
proposals are out of scope for v1: not-held sells are grounding check (a),
and sell-side tax rules (superficial loss) belong to Theme 4.7's rule layer.

Cost: zero portfolio/sector/network work unless a proposed buy is detected.
Sector math reuses check_portfolio_allocation's decomposition stack — the
full-portfolio map runs cache-only (allow_network=False), the single
candidate may resolve once over the network and lands in the daily cache.
Never raises; every failure degrades to an empty result or NOT_EVALUATED.
"""
import re
from typing import Any

# The four caps this module can enforce. Deliberately keys only — no values.
# See load_ips_constraints for why an unstated cap has no default.
_CONSTRAINT_KEYS = (
    "max_position_pct",        # single-name cap, % of portfolio
    "max_fund_position_pct",   # diversified ETF/fund cap, % of portfolio
    "max_sector_pct",          # true sector exposure cap (post fund decomposition)
    "max_risk_per_trade_pct",  # dollar-at-risk cap
)

# Where set_risk_constraints records "I mean these to be unlimited". See
# execution_readiness for why an unstated cap and a deliberately-open axis have
# to be distinguishable.
_ACK_KEY = "unconstrained_ack"

# Plain names for the four axes, for the one line a human reads. Field keys are
# fine in a payload and wrong in a sentence.
_AXIS_LABELS = {
    "max_position_pct": "single-name size",
    "max_fund_position_pct": "fund/ETF size",
    "max_sector_pct": "sector exposure",
    "max_risk_per_trade_pct": "per-trade risk",
}

_MAX_TRADES = 4          # bound worst-case lookups per RiskManager pass
_BREACH_EPSILON = 0.05   # percentage points; ignore float noise at the cap
_MAX_BLOCK_LINES = 12    # how far a ticker's evidence block may run past its line
_MAX_BLOCK_CHARS = 900
_UNRESOLVED_PASS_LIMIT = 0.15  # >15% unclassified sector mass -> a computed PASS is unreliable

# Sector buckets that are not real sector exposure — never cap-checked.
_NON_SECTOR_BUCKETS = {"Cash", "Diversified Fund", "Unclassified Fund", "Unknown"}

# Classification sources (check_portfolio_allocation holding_details) that mean
# the candidate is a diversified fund rather than a single name.
_FUND_SOURCES = {"Fund Decomposition DB", "FMP Decomposition", "Knowledge Graph", "Cache (Decomposed)"}

# --- proposed-trade extraction ------------------------------------------------

_BUY_VERB_RE = re.compile(
    r"\b(?:buy(?:ing)?|add(?:ing)?|accumulat(?:e|ing)|initiat(?:e|ing)|"
    r"start(?:ing)?(?:\s+a)?|deploy(?:ing)?|allocat(?:e|ing)|invest(?:ing)?)\b",
    re.IGNORECASE,
)
# Mirrors risk_manager's negation guard: a preceding keep/negation cue inverts the verb.
# Also covers avoid-list vocabulary ("Excluded: X and Y") — a draft's
# do-NOT-buy roster is the one place a false positive is guaranteed to invert the
# advice's actual meaning.
_BUY_NEGATION_RE = re.compile(
    r"\b(?:not|never|no|avoid(?:s|ing)?|without|instead|rather|wait|hold\s+off|"
    r"exclud(?:e|ed|ing)|omit(?:ted|ting)?|skip(?:ped|ping)?|pass\s+on|steer\s+clear|"
    r"don'?t|doesn'?t|didn'?t|won'?t|wouldn'?t|shouldn'?t|can'?t|cannot|resist)\b",
    re.IGNORECASE,
)
# CURRENT-POSITION STATUS, not a negated buy. Every per-ticker sizing line the
# advisor writes for a new name is tagged "TICKER (Not Held): Buy: $X", and the
# bare \bnot\b above read that annotation as "do not buy" — silently dropping the
# trade from the audit entirely (strictly worse than a size miss: an unaudited
# trade reads as no trade). It only survived on the live draft by luck, because
# ~25 chars of other text pushed the tag out of the negation window. Blanked
# before the negation test, so real negations in the same span still fire.
_POSITION_STATUS_RE = re.compile(
    r"\(?\s*\b(?:not\s+(?:currently\s+)?held|no\s+(?:current\s+)?position|"
    r"currently\s+unheld|unheld|new\s+position|held)\b\s*\)?",
    re.IGNORECASE,
)
# A buy verb whose DESTINATION is stated and is not a security — "deploy
# elsewhere", "redeploy into cash", "allocate to cash reserves". The verb is
# real, but it points away from the ticker that happens to share the window.
# This is what turned a pure trim instruction ("...reduce redundant exposure to
# AAPL and MSFT, locking in those gains to deploy elsewhere or hold in cash")
# into a proposed BUY of MSFT with no size — which the judge then read as a
# Rule 3 MAGNITUDE MISS on a trade the draft never proposed.
_BUY_DESTINATION_TRAIL_RE = re.compile(
    r"^\s*(?:elsewhere|somewhere\s+else|"
    r"(?:in|into|to|toward|towards)\s+(?:your\s+|the\s+)?cash\b|"
    r"(?:in|into|to)\s+(?:your\s+|the\s+)?(?:cash\s+)?(?:reserves?|sidelines?|"
    r"money\s+market|dry\s+powder))",
    re.IGNORECASE,
)
# The ticker is being used as a BENCHMARK / market-condition yardstick, not named
# as a trade target: "if SPY drops another 5%", "a deeper >5% SPY correction",
# "+4.3pp vs SPY". Index proxies appear constantly in the macro framing around a
# dip plan, and that framing routinely shares a block with a real buy verb
# ("deploying Tranche 1 ... for a deeper >5% SPY correction"), which manufactured
# a phantom SPY trade with no size and dragged the sizing audit down. Checked per
# MENTION, so a genuine "buy SPY $5,000" elsewhere in the draft still registers.
_BENCHMARK_TRAIL_RE = re.compile(
    r"^\s*(?:correction|drawdown|selloff|sell-off|pullback|dip|rally|decline|"
    r"drops?|falls?|breadth|weakness|level|close|volatility)\b",
    re.IGNORECASE,
)
_BENCHMARK_LEAD_RE = re.compile(
    r"\b(?:vs\.?|versus|against|beat(?:s|ing)?|track(?:s|ing)?|relative\s+to|"
    r"out\s?perform(?:s|ing|ed)?|under\s?perform(?:s|ing|ed)?)\s+$",
    re.IGNORECASE,
)
# "insider buying", "institutional accumulation" — third-party observations, not instructions.
_THIRD_PARTY_BUYER_RE = re.compile(
    r"\b(?:insider|insiders|executive|executives|officer|officers|management|"
    r"director|directors|founder|founders|chairman|board|"
    r"institution|institutions|institutional|hedge\s+fund|fund manager)\b",
    re.IGNORECASE,
)

_DOLLAR_SIZE_RE = re.compile(r"\$\s*([\d][\d,]*(?:\.\d+)?)\s*([kK])?\b")
# A dollar figure right after a price cue is a PRICE (entry/stop/target), not a size.
_PRICE_CUE_BEFORE_RE = re.compile(
    r"(?:\bat|\bnear|\baround|\bentry|\bstop(?:\s*-?\s*loss)?|\btarget|\bprice|\btrading)\s*:?\s*$",
    re.IGNORECASE,
)
# A price RANGE ("Entry $200-$210", "$325 to $335"): only the FIRST endpoint sits
# behind a price cue, so the second one used to fall through _PRICE_CUE_BEFORE_RE
# and get read as a position size. Both endpoints are prices. "and" is deliberately
# NOT a separator here — "buy $5,000 of NVDA and $3,000 of AMD" is two sizes.
_PRICE_RANGE_RE = re.compile(
    r"\$\s*([\d][\d,]*(?:\.\d+)?)\s*(?:[-–—]|\bto\b)\s*\$\s*([\d][\d,]*(?:\.\d+)?)",
    re.IGNORECASE,
)
# Size detection is an ALLOWLIST, not "any dollar without a price cue". A trade
# table's numbers are overwhelmingly prices; requiring a positive size cue means an
# unrecognized format degrades to "no size stated" (NOT_EVALUATED) instead of
# inventing one. The verb must sit immediately before the figure — "Buy NVDA at
# $200" is a price, only "buy $5,000" is a size.
# The optional [:=] covers the verb used as a LABEL ("Buy: $1,500 CAD",
# "Add: $1,500"), which the sizing block of a per-ticker summary line uses
# constantly. Without it the colon broke adjacency, the figure matched neither
# this nor _SIZE_LABEL_BEFORE_RE (whose allowlist is nouns, not verbs), a fully
# sized draft read as "no size stated", and the judge fired a Rule 3 MAGNITUDE
# MISS on it — the same false negative the label form below was added to fix.
# Adjacency is still required, so "Buy NVDA at $200" stays a price.
_SIZE_VERB_BEFORE_RE = re.compile(
    r"\b(?:buy|buying|add|adding|deploy|deploying|allocat(?:e|ing)|"
    r"invest(?:ing)?|put|purchase|purchasing)\s*[:=]?\s*$",
    re.IGNORECASE,
)
_SIZE_NOUN_AFTER_RE = re.compile(
    r"^\s*(?:of|into|in|to)\b|^\s*(?:position|stake|tranche|allocation|worth)\b",
    re.IGNORECASE,
)
# Label-form sizes ("Total Investment: $12,500.00", "Proposed Size: $8,000").
# The verb/noun cues above only recognize prose ("buy $5,000 of NVDA"), but a
# draft's sizing block is a labelled table, so a fully-sized trade read as "no
# size stated" and the judge fired a Rule 3 MAGNITUDE MISS on it. The separator
# is mandatory: it is what distinguishes a label from a sentence that merely
# contains the word.
_SIZE_LABEL_BEFORE_RE = re.compile(
    r"\b(?:(?:position|proposed|trade|target|total)\s+)?"
    r"(?:size|investment|notional|position\s+value|capital\s+deployed|amount)"
    r"\s*(?:\([^)]{0,20}\))?\s*[:=]\s*$",
    re.IGNORECASE,
)
# A currency code immediately after a dollar figure makes that figure's currency
# explicit. Deliberately adjacent-only — a code elsewhere in the window belongs
# to some other number (drafts routinely quote both "$X USD" and "(≈$Y CAD)").
_SIZE_CURRENCY_AFTER_RE = re.compile(r"^\s*(CAD|USD|EUR|GBP|AUD|JPY)\b", re.IGNORECASE)
# Quote/holding currencies are not limited to the six supported *base* currencies
# (a portfolio can hold HKD or CHF listings), so those are shape-checked instead.
_ISO_CURRENCY_RE = re.compile(r"[A-Z]{3}")
_PCT_SIZE_RE = re.compile(r"([\d]+(?:\.\d+)?)\s*%")
# A percentage is a PROPOSED SIZE only in a partitive ("5% of your portfolio"),
# an explicit position noun ("a 3% position"), or straight after a sizing verb
# ("allocate 5% to NVDA"). A bare "portfolio" anywhere nearby is NOT enough, and
# that laxity is what produced the worst false FAIL this module has emitted: in
# "ABC provides ballast to your 30% tech-heavy portfolio", the 30% is the
# portfolio's CURRENT tech weight, but the old cue matched the trailing
# "portfolio" and sized the trade at 30% = $150,000. The judge then reported a
# position-cap breach and a max-risk breach on a draft proposing 3 shares, and
# scored the compliant revision 2/10. A percentage describing what the user
# already holds must never become a percentage they are being told to buy.
_PCT_PARTITIVE_AFTER_RE = re.compile(
    r"\s*of\s+(?:your|the|our|my|total|net)?\s*"
    r"(?:portfolio|book|capital|assets|holdings|equity|net\s+worth)\b",
    re.IGNORECASE,
)
_PCT_SIZE_NOUN_AFTER_RE = re.compile(r"\s*(?:position|allocation|weight|stake|sleeve)\b", re.IGNORECASE)
# Must end immediately before the figure: "allocate 5%" sizes, "to your 26.5%" describes.
_PCT_SIZE_VERB_BEFORE_RE = re.compile(
    r"\b(?:allocat(?:e|ing)|deploy(?:ing)?|siz(?:e|ed|ing)|invest(?:ing)?|"
    r"buy(?:ing)?|add(?:ing)?)\s+(?:up\s+to\s+)?$",
    re.IGNORECASE,
)
# A % that is a stop distance / return figure, not a position size.
_PCT_NOT_SIZE_RE = re.compile(r"\b(?:stop|yield|return|gain|drop|down|up|below|above|upside|downside)\b", re.IGNORECASE)
_SHARES_SIZE_RE = re.compile(r"\b([\d][\d,]*)\s*shares?\b", re.IGNORECASE)
_STOP_RE = re.compile(r"\bstop(?:\s*-?\s*loss)?\b\D{0,15}\$\s*([\d][\d,]*(?:\.\d+)?)", re.IGNORECASE)
_ENTRY_RE = re.compile(r"\b(?:at|around|near|entry(?:\s+at)?)\s+\$\s*([\d][\d,]*(?:\.\d+)?)", re.IGNORECASE)

# Fallback candidate extraction for standalone use (the RiskManager wiring
# passes its own shared _extract_candidate_tickers set instead).
_TICKER_FALLBACK_RE = re.compile(r"\b[A-Z]{2,5}(?:\.[A-Z]{1,3})?\b")
_FALLBACK_STOPWORDS = {
    "BUY", "SELL", "HOLD", "TRIM", "ADD", "EXIT", "STOP", "RISK", "CASH", "ETF", "ETFS",
    "IPO", "IPS", "USD", "CAD", "EUR", "GBP", "AUD", "JPY", "CEO", "CFO", "CTO", "FOMC",
    "GDP", "CPI", "EPS", "YTD", "ATH", "DCA", "RRSP", "TFSA", "ATR", "SMA", "EMA", "PASS",
    "FAIL", "NOTE", "WATCH", "AVOID", "WAIT",
}


def _coerce_cap(value: Any) -> float | None:
    """A cap is a positive number or it is not a cap; junk reads as unstated.

    Never coerce to a fallback figure: a malformed entry that silently became
    2.0 would be indistinguishable from a user who actually wrote 2.0.
    """
    try:
        cap = float(value)
    except (TypeError, ValueError):
        return None
    return cap if cap > 0 else None


def load_ips_constraints() -> dict[str, Any]:
    """The user's OWN risk limits, from `risk_constraints` in user_memory.json.

    There are no default caps, by design. A limit the user never stated comes
    back as None and its check is skipped outright — silence means the user
    accepted unbounded risk on that axis, NOT that a prudent house default
    applies. The previous 2%/10%/25%/30% defaults were the bug: nothing in any
    profile ever set them, yet the judge enforced them and cited them back as
    "your 2% limit" / "your 10% concentration cap", which is precisely the
    invented-rule attribution RISK_RULES_JUDGE rule 8 exists to prevent. The
    profile is the only authority on the user's risk budget.

    Never raises.
    """
    constraints: dict[str, Any] = dict.fromkeys(_CONSTRAINT_KEYS)
    constraints["enabled"] = True
    constraints["restricted_symbols"] = []
    constraints[_ACK_KEY] = None
    try:
        stated = _load_memory().get("risk_constraints")
    except Exception:
        return constraints
    if not isinstance(stated, dict):
        return constraints
    for key in _CONSTRAINT_KEYS:
        if key in stated:
            constraints[key] = _coerce_cap(stated[key])
    if "enabled" in stated:
        constraints["enabled"] = bool(stated["enabled"])
    symbols = stated.get("restricted_symbols")
    if isinstance(symbols, list):
        constraints["restricted_symbols"] = [str(s) for s in symbols if s]
    ack = stated.get(_ACK_KEY)
    if isinstance(ack, dict):
        constraints[_ACK_KEY] = ack
    return constraints


def stated_caps(constraints: dict[str, Any] | None = None) -> dict[str, float]:
    """Only the caps the user actually set — the seam prompt builders read.

    Prompt text must be derived from this, never from a literal, so the rules
    the judge enforces and the rules the profile states cannot drift apart.
    """
    if constraints is None:
        constraints = load_ips_constraints()
    return {
        key: float(constraints[key])
        for key in _CONSTRAINT_KEYS
        if isinstance(constraints.get(key), (int, float))
    }


def execution_readiness(constraints: dict[str, Any] | None = None) -> dict[str, Any]:
    """Whether a sized proposal can be called execution-ready on this profile.

    THE DISTINCTION THIS EXISTS TO DRAW, and the reason it is not just
    ``bool(stated_caps())``: an axis with no cap can mean two opposite things,
    and until now they were the same silence. Either the user has decided they
    want no limit there — a real answer this app is bound to respect, per the
    standing decision that unstated means unconstrained — or nobody has ever put
    the question to them, which is what shipped for months while the gate this
    module implements had nothing to enforce. The first is a finished profile.
    The second is an unfinished one, and a proposal sized against it has not
    actually cleared anything.

    So the gate is NOT "are there caps". It is "is every open axis open on
    purpose", answered from ``unconstrained_ack`` — which records the axes the
    user confirmed, never a bare yes, so a cap that is later deleted drops out
    of the confirmed set and becomes an open question again.

    This authors no cap and blocks nothing. Callers report the result; the
    proposal is still produced, still correct, and still says what it did.
    Returns ``{execution_ready, stated, unconstrained_by_choice, unanswered,
    acknowledged_at, note}`` — ``note`` being the one shared sentence, carried
    with the flag so no surface has to compose its own. Never raises.
    """
    if constraints is None:
        constraints = load_ips_constraints()
    stated = stated_caps(constraints)

    ack = constraints.get(_ACK_KEY)
    ack = ack if isinstance(ack, dict) else {}
    axes = ack.get("axes")
    acknowledged = {str(a) for a in axes} if isinstance(axes, list) else set()

    blank = [key for key in _CONSTRAINT_KEYS if key not in stated]
    by_choice = [key for key in blank if key in acknowledged]
    unanswered = [key for key in blank if key not in acknowledged]
    readiness = {
        "execution_ready": not unanswered,
        "stated": stated,
        "unconstrained_by_choice": by_choice,
        "unanswered": unanswered,
        "acknowledged_at": str(ack.get("acknowledged_at") or "") or None,
    }
    readiness["note"] = not_ready_line(readiness)
    return readiness


# The one sentence every surface uses to say a proposal is not execution-ready.
# Single source on purpose: the pre-check, the optimizer and (when it lands) the
# decision-proposal surface must not each phrase this their own way, or the same
# profile state reads as three different findings. It names the missing ANSWER
# and never a value — the entire point is that the app has no figure to offer.
_NOT_READY_LINE = (
    "NOT EXECUTION-READY: this profile has not stated a limit on {axes}, and has not "
    "confirmed that it means to leave {them} unlimited. Nothing here is a breach and "
    "nothing is capped — the axes are simply unanswered, so a size on them has cleared "
    "no check."
)

# Kept OUT of the sentence above, because one of the surfaces that shows it is
# the page it points at. Consumers that are somewhere else append it.
WHERE_LIMITS_ARE_SET = (
    "The limits are set on the Context page, where leaving one blank on purpose is "
    "also a valid answer."
)


def not_ready_line(readiness: dict[str, Any]) -> str:
    """The shared finding, or "" when the profile is execution-ready."""
    unanswered = list(readiness.get("unanswered") or [])
    if not unanswered:
        return ""
    labels = [_AXIS_LABELS.get(key, key) for key in unanswered]
    if len(labels) == 1:
        axes, them = labels[0], "it"
    else:
        axes, them = ", ".join(labels[:-1]) + " or " + labels[-1], "them"
    return _NOT_READY_LINE.format(axes=axes, them=them)


# Lazy indirections: keep import light and give tests a single seam.
def _load_memory() -> dict[str, Any]:
    from tools.memory import load_memory
    return load_memory()



def _get_decision_context() -> dict[str, Any]:
    from tools.portfolio_csv import get_portfolio_decision_context
    return get_portfolio_decision_context()


def _get_allocation(symbols: list[str], amounts: list[float], allow_network: bool) -> dict[str, Any]:
    from tools.sector_analysis import check_portfolio_allocation
    return check_portfolio_allocation(symbols, amounts, allow_network=allow_network)


def _get_quote_price(symbol: str) -> tuple[float | None, str]:
    """(price, listing currency) for `symbol`; currency "" when unknown.

    The currency travels with the price because the caller compares the result
    against a base-currency portfolio total — see the shares-only block in
    run_ips_precheck.
    """
    from tools.market_data import get_realtime_quote
    quote = get_realtime_quote(symbol) if symbol else None
    if not isinstance(quote, dict):
        return None, ""
    price = quote.get("price")
    price = float(price) if isinstance(price, (int, float)) and price > 0 else None
    currency = str(quote.get("currency") or "").strip().upper()
    # Any ISO-shaped code is worth handing to the FX lookup; it fails closed
    # (rate 0 → abstain) for ones it cannot price, so no allowlist here.
    return price, (currency if _ISO_CURRENCY_RE.fullmatch(currency) else "")


def _get_fx_rate(from_currency: str, to_currency: str) -> float:
    from tools.portfolio_csv import get_exchange_rate
    return float(get_exchange_rate(from_currency, to_currency))


def _is_cash_symbol(symbol: str) -> bool:
    try:
        from tools.sector_analysis import _is_cash
        return bool(_is_cash(symbol))
    except Exception:
        return False


def _coerce_price(value: Any) -> float | None:
    """current_price may arrive numeric or formatted ('$150.00')."""
    if isinstance(value, (int, float)) and value > 0:
        return float(value)
    try:
        num = float(str(value).replace("$", "").replace(",", "").strip())
        return num if num > 0 else None
    except (ValueError, TypeError):
        return None


def _fallback_candidates(text: str) -> set[str]:
    return {
        t for t in _TICKER_FALLBACK_RE.findall(text)
        if t not in _FALLBACK_STOPWORDS and len(t.split(".")[0]) >= 2
    }


def _identified_prices(window: str) -> set[float]:
    """Dollar figures in `window` that are demonstrably PRICES.

    Two sources: a figure sitting behind a price cue ("entry $200", "stop at
    $185.40"), and both endpoints of a price range ("$200-$210"). Used to veto
    size candidates — see _parse_size.
    """
    prices: set[float] = set()

    def _add(raw: str) -> None:
        try:
            value = float(raw.replace(",", ""))
        except ValueError:
            return
        if value > 0:
            prices.add(value)

    for m in _DOLLAR_SIZE_RE.finditer(window):
        if _PRICE_CUE_BEFORE_RE.search(window[max(0, m.start() - 16):m.start()]):
            _add(m.group(1))
    for m in _PRICE_RANGE_RE.finditer(window):
        _add(m.group(1))
        _add(m.group(2))
    return prices


def _evidence_rank(trade: dict[str, Any]) -> tuple[bool, bool, bool]:
    """How well-evidenced a candidate mention is, for picking between mentions of
    the same ticker. Size dominates (it gates every cap check), then stop, then
    entry (together they gate dollar-at-risk)."""
    return (
        trade.get("size_usd") is not None or bool(trade.get("shares")),
        trade.get("stop") is not None,
        trade.get("stated_entry") is not None,
    )


def _line_bounds(text: str, start: int, end: int) -> tuple[int, int]:
    """The line containing [start, end)."""
    line_start = text.rfind("\n", 0, start) + 1
    line_end = text.find("\n", end)
    return line_start, len(text) if line_end == -1 else line_end


# A line CONTINUES the ticker's block when it is blank, indented, or bullet-
# marked. A new top-level line — the next numbered pick, the next section
# heading — ends it. This is the shape drafts actually use: a pick is a heading
# with its sizing bullets underneath.
_BLOCK_CONTINUATION_RE = re.compile(r"^(?:\s*$|\s+\S|\s*[-*•▪–—+]\s)")


def _block_end(text: str, line_end: int) -> int:
    """Extend a ticker's line forward over the lines subordinate to it.

    Bounded by _MAX_BLOCK_LINES/_MAX_BLOCK_CHARS. Cross-ticker contamination is
    deliberately NOT handled here: _evidence_window clamps the span at the next
    candidate ticker, so however far a block runs it can never absorb another
    name's numbers. Keeping that invariant in one place is why this function is
    purely structural.
    """
    end = line_end
    for _ in range(_MAX_BLOCK_LINES):
        if end >= len(text) or end - line_end >= _MAX_BLOCK_CHARS:
            break
        nxt = text.find("\n", end + 1)
        if nxt == -1:
            nxt = len(text)
        if not _BLOCK_CONTINUATION_RE.match(text[end + 1:nxt]):
            break
        end = nxt
    return end


def _evidence_window(text: str, ticker: str, match: re.Match, candidates: set[str]) -> str:
    """The slice of `text` that may be cited as evidence for `ticker`'s trade.

    Anchored to the ticker's own LINE plus the lines structurally subordinate to
    it (_block_end), then clamped at any other candidate ticker inside that span.
    The clamp is the load-bearing part: a fixed ±character window silently spans
    neighbouring rows of a per-ticker shopping list, and since _STOP_RE/_ENTRY_RE
    take the first match in the window, the second name in the list inherits the
    first one's stop and entry. The pre-check's whole value is that its table is
    arithmetic rather than estimate, so a number attributed to the wrong ticker
    is worse than no number at all.

    The line alone is not enough, though: drafts state a pick as a heading and
    its size/stop as bullets underneath, so a line-only window found no size on
    every such trade and abstained — and a NOT_EVALUATED sizing row is read
    downstream as a Rule 3 MAGNITUDE MISS, which turned fully-specified drafts
    into compliance failures.
    """
    lo, hi = _line_bounds(text, match.start(), match.end())
    hi = _block_end(text, hi)
    for other in candidates:
        if not other or other == ticker:
            continue
        for om in re.finditer(r"\b" + re.escape(other) + r"\b", text):
            if om.start() < lo or om.end() > hi:
                continue
            if om.end() <= match.start():
                lo = max(lo, om.end())
            elif om.start() >= match.end():
                hi = min(hi, om.start())
    return text[lo:hi]


def _to_base(dollars: float, stated_currency: str, base_currency: str) -> tuple[float | None, str]:
    """Convert an explicitly-labelled size into base currency.

    Returns (base_dollars_or_None, label). An unlabelled figure, or one already
    in base, passes through. When the rate is unavailable the size becomes None
    (NOT_EVALUATED) rather than being compared as-is: an unconverted foreign
    figure would mis-state every percent-of-portfolio cap by the FX rate, and a
    silently wrong PASS is worse than an honest abstention.
    """
    if not stated_currency or not base_currency or stated_currency == base_currency:
        return dollars, f"${dollars:,.0f}"
    try:
        rate = _get_fx_rate(stated_currency, base_currency)
    except Exception:
        rate = 0.0
    if not rate or rate <= 0:
        return None, f"${dollars:,.0f} {stated_currency} (no {stated_currency}→{base_currency} rate)"
    converted = dollars * rate
    # Base currency leads, native goes in parentheses — the profile's reporting
    # rule, applied to the pre-check's own table so the judge is never reading a
    # foreign headline off a block it is told to treat as fact.
    return converted, f"${converted:,.0f} {base_currency} (draft stated ${dollars:,.0f} {stated_currency})"


def _parse_size(
    window: str, total_value: float, base_currency: str = ""
) -> tuple[float | None, str]:
    """Extract an explicit proposed size from the window, in BASE currency.

    Returns (dollars_or_None, human_label). Preference order: dollar amount,
    then a %-of-portfolio with a portfolio cue, then a share count (converted
    by the caller, so returned here as None with a shares label). A dollar
    figure labelled with a currency code is converted via _to_base.
    """
    prices = _identified_prices(window)
    for m in _DOLLAR_SIZE_RE.finditer(window):
        if _PRICE_CUE_BEFORE_RE.search(window[max(0, m.start() - 16):m.start()]):
            continue  # "$200" after "at"/"stop"/"target" is a price, not a size
        has_verb = bool(_SIZE_VERB_BEFORE_RE.search(window[max(0, m.start() - 24):m.start()]))
        has_noun = bool(_SIZE_NOUN_AFTER_RE.search(window[m.end():m.end() + 24]))
        has_label = bool(_SIZE_LABEL_BEFORE_RE.search(window[max(0, m.start() - 30):m.start()]))
        if not (has_verb or has_noun or has_label):
            continue  # no positive size cue — see _SIZE_VERB_BEFORE_RE
        try:
            dollars = float(m.group(1).replace(",", ""))
            if m.group(2):
                dollars *= 1000
        except ValueError:
            continue
        if dollars <= 0:
            continue
        # A figure already established as a price elsewhere on the line is a price
        # here too, whatever cue precedes it: "Entry $200-$210 ... add $200" reads
        # as "add at $200", not "add $200 of stock".
        if any(abs(dollars - p) <= max(0.01, p * 0.01) for p in prices):
            continue
        cm = _SIZE_CURRENCY_AFTER_RE.match(window[m.end():m.end() + 8])
        return _to_base(dollars, cm.group(1).upper() if cm else "", base_currency.upper())

    for m in _PCT_SIZE_RE.finditer(window):
        try:
            pct = float(m.group(1))
        except ValueError:
            continue
        if not (0 < pct <= 100):
            continue
        # The % must sit in a sizing construction — not merely near the word
        # "portfolio" — and must not be a stop/return figure.
        after = window[m.end():m.end() + 40]
        before = window[max(0, m.start() - 30):m.start()]
        if not (
            _PCT_PARTITIVE_AFTER_RE.match(after)
            or _PCT_SIZE_NOUN_AFTER_RE.match(after)
            or _PCT_SIZE_VERB_BEFORE_RE.search(before)
        ):
            continue
        if _PCT_NOT_SIZE_RE.search(window[max(0, m.start() - 12):m.end() + 12]):
            continue
        if total_value > 0:
            dollars = pct / 100.0 * total_value
            return dollars, f"{pct:g}% of portfolio (≈${dollars:,.0f})"
        return None, f"{pct:g}% of portfolio"

    m = _SHARES_SIZE_RE.search(window)
    if m:
        try:
            shares = float(m.group(1).replace(",", ""))
            if shares > 0:
                return None, f"{shares:g} shares"
        except ValueError:
            pass
    return None, ""


def extract_proposed_trades(
    text: str,
    candidate_tickers: set[str] | None = None,
    total_value: float = 0.0,
    base_currency: str = "",
) -> list[dict[str, Any]]:
    """Buy-side proposed trades stated explicitly in the draft.

    Each trade: {ticker, size_usd (base-currency dollars or None),
    size_label, shares (float or None), stop (float or None),
    stated_entry (float or None), window}. At most one trade per ticker (the
    best-evidenced mention wins — see _evidence_rank), capped at _MAX_TRADES.

    `base_currency` enables conversion of currency-labelled sizes; left empty
    (the default, for callers that have no profile context) every figure is
    taken at face value, which is the pre-2.2 behaviour.
    """
    trades: dict[str, dict[str, Any]] = {}
    try:
        candidates = candidate_tickers if candidate_tickers is not None else _fallback_candidates(text)
        for ticker in sorted(candidates, key=lambda t: (text.find(t), t)):
            for match in re.finditer(r"\b" + re.escape(ticker) + r"\b", text):
                # A benchmark mention is not a trade target — see _BENCHMARK_TRAIL_RE.
                if _BENCHMARK_TRAIL_RE.match(text[match.end():match.end() + 24]):
                    continue
                if _BENCHMARK_LEAD_RE.search(text[max(0, match.start() - 20):match.start()]):
                    continue

                window = _evidence_window(text, ticker, match, candidates)

                verb = None
                for vm in _BUY_VERB_RE.finditer(window):
                    preceding = _POSITION_STATUS_RE.sub(" ", window[max(0, vm.start() - 25):vm.start()])
                    if _BUY_NEGATION_RE.search(preceding):
                        continue
                    if _THIRD_PARTY_BUYER_RE.search(preceding):
                        continue
                    if _BUY_DESTINATION_TRAIL_RE.match(window[vm.end():vm.end() + 32]):
                        continue
                    verb = vm
                    break
                if verb is None:
                    continue

                size_usd, size_label = _parse_size(window, total_value, base_currency)
                shares = None
                sm = _SHARES_SIZE_RE.search(window)
                if sm and size_usd is None:
                    try:
                        shares = float(sm.group(1).replace(",", ""))
                    except ValueError:
                        shares = None

                stop = None
                stop_m = _STOP_RE.search(window)
                if stop_m:
                    try:
                        stop = float(stop_m.group(1).replace(",", ""))
                    except ValueError:
                        stop = None
                stated_entry = None
                for entry_m in _ENTRY_RE.finditer(window):
                    # "stop at $180" also matches the entry pattern — skip
                    # matches whose lead-in is a stop cue.
                    if re.search(r"\bstop(?:\s*-?\s*loss)?\s*$", window[max(0, entry_m.start() - 14):entry_m.start()], re.IGNORECASE):
                        continue
                    try:
                        stated_entry = float(entry_m.group(1).replace(",", ""))
                    except ValueError:
                        stated_entry = None
                    break

                existing = trades.get(ticker)
                candidate_trade = {
                    "ticker": ticker,
                    "size_usd": size_usd,
                    "size_label": size_label,
                    "shares": shares,
                    "stop": stop,
                    "stated_entry": stated_entry,
                    "window": window,
                }
                # Keep the single best-evidenced mention rather than merging fields
                # across mentions: every number in a row must come from the SAME
                # window, or the table starts pairing one line's size with another
                # line's stop — cross-context inference dressed up as arithmetic.
                if existing is None or _evidence_rank(candidate_trade) > _evidence_rank(existing):
                    trades[ticker] = candidate_trade
                if _evidence_rank(candidate_trade) == (True, True, True):
                    break  # fully specified — no later mention can improve on it
            if len(trades) >= _MAX_TRADES:
                break
    except Exception:
        return []
    return list(trades.values())[:_MAX_TRADES]


# --- numeric checks -----------------------------------------------------------


def _held_value(ctx: dict[str, Any], ticker: str) -> float:
    """Current base-currency value of `ticker` (base-symbol tolerant: BCE ↔ BCE.TO)."""
    base = ticker.split(".")[0]
    total = 0.0
    for h in ctx.get("holdings", []):
        sym = str(h.get("symbol") or "").upper()
        if sym == ticker or sym.split(".")[0] == base:
            value = h.get("value_base")
            if isinstance(value, (int, float)):
                total += float(value)
    return total


def _held_price(ctx: dict[str, Any], ticker: str) -> float | None:
    base = ticker.split(".")[0]
    for h in ctx.get("holdings", []):
        sym = str(h.get("symbol") or "").upper()
        if sym == ticker or sym.split(".")[0] == base:
            price = _coerce_price(h.get("current_price"))
            if price:
                return price
    return None


def _held_currency(ctx: dict[str, Any], ticker: str) -> str:
    """Listing currency of a held position, "" when unknown.

    Preferred over the quote's currency: it is already in the decision context
    (no network round-trip) and it is the same figure the portfolio's own
    base-currency conversion was computed from, so sizes derived here stay
    consistent with `value_base`.
    """
    base = ticker.split(".")[0]
    for h in ctx.get("holdings", []):
        sym = str(h.get("symbol") or "").upper()
        if sym == ticker or sym.split(".")[0] == base:
            currency = str(h.get("currency") or "").strip().upper()
            if _ISO_CURRENCY_RE.fullmatch(currency):
                return currency
    return ""


def _cash_total(ctx: dict[str, Any]) -> float:
    total = 0.0
    for h in ctx.get("holdings", []):
        sym = str(h.get("symbol") or "").upper()
        if sym and _is_cash_symbol(sym):
            value = h.get("value_base")
            if isinstance(value, (int, float)):
                total += float(value)
    return total


def _row(trade_label: str, check: str, computed: str, limit: str, verdict: str) -> dict[str, str]:
    return {"trade": trade_label, "check": check, "computed": computed, "limit": limit, "verdict": verdict}


def run_ips_precheck(
    text: str,
    candidate_tickers: set[str] | None = None,
) -> dict[str, Any]:
    """
    Detect proposed buys in `text` and numerically check them against the
    profile's IPS constraints.

    Returns {"trades": [...], "rows": [...], "violations": [...], "block": str,
    "execution_ready": bool, "readiness": {...}}. `violations` holds only
    computed FAILs (safe to merge into the grounding violation gate); `block` is
    the <ips_precheck> prompt section for the judge, "" when there was nothing to
    say. Never raises.

    `execution_ready` is deliberately NOT folded into `violations`: an
    unanswered axis is a gap in the profile, not a fault in the draft, and
    routing it through the violation gate would cap the score of advice that did
    nothing wrong — and hand the judge an absent limit to describe, which is how
    an invented rule gets attributed to the user.
    """
    empty: dict[str, Any] = {
        "trades": [], "rows": [], "violations": [], "block": "",
        "execution_ready": True, "readiness": {},
    }
    try:
        constraints = load_ips_constraints()
        if not constraints.get("enabled", True):
            return empty
        readiness = execution_readiness(constraints)
        empty = {**empty, "execution_ready": readiness["execution_ready"], "readiness": readiness}
        # Nothing to enforce: the profile states no cap and no restricted list.
        # Return before any portfolio/network work rather than emit a table of
        # vacuous rows — _format_block announces itself as "the profile's IPS
        # constraints" and tells the judge a NOT_EVALUATED row is a Rule 3
        # MAGNITUDE MISS, so an empty check still reads as a real one.
        if not stated_caps(constraints) and not constraints.get("restricted_symbols"):
            # ...but if the draft proposes a buy and the profile has never been
            # asked about these axes, the silence itself is the finding. A buy
            # verb is the cheap text test that keeps this off every other turn:
            # no portfolio, no network, no extraction. A profile that answered
            # "unlimited" gets nothing said about it — that is a finished
            # profile, and nagging it would make the confirmation worthless.
            if readiness["execution_ready"] or not _BUY_VERB_RE.search(text or ""):
                return empty
            return {**empty, "block": _format_readiness_block(readiness)}

        ctx = _get_decision_context()
        if not isinstance(ctx, dict) or ctx.get("error"):
            return empty
        total_value = ctx.get("total_value_base")
        if not isinstance(total_value, (int, float)) or total_value <= 0:
            return empty
        total_value = float(total_value)
        base_currency = str(ctx.get("base_currency") or "USD").upper()

        trades = extract_proposed_trades(text, candidate_tickers, total_value, base_currency)
        if not trades:
            return empty

        # None means the user stated no limit on that axis. Every check below is
        # guarded on its own cap and emits NOTHING when unstated — not even a
        # NOT_EVALUATED row, which the judge reads as a Rule 3 MAGNITUDE MISS
        # and would turn an absent limit back into an enforced one.
        max_single = constraints.get("max_position_pct")
        max_fund = constraints.get("max_fund_position_pct")
        max_sector = constraints.get("max_sector_pct")
        max_risk = constraints.get("max_risk_per_trade_pct")
        restricted = {str(s).upper() for s in constraints.get("restricted_symbols", []) or []}

        cash_available = _cash_total(ctx)

        # Shares-only sizes convert to dollars up front (held price, stated
        # entry, else one cached quote) so the sector-map decision below sees
        # every trade that is actually checkable.
        #
        # shares × price lands in the SECURITY's currency, not the profile's:
        # for a CAD profile buying a US listing it is a USD figure, and
        # comparing that to total_value_base understates the position cap,
        # sector cap and dollar-at-risk by the whole FX rate — enough to hand a
        # computed PASS to a draft that actually breaches the cap. So the
        # price's currency has to travel with it, and when it cannot be
        # established the size stays None (NOT_EVALUATED). See _to_base.
        for trade in trades:
            if trade.get("size_usd") is not None or not trade.get("shares"):
                continue
            ticker = trade["ticker"]
            price = trade.get("stated_entry") or _held_price(ctx, ticker)
            # A held position's currency labels a stated entry too — the draft
            # quotes a security in the currency it trades in.
            currency = _held_currency(ctx, ticker)
            if price is None or (not currency and base_currency != "USD"):
                try:
                    quote_price, quote_currency = _get_quote_price(ticker)
                except Exception:
                    quote_price, quote_currency = None, ""
                price = price or quote_price
                currency = currency or quote_currency
            if not price:
                continue

            native = float(trade["shares"]) * price
            if currency and currency != base_currency:
                size, _label = _to_base(native, currency, base_currency)
                if size is None:
                    continue  # no rate — abstain rather than compare unconverted
                # Base currency leads, native in parentheses — same reporting
                # rule _to_base applies, for the same reason: the judge reads
                # this block as fact and must not take a foreign headline off it.
                label = (
                    f"{trade['shares']:g} shares ≈${size:,.0f} {base_currency} "
                    f"(at ${price:,.2f} {currency})"
                )
            elif currency or base_currency == "USD":
                size = native
                label = f"{trade['shares']:g} shares (≈${native:,.0f})"
            else:
                continue  # currency unknown against a non-USD base — abstain
            trade["size_usd"] = size
            trade["size_label"] = label

        # Current sector map: cache-only so the audit path never stalls cold.
        current_sectors: dict[str, float] = {}
        sector_map_ok = False
        sized_trades = [t for t in trades if t.get("size_usd")]
        if sized_trades:
            try:
                symbols = [h["symbol"] for h in ctx.get("holdings", []) if h.get("symbol")]
                amounts = [float(h.get("value_base") or 0.0) for h in ctx.get("holdings", []) if h.get("symbol")]
                if symbols:
                    alloc = _get_allocation(symbols, amounts, allow_network=False)
                    raw = alloc.get("sector_allocation_raw")
                    if isinstance(raw, dict) and raw:
                        current_sectors = {k: float(v) for k, v in raw.items()}
                        sector_map_ok = True
            except Exception:
                sector_map_ok = False

        rows: list[dict[str, str]] = []
        violations: list[str] = []
        trade_labels: list[str] = []

        for trade in trades:
            ticker = trade["ticker"]
            size = trade.get("size_usd")
            label = f"BUY {ticker}" + (f" {trade['size_label']}" if trade.get("size_label") else "")
            trade_labels.append(label)

            # 0. Restricted list (hard IPS rule; row only emitted when it applies).
            if ticker.upper() in restricted or ticker.split(".")[0] in restricted:
                rows.append(_row(label, "restricted list", "symbol is on the IPS restricted list", "no new buys", "FAIL"))
                violations.append(
                    f"IPS Pre-check FAIL: {label} — {ticker} is on the profile's restricted list (no new buys permitted)."
                )

            cash_funded = size is not None and cash_available >= size
            denom = total_value if (size is None or cash_funded) else total_value + size

            # 1. Position cap (single-name vs fund cap by classification).
            current_value = _held_value(ctx, ticker)
            cand_sectors: dict[str, float] | None = None
            is_fund = False
            if size is not None:
                try:
                    cand_alloc = _get_allocation([ticker], [1.0], allow_network=True)
                    raw = cand_alloc.get("sector_allocation_raw")
                    if isinstance(raw, dict) and raw:
                        cand_sectors = {k: float(v) for k, v in raw.items()}
                    details = cand_alloc.get("holding_details") or []
                    source = str(details[0].get("classification_source", "")) if details else ""
                    sector_details = str(details[0].get("sector_details", "")) if details else ""
                    is_fund = source in _FUND_SOURCES or "Fund" in sector_details
                except Exception:
                    cand_sectors = None

                cap = max_fund if is_fund else max_single
                cap_kind = "fund" if is_fund else "single name"
                if cap is not None:
                    post_pct = (current_value + size) / denom * 100.0
                    computed = (
                        f"{current_value / total_value * 100.0:.1f}% now → {post_pct:.1f}% post-trade ({cap_kind}"
                        + (", cash-funded" if cash_funded else ", new money")
                        + ")"
                    )
                    if post_pct > cap + _BREACH_EPSILON:
                        rows.append(_row(label, "position cap", computed, f"≤{cap:g}%", "FAIL"))
                        violations.append(
                            f"IPS Pre-check FAIL: {label} — post-trade position {post_pct:.1f}% exceeds the "
                            f"{cap:g}% {cap_kind} cap (computed deterministically from verified holdings)."
                        )
                    else:
                        rows.append(_row(label, "position cap", computed, f"≤{cap:g}%", "PASS"))
            elif max_single is not None:
                headroom = max(0.0, max_single / 100.0 * total_value - current_value)
                rows.append(_row(
                    label, "position cap",
                    f"size not stated; current {current_value / total_value * 100.0:.1f}%, "
                    f"headroom to {max_single:g}% cap ≈ ${headroom:,.0f} {base_currency}",
                    f"≤{max_single:g}%", "NOT_EVALUATED",
                ))

            # 2. Sector cap (needs a stated cap + size + both sector maps).
            if max_sector is not None and size is not None and sector_map_ok and cand_sectors:
                checkable = {s: w for s, w in cand_sectors.items() if s not in _NON_SECTOR_BUCKETS and w > 0}
                if checkable:
                    share = size / denom
                    worst_sector, worst_post = None, -1.0
                    for sector, weight in checkable.items():
                        if cash_funded:
                            post = current_sectors.get(sector, 0.0) + share * weight
                        else:
                            post = (current_sectors.get(sector, 0.0) * total_value + size * weight) / denom
                        if post > worst_post:
                            worst_sector, worst_post = sector, post
                    unresolved = sum(current_sectors.get(b, 0.0) for b in ("Unknown", "Unclassified Fund"))
                    cur_pct = current_sectors.get(worst_sector, 0.0) * 100.0
                    post_pct = worst_post * 100.0
                    computed = f"{worst_sector} {cur_pct:.1f}% now → {post_pct:.1f}% post-trade"
                    if unresolved > 0:
                        computed += f" ({unresolved * 100.0:.0f}% of portfolio unclassified)"
                    if post_pct > max_sector + _BREACH_EPSILON:
                        rows.append(_row(label, "sector cap", computed, f"≤{max_sector:g}%", "FAIL"))
                        violations.append(
                            f"IPS Pre-check FAIL: {label} — post-trade {worst_sector} exposure {post_pct:.1f}% "
                            f"exceeds the {max_sector:g}% sector cap (fund-decomposed, computed deterministically)."
                        )
                    elif unresolved > _UNRESOLVED_PASS_LIMIT:
                        rows.append(_row(label, "sector cap", computed + " — too much unclassified mass to certify a pass", f"≤{max_sector:g}%", "NOT_EVALUATED"))
                    else:
                        rows.append(_row(label, "sector cap", computed, f"≤{max_sector:g}%", "PASS"))
                else:
                    rows.append(_row(label, "sector cap", "candidate has no classifiable sector exposure", f"≤{max_sector:g}%", "NOT_EVALUATED"))
            elif max_sector is not None and size is not None:
                rows.append(_row(label, "sector cap", "sector map unavailable this pass", f"≤{max_sector:g}%", "NOT_EVALUATED"))
            elif max_sector is not None:
                rows.append(_row(label, "sector cap", "size not stated", f"≤{max_sector:g}%", "NOT_EVALUATED"))

            # 3. Dollar-at-risk vs the user's stated max-risk rule, if they set one
            #    (needs a stated cap + size + stop + entry).
            stop = trade.get("stop")
            if max_risk is not None and size is not None and stop:
                entry = trade.get("stated_entry") or _held_price(ctx, ticker)
                if entry is None:
                    try:
                        entry, _ = _get_quote_price(ticker)
                    except Exception:
                        entry = None
                if entry and 0 < stop < entry:
                    risk_dollars = size * (entry - stop) / entry
                    limit_dollars = max_risk / 100.0 * total_value
                    computed = (
                        f"${risk_dollars:,.0f} at risk (entry ${entry:,.2f}, stop ${stop:,.2f}) "
                        f"= {risk_dollars / total_value * 100.0:.2f}% of portfolio"
                    )
                    if risk_dollars > limit_dollars * (1 + _BREACH_EPSILON / 100.0):
                        rows.append(_row(label, "dollar-at-risk", computed, f"≤{max_risk:g}% (${limit_dollars:,.0f})", "FAIL"))
                        violations.append(
                            f"IPS Pre-check FAIL: {label} — ${risk_dollars:,.0f} at risk exceeds the "
                            f"{max_risk:g}% max-risk rule (${limit_dollars:,.0f} {base_currency})."
                        )
                    else:
                        rows.append(_row(label, "dollar-at-risk", computed, f"≤{max_risk:g}% (${limit_dollars:,.0f})", "PASS"))
                elif entry and stop >= entry:
                    rows.append(_row(label, "dollar-at-risk", f"stated stop ${stop:,.2f} is not below entry ${entry:,.2f} — check the draft", f"≤{max_risk:g}%", "NOT_EVALUATED"))
                else:
                    rows.append(_row(label, "dollar-at-risk", "entry price unavailable", f"≤{max_risk:g}%", "NOT_EVALUATED"))
            elif max_risk is not None:
                missing = "size" if size is None else "stop"
                rows.append(_row(label, "dollar-at-risk", f"no {missing} stated", f"≤{max_risk:g}%", "NOT_EVALUATED"))

            # 4. Account location (tax-shelter context; 4.7's rule layer extends this).
            accounts = sorted({
                str(h.get("account") or "Unknown")
                for h in ctx.get("holdings", [])
                if str(h.get("symbol") or "").upper().split(".")[0] == ticker.split(".")[0]
            })
            if accounts:
                rows.append(_row(label, "account location", f"currently held in: {', '.join(accounts)}", "informational", "INFO"))

        block = _format_block(trade_labels, rows, base_currency, readiness)
        return {
            "trades": trades,
            "rows": rows,
            "violations": violations,
            "block": block,
            "execution_ready": readiness["execution_ready"],
            "readiness": readiness,
        }
    except Exception:
        try:
            from agent.utils import safe_print
            safe_print("⚠️ IPS pre-check error (degraded to no-op)")
        except Exception:
            pass
        # `empty` carries whatever readiness was established before the failure,
        # and execution_ready=True if the failure came first. That default is
        # deliberate: a read that did not complete proves nothing about the
        # profile, and announcing a gap nobody verified is the same invention in
        # the other direction.
        return empty


def _readiness_lines(readiness: dict[str, Any]) -> list[str]:
    """The execution-readiness paragraph, or [] when the profile is ready.

    Written defensively because it is read by the judge, and this codebase has
    already shipped a judge that turned an absent limit into a rule it attributed
    to the user. So: name it as a PROFILE gap, state outright that it is not a
    violation, and forbid the two failure modes explicitly — quoting a number
    that does not exist, and scoring the draft for it.
    """
    line = not_ready_line(readiness)
    if not line:
        return []
    return [
        f"{line} {WHERE_LIMITS_ARE_SET}",
        "This is a state of the PROFILE, not a fault in the advice: do not score it, do not "
        "treat it as a breach, and do not name, infer or illustrate a limit for any axis above — "
        "no figure exists for them and one you supply gets read back later as the user's own rule. "
        "Say only that these limits are unset, that the app has none to apply, and that the "
        "proposal is therefore not execution-ready until the user states them or confirms they "
        "want them unlimited.",
    ]


def _format_readiness_block(readiness: dict[str, Any]) -> str:
    """The readiness note on its own — the profile states nothing to check.

    Deliberately NOT titled as a compliance pre-check: no table was computed and
    no cap was applied, and a block that announced one would be the empty gate
    reading as a real one, which is the whole defect this slice closes.
    """
    return "\n".join(
        ["\n<ips_execution_readiness>", *_readiness_lines(readiness), "</ips_execution_readiness>\n"]
    )


def _format_block(
    trade_labels: list[str],
    rows: list[dict[str, str]],
    base_currency: str,
    readiness: dict[str, Any] | None = None,
) -> str:
    lines = [
        "\n<ips_precheck>",
        "Deterministic IPS compliance pre-check — computed numerically from verified holdings "
        "and the profile's IPS constraints (dollar figures in "
        f"{base_currency}). These numbers are FACTS: CONFIRM the table in your verdict instead of "
        "estimating sizing or concentration yourself.",
        f"Proposed trades detected: {' · '.join(trade_labels)}",
        "| trade | check | computed | limit | verdict |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(f"| {r['trade']} | {r['check']} | {r['computed']} | {r['limit']} | {r['verdict']} |")
    lines.append(
        "A FAIL row is a confirmed Rule 10 (profile/IPS) violation — flag it; it caps the verdict at ≤6/10. "
        "NOT_EVALUATED means the draft omitted the needed number (size/stop): for a tactical single-stock "
        "trade that omission is itself a Rule 3 MAGNITUDE MISS. Do not re-derive or dispute these numbers."
    )
    # The partial-cover case, and the one most likely to mislead: a table full of
    # PASS rows on a profile that only stated two of the four axes reads as a
    # clean bill of health. It is a clean bill on the axes that exist.
    lines.extend(_readiness_lines(readiness or {}))
    lines.append("</ips_precheck>\n")
    return "\n".join(lines)
