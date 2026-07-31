import html
import os
import re
from dataclasses import dataclass, field
from datetime import datetime

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agent.dspy_setup import DSPY_AVAILABLE, configure_dspy

# --- Logging ---
from agent.logger import log_event
from agent.memory import get_user_context_string
from agent.risk_rules import risk_rules_judge
from agent.state import AgentState
from agent.utils import (
   create_agent,
   current_turn_key,
   extract_reasoning_text,
   get_llm,
   get_sonnet_llm,
   has_stream_callback,
   is_cancelled,
   safe_invoke,
   safe_print,
   send_status,
   send_stream,
   send_thinking,
)
from tools.provenance import STATUS_STALE as PROV_STALE
from tools.provenance import STATUS_UNAVAILABLE as PROV_UNAVAILABLE
from tools.provenance import merge_tool_contexts, summarize_tool_context
from tools.watch_conditions import strip_watch_blocks

# Roadmap 2.3: the ceiling a turn can score when its evidence was degraded.
# 7 is deliberate — below a clean pass (8+ is compliant), above the grounding
# cap of 6 that forces CRITICAL_FAIL. Degraded evidence should visibly cost the
# turn without blocking an answer the user still needs.
PROVENANCE_SCORE_CAP = 7

# --- LLM Config ---
# Lazy provider resolution: Anthropic/OpenAI providers use defaults from agent.utils.
MODEL_ID = os.environ.get("AIDLC_MODEL_ID")
REGION = os.environ.get("AWS_REGION", "us-east-1")

# DSPy is provider-agnostic: configure_dspy() builds the LiteLLM-backed LM for
# whichever LLM_PROVIDER is active (bedrock/openai/anthropic/google/azure).
if DSPY_AVAILABLE:
    configure_dspy(MODEL_ID, REGION, error_callback=safe_print)


def _prompt_escape(value) -> str:
    """Escape user/agent supplied text before embedding it inside prompt tags."""
    return html.escape(str(value or ""), quote=False)


def _format_cost_basis(holding: dict) -> str:
    """Render one holding's cost basis / current price / return for the judge.

    Rule 8 orders the judge to verify cost-basis and drawdown claims against this
    brief, so omitting these fields (as this brief did until 2026-07-16) makes every
    TRUE drawdown claim unverifiable and guarantees a false SOURCE FRAUD flag. When
    the cost basis really is missing or zero, say so explicitly rather than staying
    silent — that is the case Rule 8 is legitimately meant to catch.

    Prices here are the holding's NATIVE currency; the caller prints position value in
    the user's base currency on the same line.
    """
    if holding.get("is_cash_or_pension"):
        gain_loss = holding.get("gain_loss")
        return f"cash/pension (no cost basis; return {gain_loss})" if gain_loss else "cash/pension (no cost basis)"

    purchase_price = str(holding.get("purchase_price") or "").strip()
    current_price = str(holding.get("current_price") or "").strip()
    gain_loss = str(holding.get("gain_loss") or "").strip()
    currency = str(holding.get("currency") or "USD").upper()

    # "—" is what an unvalued holding carries: units on record, but no entry price and
    # no total to derive one from. That is the same answer as a missing cost basis and
    # belongs in the same branch — rendering it as "@ — cost" would read as a price.
    if not purchase_price or purchase_price in {"$0.00", "$0", "—"}:
        return "cost basis NOT in portfolio data — drawdown/return claims for this position are unverifiable"

    shares = holding.get("shares")
    shares_text = f"{shares:g} sh @ " if isinstance(shares, (int, float)) else ""
    now_text = f" → {current_price} {currency} now" if current_price else ""
    return_text = f" ({gain_loss})" if gain_loss else ""
    return f"{shares_text}{purchase_price} {currency} cost{now_text}{return_text}"


def _build_portfolio_verification_brief() -> str:
    """Compact current-holdings context for the risk judge."""
    try:
        from tools.portfolio_csv import get_portfolio_decision_context

        context = get_portfolio_decision_context()
        if context.get("error"):
            return f"Portfolio verification unavailable: {context.get('error')}"

        base_currency = str(context.get("base_currency") or "CAD").upper()
        total_value_base = context.get("total_value_base")
        if total_value_base is None:
            total_value_base = context.get("total_value_cad", 0) if base_currency == "CAD" else context.get("total_value_usd", 0)

        lines = [
            f"Profile: {context.get('profile') or 'Unknown'}",
            f"As of: {context.get('as_of') or 'Data Unavailable'}",
            f"Stale snapshot: {context.get('is_stale')}",
            f"Sync errors: {context.get('sync_errors') or 'None'}",
            f"Total value ({base_currency}, the user's base currency — treat as the only headline figure): ${total_value_base:,.2f}",
            # Stated before the holdings list so the judge reads the total as a floor
            # rather than a complete figure. An advisor that correctly notes a pension
            # is missing from the total is agreeing with this line, not contradicting it.
            *([f"INCOMPLETE TOTAL — {context['unvalued_notice']} The total above therefore "
               "understates the book. An advisor saying so is CORRECT and must not be "
               "flagged for it."] if context.get("unvalued_notice") else []),
            "Verified holdings (risk metrics cover the priced equity sleeve of these; "
            "cash/pension carry ~zero market beta, so a symbol/total mismatch with the "
            "risk tools is expected, not a data conflict — only flag a conflict if the "
            "risk tool's 'profile' differs from the profile above):",
            "Cost basis, current price, and return % below ARE verified portfolio data: "
            "ground Rule 8 cost-basis/drawdown claims against them. Note the two "
            f"currencies per line — position value is {base_currency} (converted), while "
            "cost/now/return are in the holding's own native currency; a drawdown "
            "computed from the native pair is correct and is NOT a conflict with the "
            f"{base_currency} value.",
        ]
        for holding in context.get("holdings", [])[:80]:
            allocation = holding.get("allocation_pct")
            allocation_text = f"{allocation:.2f}%" if isinstance(allocation, (int, float)) else "Data Unavailable"
            value_base = holding.get("value_base", holding.get("value_cad", 0))
            # An unvalued holding has no price and no cost basis, so it carries None
            # here. Formatting that as $0.00 would present a position we cannot price
            # as one worth nothing — and Rule 8 grounds drawdown claims on this line,
            # so a fabricated zero becomes a fabricated 100% loss.
            value_text = (f"${value_base:,.2f} {base_currency}"
                          if isinstance(value_base, (int, float))
                          else f"value unknown (not counted in the {base_currency} total)")
            lines.append(
                f"- {holding.get('symbol')}: {value_text} "
                f"({allocation_text}) | {holding.get('account', 'Unknown')} | source={holding.get('source', 'Unknown')}"
                f" | {_format_cost_basis(holding)}"
            )
        return "\n".join(lines)
    except Exception as exc:
        return f"Portfolio verification unavailable: {exc}"











# Common non-ticker uppercase terms to avoid false positives — financial/
# jurisdiction jargon plus everyday capitalized words that show up in advice
# prose (sentence-leading pronouns, verbs, connectives) and match the ticker
# shape (1-5 uppercase letters). Shared by every deterministic check below
# that scans for ticker mentions.
_IGNORED_TICKER_TERMS = {
    "USD", "CAD", "PE", "EPS", "TFSA", "RRSP", "NYSE", "NASDAQ", "SEC", "EDGAR", "GDP", "CPI",
    "USA", "US", "UK", "AI", "CAGR", "VAR", "FCF", "DCF", "ROI", "ROE", "ROIC", "PE_RATIO",
    "SPY", "QQQ", "IWM", "VIX", "ETF",
    "I", "A", "DO", "IF", "OR", "SO", "BE", "IS", "ON", "IN", "TO", "OF", "AS", "AT", "IT", "NO",
    "NOT", "ALL", "ADD", "AND", "ARE", "BUY", "SELL", "TRIM", "HOLD", "PLAN", "ACTION", "NOTE",
    "NEW", "TOP", "LOW", "HIGH", "RISK", "NOW", "WILL", "MUST", "EACH", "WEEK", "YEAR", "DAY",
    "DAYS", "NEXT", "LAST", "THIS", "THAT", "WITH", "FROM", "INTO", "YOUR", "OUR", "ITS", "CASH",
    "FUND", "GOOD", "BAD", "OK", "TIME", "TREND", "TARGET", "STOP", "ENTRY", "EXIT", "MAY", "CAN",
    # Technical-indicator acronyms and C-suite titles: routine in this app's
    # technicals/insider-activity prose, not tickers. Without these, e.g.
    # "close above its 21-day EMA" or "CEO ... sold shares" false-positived
    # as a recommendation to sell/trim a ticker named "EMA" or "CEO".
    "EMA", "SMA", "MACD", "RSI", "ATH", "ATL", "YTD", "MTD", "QTD",
    "CEO", "CFO", "COO", "CTO", "CIO",
    # Section-header / scenario-table label words in this app's own output
    # format ("🔭 EARLY SIGNALS / WATCH", "🔗 EXPOSURE MAP", the Base/Bear
    # scenario rows). They match the ticker shape and, before this, false-
    # positived as phantom tickers flagged for trimming.
    "EARLY", "WATCH", "BASE", "BEAR", "CASE", "PROB", "MAP",
    # Policy / account / market acronyms from the advisor's OWN compliance
    # vocabulary. "IPS" is the one that bit: "Enforce the IPS Limit: you must
    # trim your Technology exposure" put a sell verb 20 chars from a phantom
    # ticker, so the pre-check reported "Recommended to sell/trim IPS, but IPS
    # is NOT currently held" — and because a grounding error caps the verdict,
    # a clean draft was capped at 6/10 over a word the advisor itself wrote.
    "IPS", "IRA", "RESP", "RRIF", "LIRA", "FHSA", "RDSP", "DRIP",
    "FX", "TSX", "ESG", "MER", "AUM", "KYC",
}

# A candidate token used as a MODIFIER of a policy noun ("the IPS limit", "your
# ESG mandate", "the TFSA rule") is vocabulary, not a holding. This is the
# general form of the IPS bug above: the ignore-list catches the acronyms we
# have already seen, this catches the next one. Only fires when EVERY occurrence
# of the token is modifier-shaped — a real ticker discussed anywhere in the
# draft ("NVDA cap breach ... trim NVDA") keeps its other mentions scannable.
_POLICY_MODIFIER_NOUN_RE = re.compile(
    r"^[\s'’]*(?:limits?|caps?|rules?|constraints?|mandates?|policy|policies|"
    r"statements?|guidelines?|thresholds?|breach(?:es)?|violations?|compliance|"
    r"discipline|framework|band|bands|floor|ceiling)\b",
    re.IGNORECASE,
)


def _is_policy_modifier_only(text: str, token: str) -> bool:
    """True when every mention of `token` modifies a policy noun."""
    seen = False
    for m in re.finditer(r"\b" + re.escape(token) + r"\b", text):
        seen = True
        if not _POLICY_MODIFIER_NOUN_RE.match(text[m.end():m.end() + 24]):
            return False
    return seen

_TICKER_RE = re.compile(r'\b[A-Z]{1,5}(?:\.[A-Z]{1,2})?\b')


def _is_allcaps_heading_line(line: str) -> bool:
    """True for short, all-uppercase section-header / table-header lines in this
    app's own output format — e.g. '🔭 EARLY SIGNALS / WATCH', '💼 PORTFOLIO
    EXPOSURE', 'CASE ~PROB MECHANISM TIME CONFIRMS/INVALIDATES'. Such lines carry
    no prose, and their capitalized label words ('EARLY', 'WATCH', 'CASE') match
    the ticker shape — mining them for tickers is what produced phantom-ticker
    sell/trim flags. A lone all-caps token on its own line (possibly a real
    ticker) is deliberately NOT treated as a header."""
    stripped = line.strip()
    if not stripped:
        return False
    if " " not in stripped and "\t" not in stripped:
        return False  # single token — could be a bare ticker; leave it scannable
    if len(stripped) > 70:
        return False  # real section labels are short; an all-caps sentence is prose
    letters = [c for c in stripped if c.isalpha()]
    return bool(letters) and not any(c.islower() for c in letters)


def _extract_candidate_tickers(text: str) -> set[str]:
    """Shape+stopword-filtered ticker candidates (held-status agnostic — callers
    apply their own held/not-held semantics). Shared by every deterministic
    grounding check that needs to find ticker mentions in advice text.

    Section-header lines are stripped before extraction so that all-caps label
    words ('EARLY', 'WATCH', ...) never enter the candidate set."""
    scannable = "\n".join(
        ln for ln in text.splitlines() if not _is_allcaps_heading_line(ln)
    )
    candidates = set(_TICKER_RE.findall(scannable))
    return {
        t for t in candidates
        if t not in _IGNORED_TICKER_TERMS
        and len(t.split(".")[0]) >= 2
        and not _is_policy_modifier_only(scannable, t)
    }


# Sell/close-a-position verbs (whole-word, common suffixes). Deliberately
# excludes "close/closed/closing": in this app's advice prose that word
# overwhelmingly means a PRICE close ("close above its 21-day EMA/50-day SMA"),
# not an instruction to close a position — "sell"/"trim"/"reduce"/"exit"/
# "liquidate"/"dispose"/"divest" already cover real close-a-position phrasing
# without that false-positive source.
_SELL_VERB_RE = re.compile(
    r'\b(?:sell|sold|selling|trim(?:med|ming)?|reduc(?:e|ed|ing)|liquidat(?:e|ed|ing)|'
    r'exit(?:ed|ing)?|dispos(?:e|ed|ing)|divest(?:ed|ing)?)\b',
    re.IGNORECASE,
)

# Negation / keep cues that invert a sell verb's meaning when they precede it:
# "do not trim", "don't sell", "avoid trimming", "hold rather than sell". Without
# this, an explicit instruction to KEEP a position ("Do not trim Tech") reads as
# a sell recommendation and, via the proximity window, gets pinned on whatever
# ticker-shaped token happens to sit within 60 chars.
_SELL_NEGATION_RE = re.compile(
    r"\b(?:not|never|no|avoid(?:s|ing)?|without|instead|rather|"
    r"hold(?:s|ing)?|keep(?:s|ing)?|maintain(?:s|ing)?|retain(?:s|ing)?|"
    r"don'?t|doesn'?t|didn'?t|won'?t|wouldn'?t|shouldn'?t|can'?t|cannot)\b",
    re.IGNORECASE,
)


# Third-party sellers. A sell verb whose subject is an insider, an executive or
# an institution reports what SOMEONE ELSE did — "Heavy insider selling", "CEO
# sold ~$84M", "institutional selling" — and is evidence the advice is reasoning
# FROM, not an instruction to the user. Without this, such prose next to a
# not-held ticker ("Keep MU on WATCHING. Heavy insider selling (CEO sold $84M)
# confirms delaying entry") was flagged as "Recommended to sell/trim MU" — the
# exact opposite of the advice, which was to stay out and keep watching.
_THIRD_PARTY_SELLER_RE = re.compile(
    r"\b(?:insider|insiders|executive|executives|officer|officers|management|"
    r"director|directors|founder|founders|chairman|board|"
    r"institution|institutions|institutional|hedge\s+fund|fund manager)\b",
    re.IGNORECASE,
)

# Descriptive market-condition prose immediately AFTER a sell verb: "sell-off",
# "selling pressure", "selling volume", "sold by insiders". These name what the
# market or a third party did; none is an instruction to the user. ("sell-off"
# matters because \bsell\b matches the "sell" in "sell-off".)
_SELL_DESCRIPTIVE_TRAIL_RE = re.compile(
    r"^(?:-\s*off\b|\s*(?:pressure|volume|activity|spree)\b|"
    r"\s+by\s+(?:the\s+)?(?:insider|executive|officer|management|director|"
    r"founder|chairman|board|institution)\w*\b)",
    re.IGNORECASE,
)


def _is_descriptive_sell(window: str, match: re.Match) -> bool:
    """True when the sell verb at `match` describes what a third party or the
    market did, rather than instructing the user to sell. Insider/institutional
    activity and sell-offs are inputs to a thesis, not recommendations."""
    lead_in = window[max(0, match.start() - 25):match.start()]
    if _THIRD_PARTY_SELLER_RE.search(lead_in):
        return True
    return bool(_SELL_DESCRIPTIVE_TRAIL_RE.match(window[match.end():match.end() + 30]))


def _has_actionable_sell_verb(window: str) -> bool:
    """True only when `window` holds a sell/trim verb that is NOT negated by a
    preceding keep/negation cue within a short lead-in, and is NOT merely
    describing third-party or market selling. 'Do not trim', "don't sell",
    'hold rather than sell', 'avoid trimming' are instructions to KEEP a
    position; 'insider selling', 'CEO sold', 'a broad sell-off' are observations
    about others. Neither is a sell recommendation to the user."""
    for vm in _SELL_VERB_RE.finditer(window):
        preceding = window[max(0, vm.start() - 25):vm.start()]
        if _SELL_NEGATION_RE.search(preceding):
            continue
        if _is_descriptive_sell(window, vm):
            continue
        return True
    return False


# A line that STARTS a new block: a bullet, a numbered item, a markdown heading,
# or a blank separator. Sell verbs must not reach across one of these into the
# next item — see _proximity_window.
_BLOCK_START_RE = re.compile(r"^(?:\s*$|\s*[-*•▪–—+]\s|\s*\d+[.)]\s|\s*#{1,6}\s)")

_PROXIMITY_RADIUS = 60


def _line_spans(text: str) -> list[tuple[int, int, str]]:
    """(start, end, line) for every line, computed once per audit."""
    spans, idx = [], 0
    for line in text.splitlines(keepends=True):
        spans.append((idx, idx + len(line), line))
        idx += len(line)
    return spans


def _block_bounds(spans: list[tuple[int, int, str]], pos: int, text_len: int) -> tuple[int, int]:
    """[start, end) of the list item / paragraph containing `pos`."""
    if not spans:
        return 0, text_len
    cur = next((i for i, (s, e, _) in enumerate(spans) if s <= pos < e), len(spans) - 1)
    start_i = cur
    while start_i > 0 and not _BLOCK_START_RE.match(spans[start_i][2]):
        start_i -= 1
    end_i = cur + 1
    while end_i < len(spans) and not _BLOCK_START_RE.match(spans[end_i][2]):
        end_i += 1
    return spans[start_i][0], spans[end_i - 1][1]


def _proximity_window(text: str, spans, start: int, end: int) -> str:
    """±_PROXIMITY_RADIUS chars around [start, end), clamped to the list item or
    paragraph the mention actually sits in.

    The clamp is the load-bearing part. A bare character radius reaches over a
    line break into the NEIGHBOURING bullet, so a sell verb belonging to a
    different subject gets pinned on whatever ticker leads the next item. Real
    regression: "...weight is trimmed back below the 30% policy limit.\\n*
    Watchlist (MU): the Micron thesis remains dropped/closed ... We do not
    chase." produced "Recommended to sell/trim MU, but MU is NOT currently
    held" — the verb was about the Technology sleeve, and the draft had
    explicitly said it would NOT touch MU. Same principle (and the same failure
    it was written for) as tools.ips_precheck._evidence_window.
    """
    lo, hi = _block_bounds(spans, start, len(text))
    return text[max(lo, start - _PROXIMITY_RADIUS):min(hi, end + _PROXIMITY_RADIUS)]


def run_deterministic_grounding_audit(text: str) -> list[str]:
    """
    Pure-Python pre-audit grounding check (a): not-held tickers coupled with
    sell/trim verbs. Returns a list of violation strings.
    """
    from tools.portfolio_csv import get_portfolio_decision_context

    violations = []
    try:
        portfolio_ctx = get_portfolio_decision_context()
        held_tickers = set()
        held_base_symbols = set()
        if isinstance(portfolio_ctx, dict) and "holdings" in portfolio_ctx:
            for h in portfolio_ctx["holdings"]:
                sym = h.get("symbol")
                if sym:
                    sym = sym.strip().upper()
                    held_tickers.add(sym)
                    held_base_symbols.add(sym.split(".")[0])  # BCE.TO -> BCE

        # Sorted by first appearance for deterministic violation ordering across
        # identical runs (set iteration order is otherwise nondeterministic).
        potential_tickers = sorted(
            _extract_candidate_tickers(text), key=lambda t: (text.find(t), t)
        )
        line_spans = _line_spans(text)

        for ticker in potential_tickers:
            base_symbol = ticker.split(".")[0]
            # Compare against both the literal held symbol and its base (BCE matches held BCE.TO)
            if ticker in held_tickers or base_symbol in held_base_symbols:
                continue
            # Find occurrences of the ticker in text and look around it for an
            # actionable (non-negated) sell verb — within its own block only.
            for match in re.finditer(r'\b' + re.escape(ticker) + r'\b', text):
                window = _proximity_window(text, line_spans, match.start(), match.end())
                if _has_actionable_sell_verb(window):
                    violations.append(
                        f"Grounding Error: Recommended to sell/trim {ticker}, but {ticker} is NOT currently held in the portfolio."
                    )
                    break
    except Exception as e:
        from agent.utils import safe_print
        safe_print(f"⚠️ Grounding audit error: {e}")

    return violations


# Currency symbols/codes recognized in headline totals. CairnIQ is NOT a CAD/US-
# only app — base currency is user-configurable across all of
# tools.memory.SUPPORTED_BASE_CURRENCIES (USD/CAD/EUR/GBP/AUD/JPY), so this must
# never assume the user's currency is one of a hardcoded pair. A bare "$" is
# ambiguous (USD/CAD/AUD all use it) and deliberately left unresolved.
_CURRENCY_SYMBOL_TO_CODE = {"€": "EUR", "£": "GBP", "¥": "JPY"}
_CURRENCY_PREFIX_CODE = {"C$": "CAD", "US$": "USD", "A$": "AUD"}
_CURRENCY_CODES = {"USD", "CAD", "EUR", "GBP", "AUD", "JPY"}

# Cues that mark a figure as DERIVED FROM the total rather than being the total.
# "your total portfolio's 2% maximum risk limit ($10,000 CAD)" satisfies the
# valuation-keyword gate below on the word "total", so without this guard the
# audit reported the correctly-computed 2% limit as a hallucinated portfolio
# total — a false grounding error, which caps the verdict at ≤6/10 and forces a
# retry of advice that was right. Matched against the text BETWEEN "portfolio"
# and the figure, which is where such a qualifier always sits.
_DERIVED_FIGURE_RE = re.compile(
    r"\b(?:\d+(?:\.\d+)?\s*%|percent|per\s*cent|risk|limit|cap|ceiling|max(?:imum)?|"
    r"budget|threshold|headroom|drawdown|per\s+trade|at\s+risk|target|"
    r"gain|loss|profit|dividend|income|fee|tax)\b",
    re.IGNORECASE,
)


def run_deterministic_total_audit(text: str) -> list[str]:
    """
    Pure-Python pre-audit grounding check (b): headline portfolio total &
    currency vs `portfolio_verification`. Directly attacks the CAD/USD-headline
    class of bug this repo has fought hardest — a portfolio total stated in the
    wrong currency, or a stale/wrong number, presented as fact.

    Currency-agnostic: compares against total_value_base (the true total in
    whatever currency the profile's base_currency actually is — USD, CAD, EUR,
    GBP, AUD, or JPY) plus the total_value_cad/total_value_usd conversions the
    app always also computes, so a correct headline in ANY supported currency
    is accepted, not just CAD or USD.

    Deliberately narrow: only matches "portfolio ... worth/valued at/total(s)
    ... $X" style headline phrasing, not every dollar figure in the text (a
    single stock's price, a cash balance, a proposed stop-loss, etc. are all
    legitimate and must not be flagged).
    """
    from tools.portfolio_csv import get_portfolio_decision_context

    violations = []
    try:
        ctx = get_portfolio_decision_context()
        if not isinstance(ctx, dict) or ctx.get("error"):
            return violations

        base_currency = str(ctx.get("base_currency") or "USD").upper()

        # Every currency we can actually verify a headline against: the true
        # base-currency total plus the two conversions the app always computes.
        available_totals: dict[str, float] = {}
        for currency, key in ((base_currency, "total_value_base"), ("CAD", "total_value_cad"), ("USD", "total_value_usd")):
            value = ctx.get(key)
            if isinstance(value, (int, float)) and value > 0:
                available_totals.setdefault(currency, float(value))
        if not available_totals:
            return violations

        def _close(stated: float, actual: float) -> bool:
            return abs(stated - actual) / actual <= 0.05  # 5% tolerance for staleness/rounding

        # "portfolio ... (worth|valu*|total*) ... [$€£¥]X [CAD|USD|EUR|GBP|AUD|JPY|C$|US$|A$]"
        # — the valuation keyword must appear between "portfolio" and the currency
        # symbol (within a short window), so an unrelated dollar figure that
        # merely follows the word "portfolio" somewhere in the same paragraph (a
        # stock price, a cash balance) is not treated as a total-value claim.
        headline_re = re.compile(
            r'portfolio([^.$€£¥\n]{0,45}?)([$€£¥])\s*([\d,]+(?:\.\d+)?)\s*(CAD|USD|EUR|GBP|AUD|JPY|C\$|US\$|A\$)?',
            re.IGNORECASE,
        )
        for m in headline_re.finditer(text):
            connector = m.group(1).lower()
            # The valuation keyword may sit on EITHER side of "portfolio": the
            # app's own canonical headline is "Total Portfolio: $X" (keyword
            # before), while free prose tends to put it after ("portfolio is
            # worth $X"). Check a short same-sentence lead-in as well as the
            # trailing connector, so the primary format is actually audited
            # instead of being silently skipped.
            lead = re.split(r'[.\n]', text[max(0, m.start() - 25):m.start()])[-1].lower()
            if not any(kw in connector or kw in lead for kw in ("worth", "valu", "total")):
                continue
            # ...but a qualifier between "portfolio" and the figure means the
            # figure is derived FROM the total, not a claim about it.
            if _DERIVED_FIGURE_RE.search(connector):
                continue
            try:
                stated = float(m.group(3).replace(",", ""))
            except ValueError:
                continue
            if stated <= 0:
                continue

            symbol = m.group(2)
            trailing = (m.group(4) or "").upper()
            if trailing in _CURRENCY_CODES:
                stated_currency = trailing
            elif trailing in _CURRENCY_PREFIX_CODE:
                stated_currency = _CURRENCY_PREFIX_CODE[trailing]
            elif symbol in _CURRENCY_SYMBOL_TO_CODE:
                stated_currency = _CURRENCY_SYMBOL_TO_CODE[symbol]
            else:
                stated_currency = ""  # bare "$" — ambiguous, not a labeled claim

            matches = {cur: _close(stated, val) for cur, val in available_totals.items()}
            correct_currency = next((cur for cur, ok in matches.items() if ok), None)

            if correct_currency is None:
                display_currency, display_total = base_currency, available_totals.get(base_currency) or next(iter(available_totals.values()))
                violations.append(
                    f"Grounding Error: Advice states portfolio total {stated:,.0f}"
                    f"{(' ' + stated_currency) if stated_currency else ''}, but the verified total is "
                    f"{display_total:,.0f} {display_currency}."
                )
            elif stated_currency and stated_currency in available_totals and not matches[stated_currency]:
                violations.append(
                    f"Grounding Error: Advice labels portfolio total {stated:,.0f} as {stated_currency}, but "
                    f"that figure matches the {correct_currency} total ({available_totals[correct_currency]:,.0f}), "
                    f"not the {stated_currency} total ({available_totals[stated_currency]:,.0f})."
                )
    except Exception as e:
        from agent.utils import safe_print
        safe_print(f"⚠️ Total/currency audit error: {e}")

    return violations


# Current-price framing only — deliberately excludes proposed entry/stop/
# target/resistance/support language, which is meant to differ from the
# current price by design and must never be flagged as a mismatch.
_CURRENT_PRICE_CUE_RE = re.compile(
    r'\b(?:trading at|currently at|current price(?:\s+of)?|priced at|quote(?:d)?\s+at|'
    r'last(?:\s+price)?(?:\s+of)?|closed at|now at|sits at)\b',
    re.IGNORECASE,
)
_ALLOCATION_CUE_RE = re.compile(
    r'\b(?:of your portfolio|portfolio allocation|allocation|portfolio weight|position size)\b',
    re.IGNORECASE,
)
_MAX_QUOTE_LOOKUPS_PER_AUDIT = 8  # bound worst-case latency/cost per RiskManager pass


def run_deterministic_price_audit(text: str) -> list[str]:
    """
    Pure-Python pre-audit grounding check (c1): current-price consistency.

    For each ticker mentioned with an explicit CURRENT-price framing ("trading
    at", "current price", "priced at", ...), fuzzy-matches the stated dollar
    figure against a cached live quote (get_realtime_quote — same TTL cache
    every market-data tool already shares, so this is a fast, usually-warm
    lookup, not a fresh network call). Deliberately does not touch proposed
    entry/stop/target prices (see _CURRENT_PRICE_CUE_RE) — those are meant to
    differ from the live price, and flagging them would reintroduce the exact
    false-positive problem check (a) already had before it was hardened.
    """
    from tools.market_data import get_realtime_quote

    violations = []
    try:
        # Sorted by first appearance so the _MAX_QUOTE_LOOKUPS_PER_AUDIT cap
        # selects the SAME tickers every run — set iteration order is otherwise
        # nondeterministic, which would make a "deterministic" audit surface
        # different violations on identical input.
        candidates = sorted(
            _extract_candidate_tickers(text), key=lambda t: (text.find(t), t)
        )
        checked = 0
        for ticker in candidates:
            if checked >= _MAX_QUOTE_LOOKUPS_PER_AUDIT:
                break
            for match in re.finditer(r'\b' + re.escape(ticker) + r'\b', text):
                start = max(0, match.start() - 50)
                end = min(len(text), match.end() + 50)
                window = text[start:end]
                if not _CURRENT_PRICE_CUE_RE.search(window):
                    continue
                price_match = re.search(r'\$\s*([\d,]+(?:\.\d+)?)', window)
                if not price_match:
                    continue
                try:
                    stated = float(price_match.group(1).replace(",", ""))
                except ValueError:
                    break
                if stated <= 0:
                    break
                checked += 1
                quote = get_realtime_quote(ticker)
                actual = quote.get("price") if isinstance(quote, dict) else None
                if isinstance(actual, (int, float)) and actual > 0 and abs(stated - actual) / actual > 0.05:
                    violations.append(
                        f"Grounding Error: Advice states {ticker} is trading at ${stated:,.2f}, "
                        f"but the last verified quote is ${actual:,.2f}."
                    )
                break  # one price-consistency check per ticker occurrence is enough
    except Exception as e:
        from agent.utils import safe_print
        safe_print(f"⚠️ Price audit error: {e}")

    return violations


def run_deterministic_allocation_audit(text: str) -> list[str]:
    """
    Pure-Python pre-audit grounding check (c2): allocation-percentage
    consistency. For each held ticker mentioned with an explicit
    portfolio-allocation framing ("X% of your portfolio", "X% allocation",
    ...), fuzzy-matches the stated percentage against the holding's actual
    allocation_pct from get_portfolio_decision_context.
    """
    from tools.portfolio_csv import get_portfolio_decision_context

    violations = []
    try:
        ctx = get_portfolio_decision_context()
        if not isinstance(ctx, dict) or ctx.get("error"):
            return violations

        allocations: dict[str, float] = {}
        allocations_by_base: dict[str, float] = {}
        for h in ctx.get("holdings", []):
            sym = h.get("symbol")
            pct = h.get("allocation_pct")
            if sym and isinstance(pct, (int, float)):
                sym_u = str(sym).strip().upper()
                allocations[sym_u] = float(pct)
                allocations_by_base[sym_u.split(".")[0]] = float(pct)
        if not allocations:
            return violations

        # Sorted by first appearance so the shared lookup cap is stable across
        # identical runs (see run_deterministic_price_audit for the rationale).
        candidates = sorted(
            _extract_candidate_tickers(text), key=lambda t: (text.find(t), t)
        )
        checked = 0
        for ticker in candidates:
            if checked >= _MAX_QUOTE_LOOKUPS_PER_AUDIT:
                break
            actual = allocations.get(ticker, allocations_by_base.get(ticker.split(".")[0]))
            if actual is None:
                continue
            for match in re.finditer(r'\b' + re.escape(ticker) + r'\b', text):
                start = max(0, match.start() - 15)
                end = min(len(text), match.end() + 45)
                window = text[start:end]
                if not _ALLOCATION_CUE_RE.search(window):
                    continue
                pct_match = re.search(r'([\d.]+)\s*%', window)
                if not pct_match:
                    continue
                try:
                    stated = float(pct_match.group(1))
                except ValueError:
                    break
                checked += 1
                if abs(stated - actual) > 2.0:  # 2 percentage-point tolerance
                    violations.append(
                        f"Grounding Error: Advice states {ticker} is {stated:.1f}% of your portfolio, "
                        f"but the verified allocation is {actual:.2f}%."
                    )
                break
    except Exception as e:
        from agent.utils import safe_print
        safe_print(f"⚠️ Allocation audit error: {e}")

    return violations


def parse_risk_verdict(text: str) -> tuple[int, bool, list[str]]:
    """
    Parses the Risk Judge's Markdown output.
    Returns: (score, is_compliant, list_of_violations)

    Fails CLOSED, not open: if the judge's output doesn't match the mandated
    "Verdict: [X/10]" template — e.g. markdown-emphasized ("Verdict: **3/10**"),
    spaced ("Verdict: 3 / 10"), or missing entirely (free-form prose with no
    Verdict/Risks markers) — this must never be treated as a clean pass. Silently
    defaulting to a perfect score on template drift is exactly how a parsed,
    gated verdict degrades back into unenforced commentary.
    """
    import re

    # Deliberate low-token fast path: risk_manager_node's internal check prompt
    # explicitly permits replying with just this phrase ("Reply '✅ Risk Check
    # Passed' if safe"), so it must short-circuit to a clean pass rather than
    # fall through to the fail-closed "no verdict found" branch below.
    stripped = text.strip()
    if stripped in ("Risk Check Passed", "✅ Risk Check Passed"):
        return 10, True, []

    # Tolerate markdown emphasis/bracket noise and spaced slashes between the
    # "Verdict:" label and the digits (e.g. "[7/10]", "**3/10**", "3 / 10").
    score_match = re.search(r'Verdict:?\s*[\[\*]*\s*(\d{1,2})\s*/\s*10', text, re.IGNORECASE)
    if score_match:
        try:
            score = int(score_match.group(1))
        except ValueError:
            score = 5
    else:
        # No parseable verdict at all — fail closed to a middling score so this
        # can never resolve to an automatic PASS below.
        score = 5

    violations = []
    # Capture everything under the Risks header up to the NEXT section marker
    # (Devil's Advocate / a following Verdict / a ✅ pass line) or end of text —
    # NOT the first blank line. The judge routinely puts a blank line between
    # "Risks:" and its bullets (and between bullets); stopping at "\n\n" captured
    # an EMPTY block, so listed violations went uncounted and a high-scoring
    # verdict with a real flagged risk could slip through as compliant.
    risks_section = re.search(
        r"Risks:\**\s*(.*?)(?=\n\s*🤔|\n\s*Devil'?s Advocate|\n\s*⚖️|\n\s*✅|\n\s*Verdict\b|\Z)",
        text, re.DOTALL | re.IGNORECASE,
    )
    if risks_section:
        lines = [line.strip().lstrip("-*•").strip() for line in risks_section.group(1).split("\n") if line.strip()]
        for line in lines:
            if line and "none flagged" not in line.lower() and "no violations" not in line.lower() and "clean" not in line.lower():
                violations.append(line)
    elif not score_match:
        # Free-form prose with neither a Verdict nor a Risks block: we can't
        # enumerate specific violations, but we also can't claim there are none.
        violations.append(
            "Risk Judge output did not follow the required verdict format — "
            "unable to verify compliance deterministically."
        )

    is_compliant = (score >= 8) and len(violations) == 0
    return score, is_compliant, violations


def _build_tool_execution_context(all_messages, upstream_tool_ctx, upstream_turn_key=None):
    """Assemble the RiskManager's grounding evidence (<tool_execution_context>).

    A turn's tool results reach this node by two separate routes, and the judge needs
    BOTH. An analyst's calls land in the graph state as ToolMessages, reconstructed
    full-fidelity here. DeepReasoning's heavy path returns only its synthesis — its tool
    results never enter `messages` at all — and publishes them instead through
    data_context["tool_execution_context"].

    `upstream_turn_key` is what makes combining them safe. data_context has no state
    reducer, so an upstream copy left by an EARLIER turn is indistinguishable by content
    from one written moments ago; auditing a fast-path turn against a previous turn's
    evidence false-flags every current metric as Rule 8 source fraud. So:

      - upstream stamped with THIS turn  -> union it with the in-state results, the
        heavy path winning on collision (it runs downstream of the analysts, so its
        copy of a re-run call is the newer one);
      - upstream unstamped or from another turn -> in-state results only, and the
        upstream copy is used solely when this turn produced no ToolMessages at all
        (the pure heavy-path turn, where it IS this turn's only evidence).

    Preferring one route outright was the previous rule, and on a turn that had both —
    an analyst ran tools, then the deep path ran more — it discarded everything the deep
    path fetched. Real regression, 2026-07-29: EMA21 levels from structure_trade_setup
    and a sector weight from check_portfolio_allocation, both genuinely fetched, were
    called fabricated in a 2/10 SOURCE FRAUD verdict.

    The reconstructed evidence is deliberately NOT compressed: a shrunk copy can hide the
    very mid-result datapoint a grounding check depends on (this is the same class of bug
    that once cost a real, grounded Trump-Yap catalyst a false "fabricated" verdict).
    """
    tool_map = {}
    for msg in all_messages:
        if isinstance(msg, ToolMessage):
            tool_map[msg.tool_call_id] = {
                "content": str(msg.content),
                "name": msg.name or "unknown_tool",
            }

    # Current-turn boundary: the last GENUINE user message. Skip the synthetic
    # <compliance_correction_required> directive the retry gate injects — otherwise a
    # retry pass scans "after" it, finds no tools, and loses this turn's real evidence.
    boundary = 0
    for i in range(len(all_messages) - 1, -1, -1):
        m = all_messages[i]
        if isinstance(m, HumanMessage) and not str(
            getattr(m, "content", "")
        ).lstrip().startswith("<compliance_correction_required>"):
            boundary = i
            break

    tool_results = []
    for msg in all_messages[boundary:]:
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            for tc in msg.tool_calls:
                tc_id = tc.get("id")
                if tc_id in tool_map:
                    tool_results.append(
                        f"### Tool Call: {tc.get('name')}({tc.get('args')})\n"
                        f"Result:\n{tool_map[tc_id]['content']}"
                    )

    upstream_is_this_turn = bool(upstream_tool_ctx) and bool(upstream_turn_key) and (
        upstream_turn_key == current_turn_key(all_messages)
    )

    if tool_results:
        in_state_ctx = "\n\n".join(tool_results)
        if upstream_is_this_turn:
            return merge_tool_contexts(in_state_ctx, upstream_tool_ctx)
        return in_state_ctx
    if upstream_tool_ctx:
        return upstream_tool_ctx
    return "No tool calls executed in recent context."


def _current_turn_query(all_messages) -> str:
    """The current turn's genuine user query (skips the synthetic
    <compliance_correction_required> directive a retry pass injects)."""
    for msg in reversed(all_messages):
        if isinstance(msg, HumanMessage):
            content = str(getattr(msg, "content", ""))
            if content.lstrip().startswith("<compliance_correction_required>"):
                continue
            return content
    return ""


def _build_judge_context(all_messages, retry_count=0):
    """Build the judge's message window: sanitized copies of the recent
    conversation plus the appended risk-check HumanMessage.

    Sanitizing (all passes): tool_calls are stripped from AIMessages and their
    now-orphaned ToolMessages dropped (unresolved tool_use blocks throw
    ValidationException on Bedrock), list content is flattened to text, and
    synthetic <compliance_correction_required> directives are dropped — they
    are not user-authored, and once the retry they triggered is resolved they
    add nothing but stale "you must correct these violations" noise on every
    later turn in the same thread.

    Retry passes need two extra guards. When the compliance gate routes a
    CRITICAL_FAILed draft back through DeepReasoning, the judge runs again on
    the revision — but the recent-message window still contains the ORIGINAL
    failing draft and the judge's own failing verdict. Left as-is, the judge
    anchors on that prior verdict and re-issues it against the original draft,
    so a fully compliant revision inherits the same CRITICAL_FAIL and ships
    with an undeserved "flagged on both attempts" banner. So on a retry pass —
    the gate incremented the counter, or (if the counter was lost) a correction
    directive stands as the current turn's last HumanMessage — we (a) drop the
    judge's own prior verdict AIMessages and (b) spell out in the check message
    that ONLY the most recent assistant message, the revision, is under review.
    """
    retry_pass = retry_count > 0
    if not retry_pass:
        for m in reversed(all_messages):
            if isinstance(m, HumanMessage):
                retry_pass = str(getattr(m, "content", "")).lstrip().startswith(
                    "<compliance_correction_required>"
                )
                break

    clean_messages = []
    for m in all_messages:
        if isinstance(m, AIMessage):
            if retry_pass and getattr(m, "name", None) == "RiskManager":
                # Prior verdicts are what the judge anchors on — see docstring (a).
                continue
            # Copy without tool_calls: the judge doesn't need internal tool usage
            # details, just the text content.
            # CRITICAL: Flatten content if it is a list (Bedrock tool_use blocks)
            content = m.content
            if isinstance(content, list):
                text_parts = []
                for item in content:
                    if isinstance(item, str):
                        text_parts.append(item)
                    elif isinstance(item, dict) and item.get("type") == "text":
                        text_parts.append(item.get("text", ""))
                content = "\n".join(text_parts)

            # Drop the watch-conditions side-channel (Roadmap 3.3). It is
            # machine-readable state, not advice: the user never sees it, and
            # leaving it in the window gives the judge a block of JSON numbers to
            # audit as if they were prose claims — the false-flag class this
            # layer keeps getting bitten by.
            clean_messages.append(AIMessage(content=strip_watch_blocks(str(content)), name=m.name, id=m.id))
        elif isinstance(m, ToolMessage):
            # Orphaned once tool_calls were stripped above — see docstring.
            continue
        elif isinstance(m, HumanMessage) and str(getattr(m, "content", "")).lstrip().startswith("<compliance_correction_required>"):
            # Synthetic correction directive — see docstring.
            continue
        else:
            clean_messages.append(m)

    # Only keep the last few messages for context (no need for full history),
    # then ensure valid turn structure for Anthropic (User -> Assistant -> User).
    internal_messages = clean_messages[-6:]
    if internal_messages and isinstance(internal_messages[-1], AIMessage):
        if retry_pass:
            internal_messages.append(HumanMessage(content=(
                "Quick risk check: the advice under review is ONLY the most recent "
                "assistant message above. It is a REVISION of an earlier draft that "
                "failed compliance; any earlier drafts in this context are historical "
                "and superseded — do NOT re-flag violations that appear only in them. "
                "Judge the revision on its own content. "
                "Reply '✅ Risk Check Passed' if safe."
            )))
        else:
            internal_messages.append(HumanMessage(content="Quick risk check on the above advice. Reply '✅ Risk Check Passed' if safe."))
    elif not internal_messages:
        internal_messages.append(HumanMessage(content="Check for compliance."))
    return internal_messages


def judge_llm():
    """The LLM the compliance judge runs on.

    DEFAULTS TO THE DEEP TIER, and that default should not be changed casually.
    The judge emits at most ~200 words, which reads like an obvious candidate for
    the fast tier, but this node has a documented history of getting it wrong in
    expensive ways: inventing a profile rule the advisor then obeyed, and calling
    genuinely-fetched numbers fabricated under Rule 8 SOURCE FRAUD. A cheaper
    judge that hallucinates a violation does not save money — it corrupts advice
    and triggers a full compliance-retry cycle.

    So this is a SEAM, not a change of behaviour. Set AIDLC_JUDGE_TIER=fast and
    run the golden harness with `--live` to A/B it; the gate on flipping the
    default is that eval, not this code. `judge_advice(llm=...)` stays the pure
    injection point the harness drives.
    """
    tier = (os.environ.get("AIDLC_JUDGE_TIER") or "deep").strip().lower()
    if tier in ("fast", "sonnet"):
        return get_sonnet_llm()
    return get_llm()


# Static judge instructions (cacheable) — identical every call, so they sit
# behind the prompt's cachePoint. Factored to a module constant so the pure
# judge seam and the node share exactly one copy.
_JUDGE_RULES_SLOT = "__JUDGE_RULES__"

_JUDGE_STATIC_TEMPLATE = (
    "You are the Risk Compliance Judge. You evaluate proposed investment advice strictly for portfolio, risk, and structural violations.\n"
    "\n"
    "Return these Markdown blocks:\n"
    "<output_format strict=\"true\">\n"
    "⚖️ **Verdict: [X/10]** — [one-line compliance summary]\n"
    "\n"
    "🔴 **Risks:** [Bullet points listing specific rules violated, or 'None flagged' if clean]\n"
    "\n"
    "🤔 **Devil's Advocate:** [1-2 sentences challenging the fundamental bear case/macro risk]\n"
    "</output_format>\n"
    "\n"
    "<rules>\n"
    "- Be sharp, contrarian, and concise. Max 200 words total.\n"
    "- Print only the Markdown blocks. Omit XML tags in final output.\n"
    "- AUDIENCE: this verdict is shown VERBATIM to the investor, who sees only the advice and your reply — never your inputs. Write every finding as an investment objection in your own voice, addressed to them.\n"
    "  Never name or narrate this application's internals: the deterministic pre-check, the audit block, the IPS pre-check table, 'the parser', 'system rules', or how the score is computed. 'The automated pre-check flagged a Sell IPS error, so strict system rules cap this at 6/10' is not a risk finding — it reports plumbing the user cannot see, did not read, and cannot act on, and it reads as an unsourced claim about their own portfolio.\n"
    "  If a pre-check item looks mistaken to you, do not argue with the machinery in front of the user: state plainly what is and is not true about the ADVICE, in investor language, and move on. The score is computed for you either way, so you never need to explain or justify a number.\n"
    "- DO NOT use strikethrough (~~text~~) markdown. Present contradictions and assumptions clearly with text instead.\n"
    "- For informational scans, watchlists, screeners, or macro overviews, focus on checking the accuracy, logic, data integrity, consistency, and structural risks of the presented information. Flag contradictions (e.g. a stock flagged as 'Entry Missed' in active theses but ranked as a top 'Golden Opportunity'), data inconsistencies (e.g. negative FCF or high debt labeled as 'Strong Foundation'), or unhedged macro assumptions.\n"
    f"{_JUDGE_RULES_SLOT}"
    "</rules>"
)


@dataclass
class JudgeOutcome:
    """The result of one judge pass over a single advice draft."""
    score: int
    risk_result: str                 # "PASS" | "FAIL" | "CRITICAL_FAIL"
    is_compliant: bool
    grounding_violations: list = field(default_factory=list)
    llm_violations: list = field(default_factory=list)
    all_violations: list = field(default_factory=list)
    verdict_text: str = ""           # normalized judge output (before any banner)
    ips_result: dict = field(default_factory=dict)
    data_quality: dict = field(default_factory=dict)  # Roadmap 2.3 turn provenance


def _build_provenance_block(data_quality: dict) -> str:
    """The judge's view of what this turn's evidence was worth (Roadmap 2.3).

    Deliberately NOT phrased as a violation. A missing API key is not misconduct
    by the advice, and routing it through the deterministic-audit block would
    make every unconfigured data source a CRITICAL_FAIL — the advisor would stop
    answering rather than answer with a caveat. What the judge is told is the
    fact and the one duty that follows from it: advice resting on a source that
    could not be checked has to say so.

    Empty string when nothing was degraded, so a clean turn spends no tokens and
    the block's presence is itself the signal.
    """
    if not data_quality.get("degraded"):
        return ""

    lines = []
    for entry in data_quality.get("sources", []):
        if entry["status"] == PROV_UNAVAILABLE:
            label = entry.get("source") or entry["tool"]
            reason = entry.get("reason") or "no reason reported"
            lines.append(f"- {entry['tool']}: {label} UNAVAILABLE — {reason}")
        elif entry["status"] == PROV_STALE:
            lines.append(
                f"- {entry['tool']}: data was fetched {entry['age_minutes']:.0f} minutes ago, "
                f"not live"
            )
    if not lines:
        return ""

    return (
        "\n<data_provenance>\n"
        "These are CONFIRMED facts about the evidence above, not opinions:\n"
        + "\n".join(lines)
        + "\n\nIf the advice makes a claim that depends on one of these sources, it MUST say "
        "the source was unavailable or not live. Advice that stays silent about a "
        "degradation it depends on is overstating its own evidence. If the advice does not "
        "rely on the degraded source, this is not a fault — do not manufacture one, and do "
        "not mention this block, name it, or tell the user a pre-check produced it.\n"
        "</data_provenance>\n"
    )


def _provenance_footer_line(data_quality: dict) -> str:
    """The one-line "what this answer was built on" footer (Roadmap 2.3).

    Shown only when there is something a reader could act on — a named
    unavailable source, or data old enough to matter. A footer under every
    answer would be read for a week and then never again, and the one time it
    said something load-bearing it would be wallpaper. Same argument as the
    drawdown playbook's deep-band-only rule.
    """
    footer = (data_quality or {}).get("footer") or ""
    if not footer or not data_quality.get("degraded"):
        return ""
    return f"\n\n---\n*Data provenance: {footer}.*"


def judge_advice(
    advice_text: str,
    *,
    llm=None,
    judge_messages=None,
    user_memory_ctx: str = "",
    portfolio_verification_ctx: str = "",
    tool_execution_ctx: str = "",
    stream: bool = False,
) -> JudgeOutcome:
    """Run ONE judge pass over `advice_text`: the deterministic grounding + IPS
    pre-audit, the LLM Risk Judge, and the verdict parse/score-cap/result logic —
    and nothing else. No verdict persistence, no retry gate, no status-node
    bookkeeping. `risk_manager_node` wraps this with those production concerns;
    the Theme 2.4 eval harness calls it directly on fixtures, so it can exercise
    the real judge path WITHOUT writing to the per-profile verdict audit log.

    Args mirror the context the node assembles: `judge_messages` is the judge's
    message window (production passes `_build_judge_context(...)`; when None, a
    minimal single-draft window is built). The three `*_ctx` strings are the
    dynamic prompt context. `stream=True` reproduces the node's live token
    streaming (only when a stream callback is active); the harness leaves it
    False for a silent invoke.
    """
    if llm is None:
        llm = judge_llm()

    # Audit the ADVICE, not the machinery. The watch-conditions side-channel
    # (Roadmap 3.3) is a JSON block of trigger levels appended after the prose
    # and stripped before display; every deterministic audit below scans free
    # text for numbers, so leaving it in would let a stored threshold read as an
    # unsourced price claim or a phantom trade. Stripping here covers the node
    # and the 2.4 harness in one place, since both enter through this seam.
    advice_text = strip_watch_blocks(advice_text or "")

    # --- DETERMINISTIC GROUNDING PRE-AUDIT (checks a, b, c1, c2 + IPS 2.2) ---
    grounding_violations: list[str] = []
    ips_result = {"trades": [], "rows": [], "violations": [], "block": ""}
    if advice_text:
        grounding_violations.extend(run_deterministic_grounding_audit(advice_text))
        grounding_violations.extend(run_deterministic_total_audit(advice_text))
        grounding_violations.extend(run_deterministic_price_audit(advice_text))
        grounding_violations.extend(run_deterministic_allocation_audit(advice_text))
        from tools.ips_precheck import run_ips_precheck
        ips_result = run_ips_precheck(advice_text, _extract_candidate_tickers(advice_text))
        grounding_violations.extend(ips_result["violations"])

    grounding_block = ""
    if grounding_violations:
        violations_text = "\n".join(f"- {v}" for v in grounding_violations)
        grounding_block = (
            f"\n<deterministic_audit>\n"
            f"The following violations were found by automated pre-checks (these are CONFIRMED facts, not LLM opinions):\n"
            f"{violations_text}\n"
            f"You MUST incorporate these into your verdict — but as findings about the ADVICE, in your own investor-facing voice. "
            f"Do not quote this block, name it, or tell the user a pre-check produced it; they never see it. "
            f"The score cap for a grounding error is applied automatically after you reply, so do not compute, announce, or justify it.\n"
            f"</deterministic_audit>\n"
        )

    # Roadmap 2.3 — the turn's data provenance, derived from the very evidence
    # block above so the two can never disagree. Read-only context for the judge;
    # the score cap it drives is applied deterministically after the reply.
    data_quality = summarize_tool_context(tool_execution_ctx)
    provenance_block = _build_provenance_block(data_quality)

    _dynamic_context = (
        f"<today>{datetime.now().strftime('%Y-%m-%d')}</today>\n"
        f"<user_profile_memory>\n{_prompt_escape(user_memory_ctx)}\n</user_profile_memory>\n"
        f"<portfolio_verification_context>\n{_prompt_escape(portfolio_verification_ctx)}\n</portfolio_verification_context>\n"
        f"<tool_execution_context>\n{_prompt_escape(tool_execution_ctx)}\n</tool_execution_context>"
        f"{provenance_block}"
        f"{grounding_block}"
        f"{ips_result['block']}"
    )
    # Bound per profile rather than at import: the magnitude rule has to name the
    # user's OWN risk limit, or state that they set none. Still ahead of the
    # cachePoint — it is stable for a given profile, and a user changing their
    # stated limits is exactly when this prefix should stop being reused.
    judge_instructions = _JUDGE_STATIC_TEMPLATE.replace(_JUDGE_RULES_SLOT, risk_rules_judge())
    system_prompt = [
        {"text": judge_instructions},
        {"cachePoint": {"type": "default"}},
        {"text": _dynamic_context},
    ]

    if judge_messages is None:
        judge_messages = [
            AIMessage(content=advice_text or "", name="PortfolioManager"),
            HumanMessage(content="Quick risk check on the above advice. Reply '✅ Risk Check Passed' if safe."),
        ]

    agent = create_agent(llm, [], system_prompt)

    full_content = ""
    if stream and has_stream_callback():
        send_status("⚖️ Analyzing Risks...")
        # RiskManager's verdict always follows another node's output in the same
        # turn; the chat stream has no boundary of its own, so a separator keeps
        # the verdict from gluing onto the preceding text.
        _separator = "\n\n---\n\n"
        full_content = _separator
        send_stream(_separator)
        try:
            from agent.utils import safe_stream
            for chunk in safe_stream(agent, {"messages": judge_messages}, is_cancelled):
                content_chunk = chunk.content
                text_chunk = ""
                if isinstance(content_chunk, str):
                    text_chunk = content_chunk
                elif isinstance(content_chunk, list):
                    text_chunk = "".join([i.get("text", "") if isinstance(i, dict) else str(i) for i in content_chunk])
                if text_chunk:
                    full_content += text_chunk
                    send_stream(text_chunk)
                send_thinking(extract_reasoning_text(content_chunk))
        except Exception as e:
            safe_print(f"⚠️ Risk streaming failed: {e}")
            result = safe_invoke(agent, {"messages": judge_messages})
            full_content = _separator + str(result.content or "")
    else:
        result = safe_invoke(agent, {"messages": judge_messages})
        full_content = result.content

    # Normalize content (extract from list if needed)
    if isinstance(full_content, list):
        full_content = "".join([i.get("text", "") if isinstance(i, dict) else str(i) for i in full_content])
    elif not full_content:
        full_content = "Risk Check Passed"

    # Shorten only when the ENTIRE verdict is a bare pass (avoids dropping caveats
    # from a longer response that merely mentions these phrases).
    if isinstance(full_content, str):
        stripped = full_content.strip()
        if stripped in ("Risk Check Passed", "✅ Risk Check Passed") or (
            len(stripped) < 60 and "risk check passed" in stripped.lower()
        ):
            full_content = "✅ Risk Check Passed"

    # Strip our own known node-name prefixes (not arbitrary bracketed text).
    if isinstance(full_content, str):
        full_content = re.sub(r'^\[(RiskManager|DeepReasoning|PortfolioManager|NewsAnalyst|MarketAnalyst)\]:\s*', '', full_content)

    # --- PARSE VERDICT + COMBINE WITH GROUNDING ---
    score, is_compliant, llm_violations = parse_risk_verdict(full_content)
    all_violations = grounding_violations + llm_violations
    if grounding_violations and score > 6:
        score = min(score, 6)  # Grounding errors cap the score

    # Roadmap 2.3 — a degraded-evidence cap, deliberately WEAKER than the
    # grounding cap above and on a separate track. A missing API key is a fact
    # about the world, not misconduct by the advice: it must not force a
    # CRITICAL_FAIL, or an unconfigured data source would stop the advisor
    # answering at all. What it does mean is that the turn cannot score as a
    # clean, fully-evidenced pass — so the ceiling comes down and the verdict
    # keeps its PASS/FAIL shape.
    if data_quality.get("degraded") and score > PROVENANCE_SCORE_CAP:
        score = PROVENANCE_SCORE_CAP

    if is_compliant and not grounding_violations:
        risk_result = "PASS"
    elif score <= 4 or grounding_violations:
        risk_result = "CRITICAL_FAIL"
    else:
        risk_result = "FAIL"

    return JudgeOutcome(
        score=score,
        risk_result=risk_result,
        is_compliant=is_compliant,
        grounding_violations=grounding_violations,
        llm_violations=llm_violations,
        all_violations=all_violations,
        verdict_text=full_content if isinstance(full_content, str) else str(full_content),
        ips_result=ips_result,
        data_quality=data_quality,
    )


def risk_manager_node(state: AgentState):
    """
    Risk & Compliance Manager: Quick sanity check on advice. Kept ultra-brief to save tokens.
    """
    from tools.risk_verdict_log import log_risk_verdict

    llm = judge_llm()  # Deep tier unless AIDLC_JUDGE_TIER says otherwise
    send_status("⚖️ Quick Risk Check...")
    log_event("RiskManager", "Starting risk assessment", {"num_messages": len(state['messages'])})

    # Ghost turns still get audited and logged (the safety trail must not have
    # silent gaps), but the user's query text is not captured.
    is_ghost = bool(state.get("ghost", False))

    # --- REDUNDANCY CHECK: Skip if a Risk Assessment already exists in recent history ---
    # This prevents duplication if the node is called twice in the graph for the same
    # piece of advice. Scan back to the last HumanMessage (the current turn's boundary)
    # rather than a fixed window, so an assessment generated a few messages earlier in
    # the same turn is still caught.
    #
    # IMPORTANT: only match AIMessages actually authored by RiskManager (name=="RiskManager"),
    # not any message whose content happens to contain the verdict markers. The compliance
    # retry gate below injects a correction HumanMessage that embeds the failed verdict's
    # full text (including "⚖️ **Verdict:") so DeepReasoning can see what to fix — if that
    # HumanMessage were allowed to match here, the retried response would never actually be
    # re-audited: on the second pass this correction message becomes the new "last
    # HumanMessage" boundary, its embedded verdict text matches, and this check bypasses the
    # audit unconditionally instead of running it against the revised advice.
    all_messages = state['messages']
    turn_start = None
    for i in range(len(all_messages) - 1, -1, -1):
        if isinstance(all_messages[i], HumanMessage):
            turn_start = i
            break
    # Bound to the current turn (from its HumanMessage). If none is present
    # (e.g. history was summarized away), fall back to the last few messages so
    # we don't scan the whole transcript and trip on a stale block from an old turn.
    recent_messages = all_messages[turn_start:] if turn_start is not None else all_messages[-3:]
    for msg in recent_messages:
        if not (isinstance(msg, AIMessage) and getattr(msg, "name", None) == "RiskManager"):
            continue
        content = str(getattr(msg, "content", ""))
        if "### 🛡️ Risk Assessment" in content or "⚖️ **Verdict:" in content:
            log_event("RiskManager", "Skipping: Risk assessment already present in recent history")
            # Audit trail (Theme 2.1): record the skip so the trail has no
            # silent gaps — the verdict for this turn lives inside the
            # embedding node's output, not a judge pass of its own.
            log_risk_verdict({
                "event": "bypassed",
                "risk_result": "PASS",
                "advice_node": getattr(msg, "name", None),
                "query": "" if is_ghost else _current_turn_query(all_messages),
                "ghost": is_ghost,
            })
            # Return a bypass message so Supervisor knows RiskManager completed
            bypass_msg = AIMessage(content="[RiskManager]: Risk check bypassed (already assessed in Deep Reasoning).", name="RiskManager")
            return {"messages": [bypass_msg], "risk_assessment": "PASS"}

    # Inject user memory (profile, lessons, risk tolerance) so risk checks are personalized
    user_memory_ctx = get_user_context_string()
    portfolio_verification_ctx = _build_portfolio_verification_brief()

    # Grounding evidence for the judge (Rule 8: Source Fraud). THIS turn's tool outputs
    # from both routes they arrive by — the analysts' in-state ToolMessages and the deep
    # path's data_context publication — unioned when the latter is stamped with this
    # turn, so neither half's true figures read as fabricated.
    data_ctx = state.get("data_context", {}) or {}
    tool_execution_ctx = _build_tool_execution_context(
        all_messages,
        data_ctx.get("tool_execution_context"),
        data_ctx.get("tool_execution_turn"),
    )


    # The advice under audit: the last substantive AI message this turn.
    last_advice_text = ""
    advice_node = None  # which node authored the advice under audit (for the verdict trail)
    for msg in reversed(state['messages']):
        if isinstance(msg, AIMessage) and msg.content:
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            if len(content) > 100:  # Only check substantive messages
                last_advice_text = content
                advice_node = getattr(msg, "name", None)
                break

    # Judge context: clean copies of the recent conversation + appended check
    # message. On a compliance-retry pass this drops the judge's own prior
    # verdict and pins the audit to the revised draft — see _build_judge_context.
    retry_count = state.get("risk_retry_count", 0)
    internal_messages = _build_judge_context(all_messages, retry_count)

    # Delegate the judge pass — deterministic grounding + IPS pre-audit, the LLM
    # Risk Judge, and the verdict parse/score-cap — to the shared seam (Theme 2.4;
    # the eval harness calls the same seam on fixtures). Everything below is the
    # production wrapper the seam deliberately excludes: live token streaming is
    # requested via stream=, while the banner, retry gate, and verdict
    # persistence stay here on the node.
    outcome = judge_advice(
        last_advice_text,
        llm=llm,
        judge_messages=internal_messages,
        user_memory_ctx=user_memory_ctx,
        portfolio_verification_ctx=portfolio_verification_ctx,
        tool_execution_ctx=tool_execution_ctx,
        stream=has_stream_callback(),
    )
    grounding_violations = outcome.grounding_violations
    ips_result = outcome.ips_result
    llm_violations = outcome.llm_violations
    all_violations = outcome.all_violations
    score = outcome.score
    is_compliant = outcome.is_compliant
    risk_result = outcome.risk_result
    full_content = outcome.verdict_text
    if grounding_violations:
        log_event("RiskManager", "Grounding audit found violations",
                  {"count": len(grounding_violations), "violations": grounding_violations})

    # If there are grounding violations, append a warning banner to the output
    if grounding_violations:
        banner = "\n\n> ⚠️ **Grounding Violations Detected:**\n"
        for v in grounding_violations:
            banner += f"> - {v}\n"
        full_content += banner

    # Return message - server.py uses the [NodeName]: prefix to identify and potentially strip it
    # Roadmap 2.3 — the user-visible provenance footer. Appended to the DISPLAY
    # message only: `full_content` is what the 2.1 audit trail persists and what
    # parse_risk_verdict already read, and a footer folded into it would show up
    # later as part of the judge's own words.
    footer = _provenance_footer_line(outcome.data_quality)
    final_aimsg = AIMessage(
        content=f"[RiskManager]: \n\n---\n### 🛡️ Risk Assessment\n{full_content}{footer}"
    )
    final_aimsg.name = "RiskManager"

    log_event("RiskManager", "Risk assessment complete", {
        "verdict": str(full_content)[:50],
        "score": score,
        "risk_result": risk_result,
        "violations_count": len(all_violations),
    })

    result_state = {
        "messages": [final_aimsg],
        "risk_assessment": risk_result,
        # Roadmap 2.3 — publish the turn's provenance so the response layer can
        # footer it. Merged into a COPY of the incoming data_context: this key
        # describes one turn, and data_context has no state reducer, so
        # replacing the dict wholesale would drop the heavy path's
        # tool_execution_context and re-open the stale-evidence false-source-fraud
        # this node already fixed once.
        "data_context": {**data_ctx, "data_quality": outcome.data_quality},
    }

    # --- COMPLIANCE RETRY GATE ---
    # On CRITICAL_FAIL with retries remaining, inject a correction directive
    # and increment the counter so after_risk_manager routes to DeepReasoning.
    # On exhausted retries, prepend a warning banner (Option A: ship with warning).
    if risk_result == "CRITICAL_FAIL":
        if retry_count < 1:
            # First failure: prepare for retry
            correction_msg = HumanMessage(
                content=(
                    "<compliance_correction_required>\n"
                    "Your previous response was flagged by the Risk Manager with CRITICAL violations.\n"
                    "You MUST correct the following issues in your revised response:\n\n"
                    f"{full_content}\n\n"
                    "Revise your advice to fix these specific violations. Do NOT repeat the same errors.\n"
                    "</compliance_correction_required>"
                )
            )
            result_state["messages"].append(correction_msg)
            result_state["risk_retry_count"] = retry_count + 1
            log_event("RiskManager", "CRITICAL_FAIL: routing to DeepReasoning for correction retry",
                       {"retry_count": retry_count + 1})

            # The retry's revised draft streams into the SAME visible token pipe as
            # the flagged first draft and this verdict (send_stream/on_token —
            # see api/routers/chat.py), with no separator between them. Without an
            # explicit marker here, the user just sees the answer seemingly repeat
            # itself with no explanation. This note is itself streamed so it lands
            # in the persisted transcript, not just the transient status line.
            if has_stream_callback():
                send_stream(
                    "\n\n---\n⚖️ *First draft didn't clear compliance review — revising below.*\n\n---\n\n"
                )
        else:
            # Exhausted retries: ship with prominent warning banner (Option A)
            warning_banner = (
                "\n\n> [!CAUTION]\n"
                "> ⚠️ **Compliance Review Warning**: This advice was flagged on both attempts. "
                "The following violations could not be automatically resolved:\n"
            )
            for v in all_violations:
                warning_banner += f"> - {v}\n"
            warning_banner += "> Please verify all recommendations independently before acting.\n"

            flagged_msg = AIMessage(
                content=f"[RiskManager]: {warning_banner}",
                name="RiskManager"
            )
            result_state["messages"].append(flagged_msg)
            result_state["risk_retry_count"] = retry_count + 1  # Exceed cap so routing goes to Supervisor
            log_event("RiskManager", "CRITICAL_FAIL: retries exhausted, shipping with warning banner",
                       {"retry_count": retry_count + 1})

    # --- AUDIT TRAIL (Theme 2.1) ---
    # Persist the parsed verdict to per-profile risk_verdicts.jsonl. Written
    # after the retry gate so the record captures what actually happened to
    # this pass, not just its score.
    if risk_result != "CRITICAL_FAIL":
        retry_outcome = "accepted"  # PASS, or soft FAIL shipped as commentary
    elif retry_count < 1:
        retry_outcome = "retry_scheduled"  # correction directive injected
    else:
        retry_outcome = "retries_exhausted"  # shipped with warning banner
    # IPS pre-check summary for the calibration corpus (Theme 2.2): which
    # proposed trades were detected and how the computed rows resolved.
    ips_log_summary = None
    if ips_result["trades"]:
        ips_log_summary = {
            "trades": [t.get("ticker") for t in ips_result["trades"]],
            "fails": len(ips_result["violations"]),
            "rows": ips_result["rows"],
        }
    log_risk_verdict({
        "event": "verdict",
        "score": score,
        "risk_result": risk_result,
        "is_compliant": is_compliant,
        "grounding_violations": grounding_violations,
        "llm_violations": llm_violations,
        "ips_precheck": ips_log_summary,
        "retry_count": retry_count,
        "retry_outcome": retry_outcome,
        "advice_node": advice_node,
        "query": "" if is_ghost else _current_turn_query(all_messages),
        "ghost": is_ghost,
        "verdict_text": full_content if isinstance(full_content, str) else str(full_content),
    })

    return result_state
