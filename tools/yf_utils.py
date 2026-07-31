"""
Utility wrapper for yfinance to handle threading/I/O issues, and the readers for
``Ticker.info`` fields whose units are not obvious.

The units half exists because of one defect found on 2026-07-30 and the nine
places it had been copied to. ``Ticker.info`` carries the same quantity under
several keys in DIFFERENT units, and every reader in this codebase had
independently guessed which unit it was looking at. The guesses disagreed, so
the same holding was reported at 0.31%, 31% and 0.00% depending on which panel
asked. This is the module that fetches ``info``, so it is the module that owns
how to read it.

The rule the readers encode: **never normalise by magnitude, only by field
name.** The old idiom was ``if raw > 1.0: raw /= 100`` — a heuristic that reads
the number to decide what the number means. It is correct for a 4% yielder and
silently wrong for a sub-1% one, which is why it survived for months: the high
yielders looked right, so nobody checked the low ones.
"""
import time
from typing import Any

import yfinance as yf

from tools.exception_logger import log_exceptions

# AUTHORED CONSTANT. Above this, a "yield" is a units error rather than a
# high-income signal. 25% sits above every plausible equity, REIT or credit
# yield and well below the 32%/93% figures the percent-vs-fraction bug produced
# for AAPL and MSFT. See `dividend_yield_fraction`.
IMPLAUSIBLE_YIELD = 0.25


@log_exceptions()
def safe_yf_call(func, max_retries=3, initial_delay=0.5):
    """
    Wrapper to handle yfinance 'I/O operation on closed file' race condition
    and Yahoo Finance rate limiting (401 Invalid Crumb errors).

    Args:
        func: A callable (lambda or function) that makes the yfinance call
        max_retries: Number of retry attempts
        initial_delay: Initial delay in seconds (doubles each retry)

    Returns:
        The result of func(), or None if all retries fail
    """
    for attempt in range(max_retries):
        try:
            return func()
        except ValueError as e:
            error_str = str(e).lower()
            if "closed file" in error_str and attempt < max_retries - 1:
                time.sleep(initial_delay * (attempt + 1))
                continue
            # Handle Yahoo Finance crumb errors
            if "401" in error_str or "unauthorized" in error_str or "crumb" in error_str:
                if attempt < max_retries - 1:
                    # Longer delay for auth errors
                    time.sleep(initial_delay * (attempt + 2))
                    continue
            raise
        except OSError as e:
            if "closed file" in str(e).lower() and attempt < max_retries - 1:
                time.sleep(initial_delay * (attempt + 1))
                continue
            raise
        except Exception as e:
            error_str = str(e).lower()
            # Catch any other exception that might contain "closed file" or auth errors
            if "closed file" in error_str and attempt < max_retries - 1:
                time.sleep(initial_delay * (attempt + 1))
                continue
            # Handle Yahoo Finance crumb/auth errors
            if ("401" in error_str or "unauthorized" in error_str or "crumb" in error_str) and attempt < max_retries - 1:
                time.sleep(initial_delay * (attempt + 2))
                continue
            raise
    return None


@log_exceptions()
def get_ticker_safe(symbol: str):
    """Get a yf.Ticker object safely."""
    return yf.Ticker(symbol)


@log_exceptions()
def get_history_safe(symbol: str, period: str = "1mo"):
    """Get historical data with retry logic."""
    ticker = yf.Ticker(symbol)
    return safe_yf_call(lambda: ticker.history(period=period))


@log_exceptions()
def get_info_safe(symbol: str):
    """Get ticker info with retry logic."""
    ticker = yf.Ticker(symbol)
    return safe_yf_call(lambda: ticker.info)


@log_exceptions()
def get_news_safe(symbol: str):
    """Get ticker news with retry logic."""
    ticker = yf.Ticker(symbol)
    return safe_yf_call(lambda: ticker.news)


@log_exceptions()
def download_safe(symbols, period="1y", threads=False, **kwargs):
    """
    Wrapper for yf.download with threading disabled by default.

    Note: threads=False is critical for avoiding I/O errors in concurrent environments.
    """
    return safe_yf_call(lambda: yf.download(symbols, period=period, threads=threads, progress=False, **kwargs))


# ---------------------------------------------------------------------------
# Reading `info` — the fields that do not agree on units
# ---------------------------------------------------------------------------

def dividend_yield_fraction(info: dict[str, Any]) -> float:
    """Dividend yield as a FRACTION (0.0031 for 0.31%), from fields that disagree.

    **Found 2026-07-30, on the first live read of the asset-location panel.** The
    provider's yield fields are in DIFFERENT UNITS and always have been. Captured
    from ``yf.Ticker(...).info`` under yfinance 1.5.1, which is where the numbers
    in the pinning tests come from:

        AAPL   dividendYield 0.32   trailingAnnualDividendYield 0.003058  yield None
        ENB.TO dividendYield 5.0    trailingAnnualDividendYield 0.048699  yield None
        BND    dividendYield 3.95   trailingAnnualDividendYield None      yield 0.0395
        BRK-B  dividendYield None   trailingAnnualDividendYield 0.0       yield None

    ``dividendYield`` is a PERCENT; the other two are FRACTIONS. The previous code
    read the percent field first and normalised it with ``if raw > 1.0: /= 100`` —
    a guard that fires for a 4% yielder and CANNOT fire for a sub-1% one. So
    AAPL's 0.32% was read as 32%, and since the high-income test triggers at 3%,
    every dividend payer in a taxable account was flagged as tax drag. High
    yielders were scored correctly and low yielders were wrong by 100x, which is
    why nothing looked broken until the numbers reached a screen.

    Order of preference is by how KNOWABLE the unit is, not by which field is
    most often populated:
      1. ``trailingAnnualDividendYield`` — unambiguously a fraction. It is absent
         on ETFs, which is what the next entry is for.
      2. ``yield`` — the fund field, also a fraction. This is the ONLY fraction an
         ETF carries (see BND above), so dropping it would push every fund onto
         the percent field.
      3. ``dividendYield`` — a percent today, and it was a fraction in older
         yfinance releases. Used last, converted, and clamped.

    The clamp is the part that survives the next units change: a yield above
    ``IMPLAUSIBLE_YIELD`` is not a high-income signal, it is a unit error, and it
    returns 0.0 (unknown) rather than being reported. Asserting a 32% yield is
    worse than asserting none.

    Missing, unparseable, genuinely-zero and implausible input all return 0.0 and
    are NOT distinguished, so **0.0 means "no number to report", never "this pays
    nothing"** — do not render it as "0.00%". Use `dividend_yield_display`.
    """
    for key in ("trailingAnnualDividendYield", "yield"):
        value = info.get(key)
        try:
            fraction = float(value)
        except (TypeError, ValueError):
            continue
        if fraction > 0:
            return fraction if fraction <= IMPLAUSIBLE_YIELD else 0.0

    try:
        percent = float(info.get("dividendYield"))
    except (TypeError, ValueError):
        return 0.0
    if percent <= 0:
        return 0.0
    fraction = percent / 100.0
    return fraction if fraction <= IMPLAUSIBLE_YIELD else 0.0


def dividend_yield_display(info: dict[str, Any]) -> str:
    """Dividend yield as a labelled percent string, e.g. ``"0.31%"``, or ``"N/A"``.

    For payloads whose only consumer is a screen or the model. It exists so that
    no caller writes ``* 100`` again: every site that had the units wrong got them
    wrong at the multiplication, not at the read. A string also carries its own
    unit, which a bare ``"dividend_yield": 0.32`` never did — that field was read
    as 32% by anything downstream that assumed a fraction.

    ``"N/A"`` covers all four zero cases from `dividend_yield_fraction`. Printing
    "0.00%" would assert a non-payer, and a clamped unit error is not evidence of
    one.
    """
    fraction = dividend_yield_fraction(info)
    return f"{fraction * 100:.2f}%" if fraction > 0 else "N/A"
