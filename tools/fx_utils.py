import os
from typing import Any

import pandas as pd
import yfinance as yf

from tools.cache import cached
from tools.exception_logger import log_exceptions

# Yahoo ticker suffix -> trading currency of the quote. Yahoo always quotes a
# listing in its exchange's currency, so the suffix — not the holding record —
# is the authoritative source for the currency of a *price series*.
_SUFFIX_CURRENCY = {
    ".TO": "CAD", ".V": "CAD", ".VN": "CAD", ".CN": "CAD", ".NE": "CAD",
    ".L": "GBP",
    ".DE": "EUR", ".F": "EUR", ".PA": "EUR", ".MI": "EUR", ".AS": "EUR",
    ".BR": "EUR", ".MC": "EUR",
    ".AX": "AUD",
    ".T": "JPY",
}

# Crypto pairs quote in the currency after the dash (BTC-USD, ETH-CAD, ...).
_CRYPTO_QUOTE_CURRENCY = {
    "-USD": "USD", "-CAD": "CAD", "-EUR": "EUR",
    "-GBP": "GBP", "-AUD": "AUD", "-JPY": "JPY",
}


def infer_symbol_currency(symbol: str, default: str = "USD") -> str:
    """Trading currency of a Yahoo price series, inferred from the ticker suffix.

    Offline and deterministic. Note this is the currency the *quotes* arrive in,
    which can differ from the account currency a holding sits in.
    """
    t = (symbol or "").upper().strip()
    if not t:
        return default
    for suffix, cur in _SUFFIX_CURRENCY.items():
        if t.endswith(suffix):
            return cur
    for suffix, cur in _CRYPTO_QUOTE_CURRENCY.items():
        if t.endswith(suffix):
            return cur
    return default


@log_exceptions()
def get_fx_rate_series(currencies: list[str], base_currency: str, period: str = "1y") -> pd.DataFrame:
    """Historical FX close series converting each currency into ``base_currency``.

    Returns a DataFrame indexed by date with one column per foreign currency,
    holding the rate "units of base per 1 unit of that currency". Currencies
    equal to the base are omitted. A currency whose pair can't be fetched
    (direct ``{CUR}{BASE}=X`` or reciprocal of the inverse pair) is simply
    missing from the columns — callers must treat absent columns as
    unavailable rather than assume a rate.
    """
    base = (base_currency or "USD").upper().strip()
    wanted = sorted({(c or "").upper().strip() for c in currencies} - {base, ""})
    if not wanted:
        return pd.DataFrame()

    direct = {cur: f"{cur}{base}=X" for cur in wanted}
    inverse = {cur: f"{base}{cur}=X" for cur in wanted}

    try:
        from tools.yf_utils import download_safe
        data = download_safe(list(direct.values()) + list(inverse.values()), period=period)
        if data is None or len(data) == 0:
            return pd.DataFrame()

        if "Adj Close" in data:
            closes = data["Adj Close"]
        elif "Close" in data:
            closes = data["Close"]
        else:
            closes = data
        if isinstance(closes, pd.Series):
            closes = closes.to_frame(name=list(direct.values())[0])

        out: dict[str, pd.Series] = {}
        for cur in wanted:
            series = closes.get(direct[cur])
            if series is not None and series.notna().any():
                out[cur] = series.dropna()
                continue
            inv = closes.get(inverse[cur])
            if inv is not None and inv.notna().any():
                inv = inv.dropna()
                inv = inv[inv != 0]
                if not inv.empty:
                    out[cur] = 1.0 / inv
        return pd.DataFrame(out)
    except Exception:
        return pd.DataFrame()


@cached(key_func=lambda: "fx_usd_cad")
@log_exceptions()
def get_usd_cad_rate() -> float:
    """
    Get the current USD to CAD exchange rate.
    Returns: Float (e.g. 1.35 means 1 USD = 1.35 CAD)
    """
    try:
        # standard symbol for USD/CAD
        ticker = yf.Ticker("cad=x")
        # For some reason yfinance might return CAD=X as USD/CAD or CAD/USD depending on the day.
        # "CAD=X" typically means "USD/CAD" in Yahoo Finance.
        hist = ticker.history(period="5d", timeout=40)

        if not hist.empty:
            return float(hist["Close"].iloc[-1])
        return float(os.environ.get("USD_TO_CAD", "1.44"))
    except Exception:
        return float(os.environ.get("USD_TO_CAD", "1.44"))

@log_exceptions()
def analyze_fx_impact(holdings: dict[str, float], base_currency: str = "CAD", currencies: dict[str, str] | None = None) -> dict[str, Any]:
    """
    Analyze the impact of FX fluctuations on the portfolio.

    Args:
        holdings: Dict of {Symbol: Value_in_Native_Currency}
                  (Note: Value often comes in USD for US stocks, CAD for CA stocks)
        base_currency: The user's home currency (e.g. "CAD").
        currencies: Optional dict of {Symbol: Currency} with the actual recorded
                    currency for each holding (from the CSV/broker sync). When a
                    symbol is present here it takes priority over the ticker-suffix
                    guess below — the guess only exists for callers that don't have
                    verified currency data (e.g. private/manual assets like a named
                    pension fund have no ticker suffix to go on and would otherwise
                    silently default to USD).

    Returns:
        Dict with FX exposure analysis and sensitivity.
    """
    from tools.portfolio_csv import get_exchange_rate

    usd_cad = get_exchange_rate("USD", "CAD")
    base_currency = (base_currency or "CAD").upper().strip()
    currencies = currencies or {}

    def guess_currency(ticker_name: str) -> str:
        t_up = ticker_name.upper()
        suffix_map = {
            ".TO": "CAD", ".V": "CAD", ".VN": "CAD", ".CN": "CAD",
            ".L": "GBP",
            ".DE": "EUR", ".PA": "EUR", ".MI": "EUR", ".AS": "EUR",
            ".AX": "AUD",
            ".T": "JPY",
            "CAD": "CAD", "USD": "USD", "EUR": "EUR", "GBP": "GBP"
        }
        for suffix, cur in suffix_map.items():
            if suffix in t_up:
                return cur
        return "USD" # default assumption

    total_equity_base = 0.0
    exposure_by_currency = {}
    details = []

    for symbol, value in holdings.items():
        # Determine currency of asset: prefer the verified recorded currency,
        # fall back to the ticker-suffix guess only when it's unavailable.
        currency = currencies.get(symbol) or guess_currency(symbol)

        # Get rate to convert to base currency
        rate_to_base = get_exchange_rate(currency, base_currency)
        val_base = value * rate_to_base

        exposure_by_currency[currency] = exposure_by_currency.get(currency, 0.0) + val_base
        total_equity_base += val_base

        # Build detail record
        detail = {
            "symbol": symbol,
            "currency": currency,
            f"value_{currency.lower()}": value,
            "value_base": val_base
        }
        # Keep value_cad/value_usd for backward compatibility
        rate_to_cad = get_exchange_rate(currency, "CAD")
        rate_to_usd = get_exchange_rate(currency, "USD")
        detail["value_cad"] = value * rate_to_cad
        detail["value_usd"] = value * rate_to_usd
        details.append(detail)

    # Foreign exposure is anything not in base_currency
    exposure_foreign_base = sum(val for cur, val in exposure_by_currency.items() if cur != base_currency)

    # Sensitivity Analysis: base currency strengthening/weakening 5% against all foreign holdings
    impact_strengthen = exposure_foreign_base * 0.95 - exposure_foreign_base
    impact_weaken = exposure_foreign_base * 1.05 - exposure_foreign_base

    # Compute percentages
    pct_exposure = {cur: round((val / total_equity_base * 100), 1) if total_equity_base else 0.0
                    for cur, val in exposure_by_currency.items()}

    # Backward compatible fields
    exposure_by_currency.get("CAD", 0.0)
    exposure_usd_base = exposure_by_currency.get("USD", 0.0)

    # Convert values to CAD/USD for return structure
    rate_base_to_cad = get_exchange_rate(base_currency, "CAD")

    return {
        "rate_usd_cad": round(usd_cad, 4),
        "base_currency": base_currency,
        "total_equity_base": round(total_equity_base, 2),
        "total_equity_cad": round(total_equity_base * rate_base_to_cad, 2),
        "exposure_usd_cad_value": round(exposure_usd_base * rate_base_to_cad, 2), # CAD value of USD holdings
        "exposure_cad_pct": pct_exposure.get("CAD", 0.0),
        "exposure_usd_pct": pct_exposure.get("USD", 0.0),
        "exposure_breakdown_pct": pct_exposure,
        "sensitivity": {
            "strengthens_5pct": {
                "portfolio_impact": round(impact_strengthen, 2),
                "message": f"If {base_currency} strengthens 5%, you LOSE ${abs(impact_strengthen):,.0f} {base_currency} in foreign value."
            },
            "weakens_5pct": {
                "portfolio_impact": round(impact_weaken, 2),
                "message": f"If {base_currency} weakens 5%, you GAIN ${impact_weaken:,.0f} {base_currency} in foreign value."
            },
            # Backward compatibility keys
            "cad_strengthens_5pct": {
                "rate": round(usd_cad * 0.95, 4),
                "portfolio_impact": round(impact_strengthen * rate_base_to_cad, 2),
                "message": f"If CAD strengthens 5% (to {usd_cad * 0.95:.2f}), you LOSE ${abs(impact_strengthen * rate_base_to_cad):,.0f} in value (CAD terms)."
            },
            "cad_weakens_5pct": {
                "rate": round(usd_cad * 1.05, 4),
                "portfolio_impact": round(impact_weaken * rate_base_to_cad, 2),
                "message": f"If CAD weakens 5% (to {usd_cad * 1.05:.2f}), you GAIN ${impact_weaken * rate_base_to_cad:,.0f} in value (CAD terms)."
            }
        },
        "details": details
    }


@log_exceptions()
def analyze_my_portfolio_fx(base_currency: Any = None) -> dict[str, Any]:
    """
    Analyze the FX impact on the user's specific portfolio.
    Automatically loads data from my_portfolio.csv / Questrade.
    """
    try:
        if base_currency is None:
            try:
                from tools.memory import get_profile_base_currency
                base_currency = get_profile_base_currency()
            except Exception:
                import os
                base_currency = os.environ.get("BASE_CURRENCY") or os.environ.get("CAIRNIQ_BASE_CURRENCY") or "USD"

        # Better approach: Import get_portfolio_summary which already calculates current values
        from tools.portfolio_csv import get_portfolio_summary
        summary = get_portfolio_summary()

        if "error" in summary:
             return {"error": summary["error"]}

        # Aggregate by symbol rather than overwrite: the same ticker can appear in
        # multiple accounts (e.g. a pension fund split across DC/TFSA sleeves, or a
        # stock held in both RRSP and TFSA) — a plain dict assignment here silently
        # dropped every occurrence but the last, undercounting large positions.
        raw_holdings: dict[str, float] = {}
        currencies: dict[str, str] = {}
        for h in summary.get("holdings", []):
            sym = h.get("symbol")
            try:
                price = float(h.get("current_price", "0").replace("$","").replace(",",""))
                shares = float(h.get("shares", 0))
                val = price * shares
                raw_holdings[sym] = raw_holdings.get(sym, 0.0) + val
                currencies[sym] = h.get("currency") or currencies.get(sym, "USD")
            except Exception:
                continue

        return analyze_fx_impact(raw_holdings, base_currency, currencies=currencies)

    except Exception as e:
        return {"error": f"Portfolio FX Analysis failed: {str(e)}"}

if __name__ == "__main__":
    # Test
    # print(analyze_my_portfolio_fx())
    pass
