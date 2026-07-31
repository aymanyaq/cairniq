from typing import Any

import yfinance as yf

from tools.cache import cached
from tools.exception_logger import log_exceptions
from tools.yf_utils import dividend_yield_display


@cached(key_func=lambda symbols, mode="fundamentals", benchmark="SPY": f"compare:{','.join(sorted(s.upper() for s in symbols))}:{mode}:{benchmark.upper()}")
@log_exceptions()
def compare_assets(
    symbols: list[str],
    mode: str = "fundamentals",
    benchmark: str = "SPY"
) -> dict[str, Any]:
    """
    Unified Asset Comparator.
    Args:
        symbols: List of tickers (e.g. ['AAPL', 'MSFT'])
        mode:
            'fundamentals': PE, Market Cap, Dividend Yield
            'performance': Relative Strength vs Benchmark (1m/3m/1y)
        benchmark: Comparison ticker for performance mode.
    """
    try:
        symbols = [s.strip().upper() for s in symbols]

        # --- MODE 1: FUNDAMENTALS ---
        if mode == "fundamentals":
            comparison = []
            for sym in symbols:
                try:
                    ticker = yf.Ticker(sym)
                    info = {}
                    try:
                        info = ticker.info or {}
                    except Exception:
                        info = {}

                    price = info.get('currentPrice')
                    if not price:
                        # Fallback 1: fast_info
                        try:
                            price = ticker.fast_info.get('lastPrice')
                        except Exception:
                            price = None

                    if not price:
                        # Fallback 2: download
                        try:
                            hist = yf.download(sym, period="1d", progress=False, threads=False)
                            if not hist.empty:
                                # Cast: .iloc[-1] is np.float64 and leaks as
                                # "np.float64(...)" when the dict is str()-rendered.
                                price = float(hist['Close'].iloc[-1])
                        except Exception:
                            price = None

                    # Enhanced Fundamental Fallbacks
                    pe = info.get("trailingPE") or info.get("forwardPE")
                    mcap_raw = info.get("marketCap") or info.get("totalAssets") or info.get("netAssets")
                    mcap_b = round(mcap_raw / 1e9, 1) if mcap_raw else None

                    comparison.append({
                        "symbol": sym,
                        "price": price,
                        "pe_ratio": pe,
                        "market_cap_B": mcap_b,
                        # Was `dividendYield * 100`, which renders a 0.31% payer
                        # as "31.00%". This tool's whole job is putting tickers
                        # side by side, so whenever one name carried the fraction
                        # field and another did not, it compared a real yield
                        # against a 100x one in the same table.
                        "dividend_yield": dividend_yield_display(info),
                    })
                except Exception as e:
                    comparison.append({"symbol": sym, "error": f"Gap: {str(e)}"})

            return {
                "mode": "fundamentals",
                "comparison": comparison,
                "note": "PE = Price/Earnings ratio. Primary data via yfinance."
            }

        # --- MODE 2: PERFORMANCE (Relative Strength) ---
        elif mode == "performance":
            fetch_list = symbols + [benchmark]
            # Use simple download
            data = yf.download(fetch_list, period="1y", progress=False)

            # Handle MultiIndex
            close = data["Close"] if "Close" in data else data

            rankings = []
            if benchmark not in close.columns:
                return {"error": "Benchmark data missing"}

            bench_curr = close[benchmark].iloc[-1]
            bench_start = close[benchmark].iloc[0]
            bench_ret = (bench_curr / bench_start) - 1

            for sym in symbols:
                if sym not in close.columns: continue

                prices = close[sym].dropna()
                if prices.empty: continue

                curr = prices.iloc[-1]
                start = prices.iloc[0]
                ret = (curr / start) - 1
                rel_perf = ret - bench_ret

                rankings.append({
                    "symbol": sym,
                    "return_1y": f"{ret*100:.1f}%",
                    "vs_benchmark": f"{rel_perf*100:.1f}%",
                    "status": "Leader" if rel_perf > 0 else "Laggard"
                })

            rankings.sort(key=lambda x: float(x["vs_benchmark"].strip("%")), reverse=True)
            return {
                "mode": "performance",
                "benchmark": benchmark,
                "benchmark_return": f"{bench_ret*100:.1f}%",
                "rankings": rankings
            }

        else:
            return {"error": f"Unknown mode: {mode}"}

    except Exception as e:
        return {"error": f"Comparison failed: {str(e)}"}
