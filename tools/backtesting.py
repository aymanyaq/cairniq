from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf
from ta.momentum import RSIIndicator

from tools.exception_logger import log_exceptions


@log_exceptions()
def backtest_strategy(
    strategy_type: str = "rsi",
    symbols: list[str] = None,
    period: str = "2y",
    params: dict[str, Any] = None
) -> dict[str, Any]:
    """
    Unified Backtester for Technical (RSI) and Portfolio (DCA/Lump Sum) strategies.

    Args:
        strategy_type: "rsi", "dca", "lump_sum"
        symbols: List of tickers. For RSI, uses first symbol.
        period: "1y", "2y", "5y", "10y"
        params:
            For RSI: {"buy_threshold": 30, "sell_threshold": 70}
            For DCA/Lump: {"allocations": [0.6, 0.4], "initial_capital": 10000}
    """
    if params is None:
        params = {}
    if symbols is None:
        symbols = []
    try:
        # --- STRATEGY 1: RSI (Technical) ---
        if strategy_type.lower() == "rsi":
            symbol = symbols[0] if symbols else "SPY"
            buy_threshold = params.get("buy_threshold", 30)
            sell_threshold = params.get("sell_threshold", 70)

            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period)

            if df.empty or len(df) < 20:
                return {"error": "Insufficient historical data"}

            # Calculate RSI
            rsi = RSIIndicator(close=df["Close"], window=14).rsi()
            df["RSI"] = rsi

            # Simulate trades
            position = 0
            buy_price = 0
            trades = []

            for i in range(14, len(df)):
                current_rsi = df["RSI"].iloc[i]
                current_price = df["Close"].iloc[i]

                if position == 0 and current_rsi < buy_threshold:
                    position = 1
                    buy_price = current_price
                    trades.append({"action": "BUY", "price": current_price, "rsi": current_rsi})

                elif position == 1 and current_rsi > sell_threshold:
                    position = 0
                    profit_pct = ((current_price - buy_price) / buy_price) * 100
                    trades.append({"action": "SELL", "price": current_price, "rsi": current_rsi, "profit_pct": profit_pct})

            completed_trades = [t for t in trades if t["action"] == "SELL"]
            if not completed_trades:
                 return {"message": "No trades triggered."}

            total_return = sum(t["profit_pct"] for t in completed_trades)
            avg_return = total_return / len(completed_trades)

            return {
                "symbol": symbol.upper(),
                "strategy": f"RSI Buy<{buy_threshold}, Sell>{sell_threshold}",
                "period": period,
                "total_trades": len(completed_trades),
                "total_return": f"{total_return:.2f}%",
                "avg_return_per_trade": f"{avg_return:.2f}%",
                "verdict": "Effective" if avg_return > 2 else "Mixed"
            }

        # --- STRATEGY 2: PORTFOLIO ALLOCATION (DCA / Lump Sum) ---
        elif strategy_type.lower() in ["lump_sum", "dca"]:
            allocations = params.get("allocations", [1.0/len(symbols)]*len(symbols))
            initial_capital = params.get("initial_capital", 100000)

            # Fetch Data
            hist_data = {}
            for sym in symbols:
                ticker = yf.Ticker(sym.strip().upper())
                hist = ticker.history(period=period)
                if not hist.empty:
                    hist_data[sym] = hist["Close"]

            if not hist_data:
                return {"error": "No data found"}

            df = pd.DataFrame(hist_data).dropna()

            weights = np.array(allocations)
            if weights.sum() != 1.0: weights = weights / weights.sum()

            if strategy_type == "lump_sum":
                shares = (initial_capital * weights) / df.iloc[0]
                portfolio_values = df.dot(shares)

            elif strategy_type == "dca":
                # DCA Logic: Invest 1/Nth of capital each month
                monthly_invest = initial_capital / 60 if "5y" in period else initial_capital / 12
                monthly_df = df.resample('ME').last() # Updated for pandas future warning

                cum_shares = np.zeros(len(symbols))
                dca_values = []

                for _, row in monthly_df.iterrows():
                    shares_bought = (monthly_invest * weights) / row.values
                    cum_shares += shares_bought
                    dca_values.append(np.dot(row.values, cum_shares))

                portfolio_values = pd.Series(dca_values, index=monthly_df.index)

            portfolio_values.iloc[0]
            end_val = portfolio_values.iloc[-1]
            total_ret = ((end_val - initial_capital) / initial_capital) * 100

            return {
                "strategy": strategy_type.upper(),
                "period": period,
                "final_value": f"${end_val:,.2f}",
                "total_return": f"{total_ret:.2f}%",
                "note": "DCA assumes spreading capital over period."
            }

        else:
            return {"error": f"Unknown strategy type: {strategy_type}"}

    except Exception as e:
        return {"error": f"Backtest failed: {str(e)}"}

if __name__ == "__main__":
    # Test
    print(backtest_strategy("rsi", ["AAPL"]))
    print(backtest_strategy("dca", ["AAPL", "MSFT"], params={"allocations": [0.5, 0.5]}))
