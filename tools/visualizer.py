"""
Portfolio Visualizer
Generates ASCII charts for checking trends in the terminal.
Supports basic price history and moving average overlays.
"""
import json
from typing import Any

import yfinance as yf
from asciichartpy import plot

from agent.utils import safe_print
from tools.exception_logger import log_exceptions

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

@log_exceptions()
def generate_ascii_chart(symbol: str, period: str = "3mo") -> str:
    """
    Generates an ASCII chart for the stock price history.
    Includes simple Moving Average overlay if data permits.
    """
    try:
        ticker = yf.Ticker(symbol)

        # Valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
        hist = ticker.history(period=period)

        if hist.empty:
            return f"No data found for {symbol}"

        prices = hist["Close"].tolist()

        if not prices:
             return f"No price data available for {symbol}"

        # Downsample if too many points (limit fit to standard terminal width ~80-100 cols)
        # 40 chars + 9 chars for labels = ~50 chars total, fitting safely inside ANY panel
        max_width = 40
        if len(prices) > max_width:
            step = len(prices) // max_width + 1
            prices = prices[::step]

        # Basic stats for legend
        start_price = prices[0]
        end_price = prices[-1]
        change_pct = ((end_price - start_price) / start_price) * 100

        # Chart configuration
        config = {
            'height': 10, # Slightly shorter height
            'format': '{:8.2f}'
        }

        chart = plot(prices, config)

        # Post-process to Safe ASCII to prevent terminal wrapping issues
        # Replace complex box-drawing chars with simple ASCII
        replacements = {
            '┤': '|', '│': '|', '─': '-',
            '╭': '+', '╮': '+', '╰': '+', '╯': '+',
            '┼': '+'
        }
        for old, new in replacements.items():
            chart = chart.replace(old, new)

        trend_arrow = "UP" if end_price >= start_price else "DOWN"

        header = f"CHART {symbol.upper()} ({period}): ${end_price:.2f} ({trend_arrow} {change_pct:+.2f}%)"

        return f"\n{header}\n{chart}\n"

    except Exception as e:
        return f"Error generating chart for {symbol}: {e}"

@log_exceptions()
def generate_interactive_chart(symbol: str, period: str = "3mo") -> str:
    """Generate Plotly figure and return as JSON string."""
    ticker = yf.Ticker(symbol)
    hist = ticker.history(period=period)

    if hist.empty:
        raise ValueError("No data")

    fig = go.Figure(data=[go.Candlestick(x=hist.index,
                open=hist['Open'],
                high=hist['High'],
                low=hist['Low'],
                close=hist['Close'])])

    fig.update_layout(title=f'{symbol} Price ({period})', xaxis_rangeslider_visible=False)
    # Convert to JSON
    return json.dumps(fig.to_dict(), default=str)

@log_exceptions()
def generate_price_chart(symbol: str, period: str = "3mo") -> str:
    """Main entry point: Tries Plotly JSON first, falls back to ASCII."""
    if PLOTLY_AVAILABLE:
        try:
            json_str = generate_interactive_chart(symbol, period)
            # Return special token for nodes.py to intercept
            return f"[PLOTLY_JSON:{json_str}]"
        except Exception as e:
            safe_print(f"Plotly generation failed: {e}")

    return generate_ascii_chart(symbol, period)

@log_exceptions()
def generate_monte_carlo_chart(sim_data: dict[str, Any]) -> str:
    """
    Generate a Plotly chart for Monte Carlo simulation results.
    Args:
        sim_data: The dictionary returned by run_monte_carlo(), containing 'charts' key.
    """
    if not PLOTLY_AVAILABLE:
        return ""

    try:
        charts = sim_data.get("charts", {})
        years = charts.get("years", [])
        p10 = charts.get("p10", [])
        p50 = charts.get("p50", [])
        p90 = charts.get("p90", [])

        if not years:
            return ""

        fig = go.Figure()

        # 90th Percentile (Best Case)
        fig.add_trace(go.Scatter(
            x=years, y=p90,
            mode='lines',
            name='90th Percentile (Best Case)',
            line=dict(width=0),
            showlegend=False
        ))

        # 10th Percentile (Worst Case) - Fill to 90th
        fig.add_trace(go.Scatter(
            x=years, y=p10,
            mode='lines',
            name='Range (10th-90th)',
            line=dict(width=0),
            fill='tonexty', # Fill to previous trace (90th)
            fillcolor='rgba(0, 100, 80, 0.2)',
            showlegend=True
        ))

        # Median (Base Case)
        fig.add_trace(go.Scatter(
            x=years, y=p50,
            mode='lines',
            name='Median Outcome',
            line=dict(color='rgb(0,176,246)', width=3)
        ))

        fig.update_layout(
            title="Monte Carlo Simulation (Wealth Projection)",
            xaxis_title="Years",
            yaxis_title="Portfolio Value ($)",
            yaxis=dict(tickprefix="$"),
            hovermode="x unified",
            template="plotly_dark"
        )

        json_str = json.dumps(fig.to_dict(), default=str)
        return f"[PLOTLY_JSON:{json_str}]"

    except Exception as e:
        safe_print(f"Monte Carlo Chart Error: {e}")
        return ""

        return ""

@log_exceptions()
def generate_options_chart(symbol: str) -> str:
    """
    Generates a Volume vs Open Interest chart for the nearest options expiry.
    Visualizes 'Unusual Activity' (Volume > OI) and Put/Call walls.
    """
    try:
        import json

        import plotly.graph_objects as go
        import yfinance as yf

        ticker = yf.Ticker(symbol)
        if not ticker.options: return ""

        # Get nearest expiry
        expiry = ticker.options[0]
        chain = ticker.option_chain(expiry)
        calls = chain.calls
        puts = chain.puts

        # Filter near-the-money (within 15% of price) to keep chart readable
        current_price = ticker.info.get("currentPrice", 0)

        # Fallback price if info fails
        if current_price == 0:
            hist = ticker.history(period="1d")
            if not hist.empty:
                current_price = hist["Close"].iloc[-1]
            elif not calls.empty:
                current_price = calls['strike'].mean()

        lower_bound = current_price * 0.85
        upper_bound = current_price * 1.15

        calls = calls[(calls['strike'] >= lower_bound) & (calls['strike'] <= upper_bound)]
        puts = puts[(puts['strike'] >= lower_bound) & (puts['strike'] <= upper_bound)]

        fig = go.Figure()

        # Calls Volume
        fig.add_trace(go.Bar(
            x=calls['strike'],
            y=calls['volume'],
            name='Call Vol',
            marker_color='green',
            opacity=0.6,
            legendgroup="Calls"
        ))

        # Puts Volume
        fig.add_trace(go.Bar(
            x=puts['strike'],
            y=puts['volume'],
            name='Put Vol',
            marker_color='red',
            opacity=0.6,
            legendgroup="Puts"
        ))

        # Open Interest Lines (to compare)
        fig.add_trace(go.Scatter(
            x=calls['strike'],
            y=calls['openInterest'],
            name='Call OI',
            mode='lines+markers',
            line=dict(color='darkgreen', dash='dot', width=2),
            marker=dict(size=6),
            legendgroup="Calls"
        ))

        fig.add_trace(go.Scatter(
            x=puts['strike'],
            y=puts['openInterest'],
            name='Put OI',
            mode='lines+markers',
            line=dict(color='darkred', dash='dot', width=2),
            marker=dict(size=6),
            legendgroup="Puts"
        ))

        fig.update_layout(
            title=f"Options Flow: {symbol} (Exp: {expiry})<br><sub>Bars=Volume, Lines=Open Interest. High Bars > Lines = Unusual Accumulation</sub>",
            xaxis_title="Strike Price",
            yaxis_title="Contracts",
            barmode='group',
            template="plotly_dark",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        json_str = json.dumps(fig.to_dict(), default=str)
        return f"[PLOTLY_JSON:{json_str}]"

    except Exception as e:
        return f"Error generating chart: {e}"

if __name__ == "__main__":
    print(generate_price_chart("AAPL", "1y")[:100] + "...")
