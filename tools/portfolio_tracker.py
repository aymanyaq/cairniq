"""
Portfolio Performance Tracker
Tracks total portfolio value over time to enable historical performance analysis.
"""
import csv
import os
from datetime import date

import pandas as pd

from agent.utils import safe_print
from tools.exception_logger import log_exceptions
from tools.user_profile import get_data_path, is_demo_mode


@log_exceptions()
def get_history_file():
    """Returns the profile-specific history file path."""
    is_demo = is_demo_mode()
    filename = "demo_portfolio_history.csv" if is_demo else "portfolio_history.csv"
    return get_data_path(filename)

@log_exceptions()
def _init_history_file():
    """Initialize CSV with headers if it doesn't exist."""
    history_file = get_history_file()
    if not os.path.exists(history_file):
        with open(history_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["date", "total_value_cad", "total_value_usd", "invested_cad", "invested_usd", "percent_return"])

@log_exceptions()
def snapshot_portfolio(force=False):
    """
    Take a daily snapshot of the portfolio value.
    Idempotent: Only records one entry per day.
    Checks for existing entry today to avoid expensive fetching.
    """
    today_str = date.today().isoformat()

    history_file = get_history_file()

    # Check if today already exists to avoid expensive fetches
    if not force and os.path.exists(history_file):
        try:
            df = pd.read_csv(history_file)
            if not df.empty and today_str in df['date'].astype(str).values:
                safe_print(f"✅ Snapshot already exists for {today_str}. Skipping to save time.")
                return
        except Exception as e:
            safe_print(f"Optional check failed (continuing): {e}")

    try:
        from tools.portfolio_csv import get_portfolio_summary

        # Get current stats - MOVED DOWN HERE to avoid expensive fetch if already exists!
        summary = get_portfolio_summary()
        if "error" in summary:
            safe_print(f"Skipping snapshot due to error: {summary['error']}")
            return

        # Extract metrics
        val_cad = summary.get("total_value_cad", 0)
        val_usd = summary.get("total_value_usd", 0)
        inv_cad = summary.get("total_invested_cad", val_cad) # Fallback if missing
        inv_usd = summary.get("total_invested_usd", val_usd)

        # Calculate invested if not present (back-calculate from return if needed, or just use 0)
        # portfolio_csv provides total_gain_loss_usd/cad, so invested = value - gain/loss
        if inv_usd == 0 and val_usd != 0:
             gain_loss = summary.get("total_gain_loss_usd", 0)
             inv_usd = val_usd - gain_loss

        # ROI
        pct_return = summary.get("percent_return", 0)

        _init_history_file()

        # Read existing to check for today
        rows = []
        already_exists = False

        if os.path.exists(history_file):
            with open(history_file) as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            for row in rows:
                if row["date"] == today_str:
                    already_exists = True
                    break

        if already_exists and not force:
            safe_print(f"📸 Snapshot already exists for {today_str}. Skipping.")
            return

        if already_exists and force:
            # Update existing row
            for row in rows:
                if row["date"] == today_str:
                    row["total_value_cad"] = val_cad
                    row["total_value_usd"] = val_usd
                    row["invested_cad"] = inv_cad
                    row["invested_usd"] = inv_usd
                    row["percent_return"] = pct_return
        else:
            # Append new row
            rows.append({
                "date": today_str,
                "total_value_cad": val_cad,
                "total_value_usd": val_usd,
                "invested_cad": inv_cad,
                "invested_usd": inv_usd,
                "percent_return": pct_return
            })

        # Write back
        with open(history_file, 'w', newline='') as f:
            fieldnames = ["date", "total_value_cad", "total_value_usd", "invested_cad", "invested_usd", "percent_return"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        safe_print(f"📸 Portfolio Snapshot saved for {today_str}: ${val_usd:,.2f} USD")

    except Exception as e:
        safe_print(f"Snapshot failed: {e}")

@log_exceptions()
def get_portfolio_history(period: str = "all") -> pd.DataFrame:
    """
    Get historical portfolio performance.
    period: '1m', '3m', '6m', '1y', 'all'
    """
    history_file = get_history_file()
    if not os.path.exists(history_file):
        return pd.DataFrame()

    df = pd.read_csv(history_file)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')

    if period == "1m":
        start_date = pd.Timestamp.now() - pd.DateOffset(months=1)
    elif period == "3m":
        start_date = pd.Timestamp.now() - pd.DateOffset(months=3)
    elif period == "6m":
        start_date = pd.Timestamp.now() - pd.DateOffset(months=6)
    elif period == "1y":
        start_date = pd.Timestamp.now() - pd.DateOffset(years=1)
    else:
        start_date = None

    if start_date:
        df = df[df['date'] >= start_date]

    return df

if __name__ == "__main__":
    snapshot_portfolio()
    print(get_portfolio_history().tail())

@log_exceptions()
def seed_mock_history():
    """Seed 30 days of mock history for demonstration."""
    history_file = get_history_file()
    if os.path.exists(history_file):
        df = pd.read_csv(history_file)
        if len(df) > 5: return # Already adequate history

        # Take the last real row as anchor
        last_row = df.iloc[-1]
        anchor_val = float(last_row["total_value_usd"])
        anchor_inv = float(last_row["invested_usd"])

        import numpy as np

        # Generate 30 days back
        dates = pd.date_range(end=pd.Timestamp.now(), periods=30)

        # Random walk
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.01, 30) # slight upward drift
        values = [anchor_val]
        for r in returns[::-1]: # walk backwards
            values.append(values[-1] / (1 + r))
        values = values[::-1][1:] # reverse back and drop the extra

        # Create rows
        rows = []
        for i, d in enumerate(dates):
            val = values[i]
            # invested stays roughly constant/slightly lower in past
            inv = anchor_inv * (0.95 + (i/30)*0.05)
            ret_pct = ((val - inv) / inv) * 100

            rows.append({
                "date": d.strftime("%Y-%m-%d"),
                "total_value_cad": val * 1.35, # approx
                "total_value_usd": val,
                "invested_cad": inv * 1.35,
                "invested_usd": inv,
                "percent_return": ret_pct
            })

        pd.DataFrame(rows).to_csv(history_file, index=False)
        print("✅ Seeded 30 days of mock history")

if __name__ == "__main__":
    snapshot_portfolio()
    seed_mock_history()
    print(get_portfolio_history().tail())
