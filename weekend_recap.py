import pandas as pd
import json
import os
import sys
import numpy as np # Needed for handling potential NumPy types in data
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List

# --- Local Imports ---
# Assumes Config is defined correctly with the @classmethod load() method
try:
    from main import Config
except ImportError:
    print("FATAL: Cannot import Config. Ensure main.py is in the directory.")
    sys.exit(1)

# --- Third-Party Imports ---
import yfinance as yf
from slack_sdk import WebClient

# --- Constants ---
LOCAL_DATA_FILE = "processed_stocks_v4.json"
WEEK_DAYS = 7 # Use 7 days to cover the entire week including weekend non-trading days

class WeekendAuditor:
    """
    The Auditor reads the local trade history, calculates weekly performance,
    and sends a non-trading summary report to Slack.

    Version: V4.2 Final
    """
    def __init__(self):
        print("Auditor Initializing...")
        self.cfg = Config.load()
        self.slack = WebClient(token=self.cfg.SLACK_TOKEN)
        self.data: Dict[str, Any] = {}
        self._load_local_data()

    def _load_local_data(self):
        """Loads the trade history from the JSON file."""
        if not os.path.exists(LOCAL_DATA_FILE):
            print(f"⚠️ Warning: Local file '{LOCAL_DATA_FILE}' not found. No data to process.")
            return

        try:
            with open(LOCAL_DATA_FILE, 'r') as f:
                self.data = json.load(f)
            print(f"✅ Loaded {len(self.data)} historical trade records.")
        except Exception as e:
            print(f"❌ Error loading JSON data: {e}. Starting with empty data.")
            self.data = {}

    def _calculate_weekly_metrics(self) -> Dict[str, Any]:
        """Calculates Realized P/L, Unrealized P/L, and Win Rate."""
        if not self.data:
            return {"profit": 0.0, "realized_pl": 0.0, "unrealized_pl": 0.0, "win_rate": 0.0, "count": 0, "sectors": {}}

        # Create DataFrame from the list of dictionaries
        df = pd.DataFrame(self.data.values())

        # --- FIX: Ensure critical columns exist with safe defaults (0.0) ---

        required_cols = ['position_qty', 'avg_entry_price', 'current_price', 'PL', 'TotalValue']
        for col in required_cols:
            if col not in df.columns:
                df[col] = 0.0

        # Standardize data types
        df['updated_at'] = pd.to_datetime(df['updated_at'], errors='coerce')

        # --- 1. Realized P/L (from Sells closed this week) ---
        week_start = datetime.now(timezone.utc) - timedelta(days=WEEK_DAYS)
        weekly_sells = df[
            (df['updated_at'] >= week_start) &
            (df['Action'] == 'EXECUTED_SELL')
        ].copy()

        total_realized_pl, total_trades, wins = 0.0, 0, 0
        if not weekly_sells.empty:
            # Clean and calculate P/L for realized trades
            weekly_sells['PL_PCT'] = weekly_sells['PL'].str.rstrip('%').astype(float, errors='ignore').replace(np.nan, 0) / 100
            weekly_sells['TotalValue'] = pd.to_numeric(weekly_sells['TotalValue'], errors='coerce').fillna(0)
            weekly_sells['Realized_PL'] = weekly_sells['TotalValue'] * weekly_sells['PL_PCT']

            total_realized_pl = weekly_sells['Realized_PL'].sum()
            total_trades = len(weekly_sells)
            wins = len(weekly_sells[weekly_sells['Realized_PL'] > 0])

        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0

        # --- 2. Unrealized P/L (from Held Positions) ---
        unrealized_pl = 0.0

        # Get the LATEST state for all currently held positions (qty > 0)
        current_positions = df[
            (df['position_qty'] > 0)
        ].sort_values(by='updated_at').drop_duplicates(subset=['ticker'], keep='last')

        if not current_positions.empty:
            symbols = current_positions['ticker'].tolist()
            print(f"   Fetching live prices for {len(symbols)} held positions...")

            try:
                live_prices_df = yf.download(symbols, period="1d", interval="1m", progress=False)['Close']

                if isinstance(live_prices_df, pd.Series):
                    live_prices = {symbols[0]: live_prices_df.iloc[-1]}
                else:
                    live_prices = live_prices_df.iloc[-1].to_dict()

                for _, pos in current_positions.iterrows():
                    ticker = pos['ticker']
                    qty = pos['position_qty']
                    entry_price = pos['avg_entry_price']
                    current_price = live_prices.get(ticker, entry_price)

                    if entry_price > 0 and current_price > 0:
                        unrealized_pl += qty * (current_price - entry_price)
            except Exception as e:
                print(f"   ⚠️ Error fetching live prices: {e}. Unrealized P/L may be inaccurate.")

        # --- 3. Final Metrics ---
        total_net_profit = total_realized_pl + unrealized_pl

        return {
            "profit": total_net_profit,
            "realized_pl": total_realized_pl,
            "unrealized_pl": unrealized_pl,
            "win_rate": win_rate,
            "count": total_trades,
            "sectors": {}
        }

    def _send_weekly_slack_report(self, metrics: Dict[str, Any]):
        """Formats and sends the weekly summary to Slack."""
        profit = metrics['profit']
        realized_pl = metrics['realized_pl']
        unrealized_pl = metrics['unrealized_pl']
        win_rate = metrics['win_rate']
        count = metrics['count']

        emoji = "💰" if profit >= 0 else "📉"

        blocks = [
            {"type": "header", "text": {"type": "plain_text", "text": f"🥂 DarwinianSwarm V4.2 Final Recap"}},
            {"type": "section", "text": {"type": "mrkdwn", "text": f"*Weekly Net P/L:* {emoji} *${profit:,.2f}*"}},
            {"type": "section", "fields": [
                {"type": "mrkdwn", "text": f"*Realized (Sells):* ${realized_pl:,.2f}"},
                {"type": "mrkdwn", "text": f"*Unrealized (Holds):* ${unrealized_pl:,.2f}"},
            ]},
            {"type": "section", "fields": [
                {"type": "mrkdwn", "text": f"*Win Rate (Sells):* {win_rate:.1f}%"},
                {"type": "mrkdwn", "text": f"*Trades Closed:* {count}"}
            ]},
            {"type": "divider"}
        ]

        try:
            self.slack.chat_postMessage(channel=self.cfg.SLACK_CHANNEL, text=f"Weekly P/L: ${profit:,.2f}", blocks=blocks)
            print("✅ Weekly Recap sent to Slack.")
        except Exception as e:
            print(f"❌ Slack Send Error: {e}")

    def run(self):
        """Main entry point for the Auditor."""
        print(f"Auditor running for weekly review.")
        metrics = self._calculate_weekly_metrics()
        self._send_weekly_slack_report(metrics)

# --- Main Execution ---
if __name__ == "__main__":
    auditor = WeekendAuditor()
    auditor.run()
