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
from google.cloud import firestore

# --- Constants ---
# LOCAL_DATA_FILE = "processed_stocks_v4.json" # Deprecated
WEEK_DAYS = 7 # Use 7 days to cover the entire week including weekend non-trading days

class WeekendAuditor:
    """
    The Auditor reads the trade history from Firestore, calculates weekly performance,
    and sends a non-trading summary report to Slack.

    Version: V4.3 Firestore
    """
    def __init__(self):
        print("Auditor Initializing...")
        self.cfg = Config.load()
        self.slack = WebClient(token=self.cfg.SLACK_TOKEN)
        self.data: Dict[str, Any] = {}
        self._load_data()

    def _load_data(self):
        """Loads the trade history from Firestore."""
        try:
            print("   ☁️ Connecting to Firestore (DB: stocs)...")
            db = firestore.Client(database="stocs")
            collection = db.collection("darwinian_analysis")
            docs = collection.stream()

            self.data = {}
            for doc in docs:
                self.data[doc.id] = doc.to_dict()

            print(f"✅ Loaded {len(self.data)} trade records from Firestore.")

        except Exception as e:
            print(f"❌ Error loading Firestore data: {e}")
            self.data = {}

    def _calculate_weekly_metrics(self) -> Dict[str, Any]:
        """Calculates Realized P/L, Unrealized P/L, and Win Rate."""
        if not self.data:
            return {"profit": 0.0, "realized_pl": 0.0, "unrealized_pl": 0.0, "win_rate": 0.0, "count": 0, "sectors": {}}

        # Create DataFrame from the list of dictionaries
        df = pd.DataFrame(self.data.values())

        # --- FIX: Ensure critical columns exist with safe defaults (0.0) ---

        required_cols = ['position_qty', 'avg_entry_price', 'current_price', 'PL', 'TotalValue', 'updated_at', 'Action', 'ticker']
        for col in required_cols:
            if col not in df.columns:
                # Assign appropriate default values based on column type expectation
                if col in ['updated_at', 'Action', 'ticker']:
                    df[col] = "" # or None
                else:
                    df[col] = 0.0

        # Standardize data types
        # Handle Firestore datetime objects or string ISO formats
        df['updated_at'] = pd.to_datetime(df['updated_at'], errors='coerce', utc=True)

        # --- 1. Realized P/L (from Sells closed this week) ---
        week_start = datetime.now(timezone.utc) - timedelta(days=WEEK_DAYS)
        weekly_sells = df[
            (df['updated_at'] >= week_start) &
            (df['Action'] == 'EXECUTED_SELL')
        ].copy()

        total_realized_pl, total_trades, wins = 0.0, 0, 0
        if not weekly_sells.empty:
            # Clean and calculate P/L for realized trades
            # Assuming PL is stored as string "X%" or similar, or float
            # Check data type of PL
            if weekly_sells['PL'].dtype == object:
                 weekly_sells['PL_PCT'] = weekly_sells['PL'].str.rstrip('%').astype(float, errors='ignore').replace(np.nan, 0) / 100
            else:
                 weekly_sells['PL_PCT'] = weekly_sells['PL']

            weekly_sells['TotalValue'] = pd.to_numeric(weekly_sells['TotalValue'], errors='coerce').fillna(0)
            weekly_sells['Realized_PL'] = weekly_sells['TotalValue'] * weekly_sells['PL_PCT']

            total_realized_pl = weekly_sells['Realized_PL'].sum()
            total_trades = len(weekly_sells)
            wins = len(weekly_sells[weekly_sells['Realized_PL'] > 0])

        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0

        # --- 2. Unrealized P/L (from Held Positions) ---
        unrealized_pl = 0.0

        # Get the LATEST state for all currently held positions (qty > 0)
        # Note: self.data is keyed by ticker, so we likely have one record per ticker (the latest).
        # However, checking 'updated_at' is good practice if we had history, but here we likely just have current state per ticker.
        # The logic below assumes df contains unique tickers if loaded from self.data[doc.id].

        current_positions = df[
            (df['position_qty'] > 0)
        ].copy()

        if not current_positions.empty:
            symbols = current_positions['ticker'].tolist()
            # Filter out empty symbols
            symbols = [s for s in symbols if s]

            if symbols:
                print(f"   Fetching live prices for {len(symbols)} held positions...")

                try:
                    # yfinance download
                    live_prices_df = yf.download(symbols, period="1d", interval="1m", progress=False)['Close']

                    live_prices = {}
                    if not live_prices_df.empty:
                        if isinstance(live_prices_df, pd.Series):
                            # Single symbol result
                            live_prices = {symbols[0]: live_prices_df.iloc[-1]}
                        else:
                            # Multiple symbols
                            live_prices = live_prices_df.iloc[-1].to_dict()

                    for _, pos in current_positions.iterrows():
                        ticker = pos['ticker']
                        qty = float(pos['position_qty'])
                        entry_price = float(pos['avg_entry_price']) if pos['avg_entry_price'] else 0.0
                        current_price = live_prices.get(ticker, entry_price)

                        # Fallback if yfinance failed for this ticker, try to use price in DB
                        if pd.isna(current_price):
                             current_price = float(pos['current_price']) if pos['current_price'] else entry_price

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
            {"type": "header", "text": {"type": "plain_text", "text": f"🥂 DarwinianSwarm V4.3 Recap"}},
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
