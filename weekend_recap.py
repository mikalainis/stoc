import pandas as pd
import sys
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List

# --- Local Imports ---
# Assumes Config is defined correctly in config.py or main.py
try:
    from main import Config
except ImportError:
    # Fallback for folder structure variations
    try:
        from config import Config
    except ImportError:
        print("[CRITICAL] FATAL: Cannot import Config.")
        sys.exit(1)

# --- Third-Party Imports ---
import yfinance as yf
from slack_sdk import WebClient
from google.cloud import firestore

# --- Constants ---
WEEK_DAYS = 7

class WeekendAuditor:
    """
    The Auditor reads the trade history from Firestore, calculates weekly performance,
    and sends a non-trading summary report to Slack.

    Version: V5.1 (Darwinian Robustness)
    """
    def __init__(self):
        print(f"[{datetime.now(timezone.utc)}] 🛡️ Darwinian Auditor Initializing...")
        self.cfg = Config.load()
        self.slack = WebClient(token=self.cfg.SLACK_TOKEN)
        self.data: Dict[str, Any] = {}

        # Initialize DB connection immediately to fail fast if creds are wrong
        self.db = firestore.Client(database="stocs")

    def _load_data(self):
        """Loads the trade history from Firestore."""
        try:
            print("   ☁️ Connecting to Firestore (DB: stocs)...")
            collection = self.db.collection("darwinian_analysis")

            # Optimization: If dataset grows >10k docs, add .where() filter here.
            docs = collection.stream()

            self.data = {}
            for doc in docs:
                self.data[doc.id] = doc.to_dict()

            print(f"   ✅ Loaded {len(self.data)} trade records.")

        except Exception as e:
            print(f"   ❌ [ERROR] Firestore Load Failed: {e}")
            self.data = {}
            sys.exit(1) # Force Job Failure so Cloud Run retries

    def _clean_pl_column(self, val):
        """Helper to sanitize P/L values mixed between strings and floats."""
        if pd.isna(val) or val == "":
            return 0.0
        if isinstance(val, (int, float)):
            return float(val)
        if isinstance(val, str):
            # Remove % and convert, handle '12%' -> 0.12
            clean = val.replace('%', '').replace(',', '').strip()
            try:
                f = float(clean)
                # If string had %, it was likely 12.5%, so we want 0.125?
                # Darwinian Convention: Usually raw string '12.5' means 12.5%.
                # Adjust based on your specific storage logic.
                # Assuming storage is raw percentage (e.g. 12.5) -> 0.125
                return f / 100 if '%' in val else f
            except ValueError:
                return 0.0
        return 0.0

    def _calculate_weekly_metrics(self) -> Dict[str, Any]:
        """Calculates Realized P/L, Unrealized P/L, and Win Rate."""
        self._load_data()

        if not self.data:
            print("   ⚠️ No data found in Firestore.")
            return {"profit": 0.0, "realized_pl": 0.0, "unrealized_pl": 0.0, "win_rate": 0.0, "count": 0}

        # Create DataFrame
        df = pd.DataFrame(self.data.values())

        # --- Safe Column Initialization ---
        required_cols = ['position_qty', 'avg_entry_price', 'current_price', 'PL', 'TotalValue', 'updated_at', 'Action', 'ticker']
        for col in required_cols:
            if col not in df.columns:
                df[col] = 0.0 if col not in ['updated_at', 'Action', 'ticker'] else None

        # Standardize Datetime (Force UTC)
        df['updated_at'] = pd.to_datetime(df['updated_at'], errors='coerce', utc=True)

        # --- 1. Realized P/L (Closed Trades this week) ---
        week_start = datetime.now(timezone.utc) - timedelta(days=WEEK_DAYS)

        weekly_sells = df[
            (df['updated_at'] >= week_start) &
            (df['Action'].isin(['EXECUTED_SELL', 'SELL'])) # Handle inconsistent casing/naming
        ].copy()

        total_realized_pl, total_trades, wins = 0.0, 0, 0

        if not weekly_sells.empty:
            # Vectorized P/L Cleaning
            weekly_sells['PL_PCT'] = weekly_sells['PL'].apply(self._clean_pl_column)
            weekly_sells['TotalValue'] = pd.to_numeric(weekly_sells['TotalValue'], errors='coerce').fillna(0)

            # Calculate absolute dollar P/L
            weekly_sells['Realized_PL'] = weekly_sells['TotalValue'] * weekly_sells['PL_PCT']

            total_realized_pl = weekly_sells['Realized_PL'].sum()
            total_trades = len(weekly_sells)
            wins = len(weekly_sells[weekly_sells['Realized_PL'] > 0])

        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0.0

        # --- 2. Unrealized P/L (Active Holds) ---
        unrealized_pl = 0.0

        # Filter for positive quantity (Active Positions)
        # We ensure 'position_qty' is numeric first
        df['position_qty'] = pd.to_numeric(df['position_qty'], errors='coerce').fillna(0)

        current_positions = df[df['position_qty'] > 0.0001].copy() # Tolerance for float rounding

        if not current_positions.empty:
            symbols = [s for s in current_positions['ticker'].unique() if s]
            print(f"   🔎 Auditing {len(symbols)} active positions...")

            if symbols:
                try:
                    # Efficient Batch Download
                    # auto_adjust=True fixes split/dividend issues
                    live_data = yf.download(symbols, period="1d", progress=False, auto_adjust=True)['Close']

                    # Handle single ticker returning Series vs multiple returning DataFrame
                    if isinstance(live_data, pd.Series):
                        # If only one symbol, pandas returns a Series with Date index
                        current_prices = {symbols[0]: live_data.iloc[-1]}
                    else:
                        current_prices = live_data.iloc[-1].to_dict()

                    for _, pos in current_positions.iterrows():
                        ticker = pos['ticker']
                        qty = float(pos['position_qty'])
                        entry = float(pos['avg_entry_price']) if pos['avg_entry_price'] else 0.0

                        # Get live price or fallback to last known DB price
                        live_price = current_prices.get(ticker, float(pos['current_price']) if pos['current_price'] else entry)

                        if entry > 0:
                            pnl = (live_price - entry) * qty
                            unrealized_pl += pnl

                except Exception as e:
                    print(f"   ⚠️ [WARN] Price Fetch Error: {e}")

        # --- 3. Final Metrics ---
        total_net_profit = total_realized_pl + unrealized_pl

        return {
            "profit": total_net_profit,
            "realized_pl": total_realized_pl,
            "unrealized_pl": unrealized_pl,
            "win_rate": win_rate,
            "count": total_trades
        }

    def _send_weekly_slack_report(self, metrics: Dict[str, Any]):
        """Formats and sends the weekly summary to Slack."""
        profit = metrics['profit']
        realized = metrics['realized_pl']
        unrealized = metrics['unrealized_pl']
        win_rate = metrics['win_rate']
        count = metrics['count']

        emoji = "🟢" if profit >= 0 else "🔴"
        # Determine sentiment string
        if profit > 500: sentiment = "Great week! The swarm is feeding."
        elif profit < -500: sentiment = "Defensive measures required. Swarm retreating."
        else: sentiment = "Holding steady."

        blocks = [
            {"type": "header", "text": {"type": "plain_text", "text": f"📊 Darwinian Swarm Weekly Recap"}},
            {"type": "section", "text": {"type": "mrkdwn", "text": f"*{sentiment}*"}},
            {"type": "divider"},
            {"type": "section", "fields": [
                {"type": "mrkdwn", "text": f"*Net P/L:* {emoji} ${profit:,.2f}"},
                {"type": "mrkdwn", "text": f"*Win Rate:* {win_rate:.1f}%"}
            ]},
            {"type": "section", "fields": [
                {"type": "mrkdwn", "text": f"*Realized (Banked):* ${realized:,.2f}"},
                {"type": "mrkdwn", "text": f"*Unrealized (Floating):* ${unrealized:,.2f}"}
            ]},
            {"type": "context", "elements": [{"type": "mrkdwn", "text": f"Trades Closed: {count} | audited by Cloud Run"}]}
        ]

        try:
            self.slack.chat_postMessage(channel=self.cfg.SLACK_CHANNEL, text=f"Weekly P/L: ${profit:,.2f}", blocks=blocks)
            print("✅ Weekly Recap sent to Slack.")
        except Exception as e:
            print(f"❌ Slack Send Error: {e}")
            # Don't exit(1) here, math was successful, just reporting failed.

    def run(self):
        """Main entry point."""
        try:
            metrics = self._calculate_weekly_metrics()
            self._send_weekly_slack_report(metrics)
            print("🏁 Audit Complete.")
        except Exception as e:
            print(f"🔥 CRITICAL FAILURE: {e}")
            sys.exit(1)

# --- Main Execution ---
if __name__ == "__main__":
    auditor = WeekendAuditor()
    auditor.run()
