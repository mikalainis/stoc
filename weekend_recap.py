import pandas as pd
import sys
import os
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List

# --- Local Imports ---
try:
    from main import Config
except ImportError:
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

    Version: V5.2 (Context-Aware & Optimized)
    """
    def __init__(self):
        print(f"[{datetime.now(timezone.utc)}] 🛡️ Darwinian Auditor V5.2 Initializing...")
        self.cfg = Config.load()
        self.slack = WebClient(token=self.cfg.SLACK_TOKEN)
        self.data: Dict[str, Any] = {}

        # Initialize DB connection immediately, with fallback for CI/Test
        try:
            self.db = firestore.Client(database="stocs")
        except Exception as e:
            if os.getenv("IS_PAPER") == "True":
                print(f"    ⚠️ [TEST MODE] Firestore Init Failed (Expected): {e}")
                self.db = None
            else:
                raise e

    def _get_market_context(self) -> Dict[str, Any]:
        """
        Fetches macro context (VIX) to explain performance.
        """
        try:
            print("    ☁️ Checking Market Weather (VIX)...")
            # Fetch VIX to determine if we were in High Volatility mode
            vix_data = yf.download("^VIX", period="5d", progress=False, auto_adjust=True)

            if vix_data.empty:
                return {"vix": 0.0, "mode": "UNKNOWN"}

            # Handle MultiIndex if necessary, though single ticker usually simple
            current_vix = float(vix_data['Close'].iloc[-1])

            if current_vix > 30:
                mode = "PANIC (Bear)"
            elif current_vix < 20:
                mode = "CALM (Bull/Neutral)"
            else:
                mode = "CHOPPY (Neutral)"

            return {"vix": current_vix, "mode": mode}
        except Exception as e:
            print(f"    ⚠️ [WARN] Could not fetch VIX: {e}")
            return {"vix": 0.0, "mode": "UNKNOWN"}

    def _load_data(self):
        """
        Loads trade history.
        OPTIMIZATION V5.2: Merges 'Recent' and 'Active' queries to save RAM.
        """
        if self.db is None:
            print("    ⚠️ [TEST MODE] Skipping Firestore load. Using empty data.")
            self.data = {}
            return

        try:
            print("    ☁️ Connecting to Firestore (DB: stocs)...")
            collection = self.db.collection("darwinian_analysis")

            self.data = {}

            # --- Query 1: Recent History (Last 7 Days) ---
            # Finds all trades (Buy or Sell) touched this week
            week_start = datetime.now(timezone.utc) - timedelta(days=WEEK_DAYS)
            recent_docs = collection.where(field_path='updated_at', op_string='>=', value=week_start).stream()

            count_recent = 0
            for doc in recent_docs:
                self.data[doc.id] = doc.to_dict()
                count_recent += 1

            # --- Query 2: Active Positions (The Survivors) ---
            # Finds old positions that are still open.
            # Note: Requires composite index on position_qty if combined with other filters.
            # Simple inequality on one field is usually safe.
            active_docs = collection.where(field_path='position_qty', op_string='>', value=0.0001).stream()

            count_active = 0
            for doc in active_docs:
                # Deduplicate: Only add if not already loaded by Query 1
                if doc.id not in self.data:
                    self.data[doc.id] = doc.to_dict()
                    count_active += 1

            print(f"    ✅ Loaded {len(self.data)} records ({count_recent} recent, {count_active} older active).")

        except Exception as e:
            print(f"    ❌ [ERROR] Firestore Load Failed: {e}")
            # Fallback: If indexes fail, try loading everything (Survival Mode)
            try:
                print("    ⚠️ Attempting fallback: Loading FULL history...")
                all_docs = self.db.collection("darwinian_analysis").stream()
                self.data = {doc.id: doc.to_dict() for doc in all_docs}
                print(f"    ✅ Fallback successful: {len(self.data)} records.")
            except Exception as fatal_e:
                print(f"    🔥 CRITICAL: Fallback failed. {fatal_e}")
                sys.exit(1)

    def _clean_pl_column(self, val):
        """Helper to sanitize P/L values mixed between strings and floats."""
        if pd.isna(val) or val == "":
            return 0.0
        if isinstance(val, (int, float)):
            return float(val)
        if isinstance(val, str):
            clean = val.replace('%', '').replace(',', '').strip()
            try:
                f = float(clean)
                return f / 100 if '%' in val else f
            except ValueError:
                return 0.0
        return 0.0

    def _calculate_weekly_metrics(self) -> Dict[str, Any]:
        """Calculates P/L and merges with Market Context."""
        # 1. Get Context
        context = self._get_market_context()

        # 2. Load Data
        self._load_data()

        if not self.data:
            print("    ⚠️ No data found in Firestore.")
            return {**context, "profit": 0.0, "realized_pl": 0.0, "unrealized_pl": 0.0, "win_rate": 0.0, "count": 0}

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
            (df['Action'].isin(['EXECUTED_SELL', 'SELL']))
        ].copy()

        total_realized_pl, total_trades, wins = 0.0, 0, 0

        if not weekly_sells.empty:
            weekly_sells['PL_PCT'] = weekly_sells['PL'].apply(self._clean_pl_column)
            weekly_sells['TotalValue'] = pd.to_numeric(weekly_sells['TotalValue'], errors='coerce').fillna(0)

            weekly_sells['Realized_PL'] = weekly_sells['TotalValue'] * weekly_sells['PL_PCT']

            total_realized_pl = weekly_sells['Realized_PL'].sum()
            total_trades = len(weekly_sells)
            wins = len(weekly_sells[weekly_sells['Realized_PL'] > 0])

        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0.0

        # --- 2. Unrealized P/L (Active Holds) ---
        unrealized_pl = 0.0

        df['position_qty'] = pd.to_numeric(df['position_qty'], errors='coerce').fillna(0)
        current_positions = df[df['position_qty'] > 0.0001].copy()

        if not current_positions.empty:
            symbols = [s for s in current_positions['ticker'].unique() if s]
            print(f"    🔎 Auditing {len(symbols)} active positions...")

            if symbols:
                try:
                    live_data = yf.download(symbols, period="1d", progress=False, auto_adjust=True)['Close']

                    if isinstance(live_data, pd.Series):
                        current_prices = {symbols[0]: live_data.iloc[-1]}
                    else:
                        current_prices = live_data.iloc[-1].to_dict()

                    for _, pos in current_positions.iterrows():
                        ticker = pos['ticker']
                        qty = float(pos['position_qty'])
                        entry = float(pos['avg_entry_price']) if pos['avg_entry_price'] else 0.0

                        live_price = current_prices.get(ticker, float(pos['current_price']) if pos['current_price'] else entry)

                        if entry > 0:
                            pnl = (live_price - entry) * qty
                            unrealized_pl += pnl

                except Exception as e:
                    print(f"    ⚠️ [WARN] Price Fetch Error: {e}")

        # --- 3. Final Metrics ---
        total_net_profit = total_realized_pl + unrealized_pl

        return {
            "profit": total_net_profit,
            "realized_pl": total_realized_pl,
            "unrealized_pl": unrealized_pl,
            "win_rate": win_rate,
            "count": total_trades,
            "vix": context['vix'],
            "mode": context['mode']
        }

    def _send_weekly_slack_report(self, metrics: Dict[str, Any]):
        """Formats and sends the weekly summary to Slack with Context."""
        profit = metrics['profit']
        realized = metrics['realized_pl']
        unrealized = metrics['unrealized_pl']
        win_rate = metrics['win_rate']
        count = metrics['count']
        vix = metrics['vix']
        mode = metrics['mode']

        emoji = "🟢" if profit >= 0 else "🔴"

        # Adaptive Sentiment
        if profit > 500:
            sentiment = "Great week! The swarm is feeding."
        elif profit < -500:
            sentiment = "Defensive measures required. Swarm retreating."
        else:
            sentiment = "Holding steady."

        # Add Contextual Warning
        context_str = f"Market Mode: {mode} (VIX: {vix:.2f})"

        blocks = [
            {"type": "header", "text": {"type": "plain_text", "text": f"📊 Darwinian Swarm Weekly Recap"}},
            {"type": "section", "text": {"type": "mrkdwn", "text": f"*{sentiment}*"}},
            {"type": "context", "elements": [{"type": "mrkdwn", "text": f"🌪️ {context_str}"}]},
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
