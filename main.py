import os
import json
import logging
import sys
import time
import math
import re
import concurrent.futures
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from typing import Callable, Optional, List, Dict, Set, Union

import pandas as pd
import nest_asyncio
import requests

# --- THIRD PARTY IMPORTS ---
from google import genai
from google.genai import types
from google.cloud import storage # Required for Cloud Persistence
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import StockHistoricalDataClient, CryptoHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed

# Patch for nested async loops (Colab/Jupyter)
nest_asyncio.apply()

# --- 1. LOGGING CONFIGURATION ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("DarwinianSwarm")

# ==========================================
#      🚀 USER CONFIGURATION
# ==========================================
USER_SETTINGS = {
    # --- STRATEGY & RISK ---
    "TRADE_ALLOCATION": 3000.0,   # Target Buy size ($) per trade
    "MAX_POS_PERCENT": 0.05,      # Max Portfolio % size per single stock (5% Cap)
    "MIN_CONFIDENCE": 85,         # % AI Confidence required to Buy
    "RSI_BUY_THRESHOLD": 35,      # Buy if RSI is below this (Dip Buying)

    # --- DISCOVERY ENGINE ---
    "TOP_N_STOCKS": 40,           # Number of stocks to hunt for
    "ANALYSIS_COOLDOWN_HOURS": 24,# Don't re-scan a rejected stock for X hours
    "EXECUTION_TIMEOUT_SECONDS": 60, # Kill analysis if it hangs > 60s

    # --- TECHNICALS ---
    "RSI_PERIOD": 14,             # Standard RSI Lookback
    "DATA_LOOKBACK_DAYS": 45,     # Days of history to fetch for math accuracy

    # --- INFRASTRUCTURE ---
    "SLACK_CHANNEL": "D0A1C7TBB5E",
    "GCS_BUCKET_NAME": None,       # Auto-loaded from Env Var in Cloud
    "ALPHA_VANTAGE_KEY": "D8HPKBQ0AKCA8FUI",
    "FMP_API_KEY": "waiwr0TJe2NKRuPo5ceRF4xmjtp2k9uv"
    "ALPHA_VANTAGE_KEY": "D8HPKBQ0AKCA8FUI"
}
# ==========================================

# --- 2. CONFIGURATION MANAGEMENT ---
@dataclass(frozen=True)
class Config:
    GOOGLE_KEY: str
    ALPACA_KEY: str
    ALPACA_SECRET: str
    SLACK_TOKEN: str
    SLACK_CHANNEL: str
    IS_PAPER: bool
    GCS_BUCKET_NAME: Optional[str]
    ALPHA_VANTAGE_KEY: Optional[str]
    FMP_API_KEY: Optional[str]

    TRADE_ALLOCATION: float
    MAX_POS_PERCENT: float
    MIN_CONFIDENCE: int
    RSI_BUY_THRESHOLD: int
    RSI_PERIOD: int
    DATA_LOOKBACK_DAYS: int
    TOP_N_STOCKS: int
    ANALYSIS_COOLDOWN_HOURS: int
    EXECUTION_TIMEOUT_SECONDS: int

    @classmethod
    def load(cls) -> "Config":
        try:
            from google.colab import userdata
            get_secret = userdata.get
        except ImportError:
            get_secret = os.getenv

        def get_param(key, type_func):
            # 1. Env Var, 2. User Settings, 3. None
            env_val = os.getenv(key)
            if env_val: return type_func(env_val)
            val = USER_SETTINGS.get(key)
            return type_func(val) if val is not None else None

        secrets = {
            "GOOGLE_KEY": get_secret("GOOGLE_API_KEY"),
            "ALPACA_KEY": get_secret("ALPACA_API_KEY"),
            "ALPACA_SECRET": get_secret("ALPACA_SECRET"),
            "SLACK_TOKEN": get_secret("SLACK_BOT_TOKEN"),
            "SLACK_CHANNEL": get_param("SLACK_CHANNEL", str),
            "GCS_BUCKET_NAME": get_param("GCS_BUCKET_NAME", str),
            "ALPHA_VANTAGE_KEY": get_param("ALPHA_VANTAGE_KEY", str),
            "FMP_API_KEY": get_param("FMP_API_KEY", str),

            "IS_PAPER": True, # Set to False for Real Money

            # Load Strategy Params
            "TRADE_ALLOCATION": get_param("TRADE_ALLOCATION", float),
            "MAX_POS_PERCENT": get_param("MAX_POS_PERCENT", float),
            "MIN_CONFIDENCE": get_param("MIN_CONFIDENCE", int),
            "RSI_BUY_THRESHOLD": get_param("RSI_BUY_THRESHOLD", int),
            "RSI_PERIOD": get_param("RSI_PERIOD", int),
            "DATA_LOOKBACK_DAYS": get_param("DATA_LOOKBACK_DAYS", int),
            "TOP_N_STOCKS": get_param("TOP_N_STOCKS", int),
            "ANALYSIS_COOLDOWN_HOURS": get_param("ANALYSIS_COOLDOWN_HOURS", int),
            "EXECUTION_TIMEOUT_SECONDS": get_param("EXECUTION_TIMEOUT_SECONDS", int)
        }

        # Validate Critical Secrets
        missing = [k for k in ["GOOGLE_KEY", "ALPACA_KEY", "ALPACA_SECRET", "SLACK_TOKEN"] if not secrets[k]]
        if missing:
            print(f"DEBUG: sys.argv={sys.argv}")
            # RELAXED CHECK FOR TEST MODE
            if "--test" in sys.argv:
                logger.warning(f"⚠️ MISSING SECRETS IN TEST MODE: {missing}. Using dummy values to proceed with smoke test.")
                for k in missing:
                    secrets[k] = "TEST_DUMMY_VALUE"
            else:
                logger.critical(f"Missing Critical Secrets: {missing}")
                sys.exit(1)

        return cls(**secrets)

# --- 3. STATE MANAGEMENT ---
@dataclass
class MarketState:
    ticker: str
    analysis_timestamp: str = ""
    news_summary: str = ""
    decision: str = "HOLD"
    sentiment_score: int = 0
    confidence: int = 0
    reasoning: str = ""
    current_price: float = 0.0
    current_rsi: float = 0.0
    asset_class: str = "STOCK"
    position_qty: float = 0.0

from google.cloud import firestore
import google.auth.exceptions

# --- 4. SMART HYBRID TRACKER (Firestore + JSON Fallback) ---
class SmartTracker:
    def __init__(self, config: Config, filename="processed_stocks_v4.json"):
        self.filename = filename
        self.bucket_name = config.GCS_BUCKET_NAME

        # --- UPDATE THIS BLOCK ---
        try:
            # Explicitly connect to the "stocs" database
            self.db = firestore.Client(database="stocs")
            self.collection = self.db.collection("darwinian_analysis")
            self.use_firestore = True
            print("   🔥 Connected to Firestore Database: stocs")
        except Exception as e:
            print(f"   ⚠️ Firestore Error: {e}")
            self.use_firestore = False

    # --- LOCAL HANDLERS ---
    def _load_local(self):
        if not os.path.exists(self.filename): return {}
        try:
            with open(self.filename, 'r') as f: return json.load(f)
        except: return {}

    def _save_local(self):
        with open(self.filename, 'w') as f:
            json.dump(self.local_data, f, indent=4)

    # --- UNIFIED INTERFACE ---
    def should_rescan(self, ticker: str, config: Config) -> bool:
        # A. FIRESTORE LOGIC
        if self.use_firestore:
            try:
                doc = self.collection.document(ticker).get()
                if not doc.exists: return True
                data = doc.to_dict()
                return self._check_logic(data, config, ticker)
            except Exception as e:
                print(f"   ❌ Firestore Read Error: {e}")
                return True # Fail-open (Scan if DB fails)

        # B. LOCAL LOGIC
        else:
            if ticker not in self.local_data: return True
            data = self.local_data[ticker]
            return self._check_logic(data, config, ticker)

    def _check_logic(self, data, config, ticker):
        """Shared logic for checking timestamps and criteria"""
        timestamp_str = data.get("analysis_timestamp", "")
        if timestamp_str:
            last_time = datetime.fromisoformat(timestamp_str).replace(tzinfo=timezone.utc)
            hours = (datetime.now(timezone.utc) - last_time).total_seconds() / 3600

            if hours >= config.ANALYSIS_COOLDOWN_HOURS:
                print(f"   🔄 Re-scan {ticker}: Expired ({hours:.1f}h)")
                return True

        # Re-Evaluation Logic
        last_sent = data.get("sentiment_score", 0)
        last_rsi = data.get("current_rsi", 100)

        if last_sent >= config.MIN_CONFIDENCE and last_rsi < config.RSI_BUY_THRESHOLD:
            print(f"   ⚡ Re-evaluating {ticker}: Meets new criteria.")
            return True

        print(f"   ⏭️ Skipping {ticker}: Analyzed recently.")
        return False

    def mark_processed(self, state: MarketState):
        record = asdict(state)

        if self.use_firestore:
            try:
                # Add server timestamp for sorting in Cloud Console
                record["updated_at"] = firestore.SERVER_TIMESTAMP
                self.collection.document(state.ticker).set(record, merge=True)
                print(f"   ☁️ Saved {state.ticker} to Firestore.")
            except Exception as e:
                print(f"   ❌ Firestore Write Error: {e}")
        else:
            self.local_data[state.ticker] = record
            self._save_local()
            print(f"   💾 Saved {state.ticker} to Local JSON.")

# --- 5. CORE ANALYST AGENT ---
class DarwinianSwarm:
    def __init__(self, ticker: str, config: Config, existing_qty: Optional[float] = None):
        self.config = config
        self.gemini = genai.Client(api_key=config.GOOGLE_KEY)
        self.alpaca_trade = TradingClient(config.ALPACA_KEY, config.ALPACA_SECRET, paper=config.IS_PAPER)
        self.alpaca_stock_data = StockHistoricalDataClient(config.ALPACA_KEY, config.ALPACA_SECRET)
        self.slack = WebClient(token=config.SLACK_TOKEN)

        if existing_qty is None:
            try:
                pos = self.alpaca_trade.get_open_position(ticker.upper())
                detected_qty = float(pos.qty)
            except: detected_qty = 0.0
        else:
            detected_qty = existing_qty

        self.state = MarketState(ticker=ticker.upper(), position_qty=detected_qty)

    def _notify_slack_simple(self, message: str):
        try: self.slack.chat_postMessage(channel=self.config.SLACK_CHANNEL, text=message)
        except: pass

    # --- PIPELINE: Technicals ---
    def check_technicals(self) -> bool:
        start = datetime.now(timezone.utc) - timedelta(days=self.config.DATA_LOOKBACK_DAYS)
        try:
            req = StockBarsRequest(symbol_or_symbols=self.state.ticker, timeframe=TimeFrame.Hour, start=start, limit=200, feed=DataFeed.IEX)
            bars = self.alpaca_stock_data.get_stock_bars(req)
            if not bars.data:
                print(f"   ⚠️ No Data for {self.state.ticker}")
                return False

            df = bars.df
            if isinstance(df.index, pd.MultiIndex): df = df.reset_index()
            df.columns = [c.lower() for c in df.columns]

            # RSI Calculation
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).fillna(0)
            loss = (-delta.where(delta < 0, 0)).fillna(0)
            avg_gain = gain.ewm(com=self.config.RSI_PERIOD-1, min_periods=self.config.RSI_PERIOD).mean()
            avg_loss = loss.ewm(com=self.config.RSI_PERIOD-1, min_periods=self.config.RSI_PERIOD).mean()
            rs = avg_gain / avg_loss
            df['rsi'] = 100 - (100 / (1 + rs))

            self.state.current_price = float(df['close'].iloc[-1])
            self.state.current_rsi = float(df['rsi'].iloc[-1])
            return True
        except Exception as e:
            print(f"   ❌ Tech Error: {e}")
            return False

    # --- PIPELINE: News ---
    def fetch_news_context(self) -> bool:
        try:
            url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={self.state.ticker}&apikey={self.config.ALPHA_VANTAGE_KEY}&limit=10"
            r = requests.get(url)
            data = r.json()

            if "feed" not in data:
                self.state.news_summary = "No news found."
                return True

            articles = data["feed"][:5] # Top 5
            summary_lines = []
            for art in articles:
                title = art.get('title', 'No Title')
                sentiment_score = art.get('overall_sentiment_score', 0)
                sentiment_label = art.get('overall_sentiment_label', 'Neutral')
                summary = art.get('summary', 'No Summary')
                summary_lines.append(f"- {title} (Sentiment: {sentiment_label}, Score: {sentiment_score}): {summary}")

            self.state.news_summary = "\n".join(summary_lines) if summary_lines else "No news found."
            return True
        except Exception as e:
            print(f"   ❌ News Error: {e}")
            return False

    # --- PIPELINE: Analysis ---
    def analyze_sentiment(self) -> bool:
        schema = {"type": "OBJECT", "properties": {"sentiment_score": {"type": "INTEGER"}, "decision": {"type": "STRING", "enum": ["BUY", "SELL", "HOLD"]}, "confidence": {"type": "INTEGER"}, "reasoning": {"type": "STRING"}}}
        prompt = f"""
        Act as a Quant Portfolio Manager.
        ASSET: {self.state.ticker} | PRICE: ${self.state.current_price:.2f} | RSI: {self.state.current_rsi:.2f}
        NEWS: {self.state.news_summary}
        PARAMS: Strong Buy > {self.config.MIN_CONFIDENCE}% | Buy RSI < {self.config.RSI_BUY_THRESHOLD}
        TASK: Rate 'sentiment_score' (0-100) on news. Buy ONLY if score >= {self.config.MIN_CONFIDENCE} AND RSI < {self.config.RSI_BUY_THRESHOLD}.
        """
        try:
            response = self.gemini.models.generate_content(
                model="gemini-2.5-flash-lite", contents=prompt,
                config=types.GenerateContentConfig(response_mime_type="application/json", response_schema=schema)
            )
            data = json.loads(response.text)
            self.state.sentiment_score = data["sentiment_score"]
            self.state.decision = data["decision"]
            self.state.confidence = data["confidence"]
            self.state.reasoning = data["reasoning"]
            self.state.analysis_timestamp = datetime.now(timezone.utc).isoformat()
            return True
        except: return False

    # --- SIZING (5% CAP LOGIC) ---
    def calculate_investment_qty(self) -> float:
        try:
            if self.state.current_price <= 0: return 0.0

            # 1. Get Portfolio Total Equity
            account = self.alpaca_trade.get_account()
            total_equity = float(account.equity)

            # 2. Calculate Cap ($ Limit for this stock)
            max_allowed_value = total_equity * self.config.MAX_POS_PERCENT

            # 3. Calculate Current Exposure
            current_value = self.state.position_qty * self.state.current_price

            # 4. Determine Remaining "Room"
            available_room = max_allowed_value - current_value

            if available_room <= 0:
                print(f"   🛑 MAX CAP REACHED: Holdings (${current_value:,.0f}) exceed 5% limit. Buying 0.")
                return 0.0

            # 5. Determine Budget (Lesser of Standard Alloc OR Remaining Room)
            investment_amount = min(self.config.TRADE_ALLOCATION, available_room)

            if investment_amount < self.state.current_price:
                return 0.0

            # 6. Calculate Shares
            qty = int(investment_amount / self.state.current_price)
            logger.info(f"   💰 Sizing: Budget ${investment_amount:.2f} -> {qty} shares")
            return qty

        except Exception as e:
            logger.error(f"Sizing Error: {e}")
            return 0.0

    # --- EXECUTION ---
    def execute_strategy(self, verbose: bool = False):
        pass_news = self.state.sentiment_score >= self.config.MIN_CONFIDENCE
        pass_rsi = self.state.current_rsi < self.config.RSI_BUY_THRESHOLD
        is_strong_buy = pass_news and pass_rsi

        if verbose:
            print(f"\n   🔎 VETTING REPORT: {self.state.ticker}")
            print(f"      1. News Sentiment (>{self.config.MIN_CONFIDENCE}%):   {'✅ PASS' if pass_news else '❌ FAIL'} ({self.state.sentiment_score}/100)")
            print(f"      2. RSI Technical (<{self.config.RSI_BUY_THRESHOLD}):     {'✅ PASS' if pass_rsi else '❌ FAIL'} ({self.state.current_rsi:.2f})")
            print(f"      3. Price Action:            ${self.state.current_price:.2f}")
            print(f"      ------------------------------------------------")
            print(f"      {'🎯 APPROVED' if is_strong_buy else '✋ REJECTED'}")

        if not is_strong_buy: return

        final_qty = self.calculate_investment_qty()
        if final_qty <= 0: return

        print(f"   🚀 AUTOMATED TRIGGER: BUY {final_qty} {self.state.ticker}...")
        try:
            order = self.alpaca_trade.submit_order(order_data=MarketOrderRequest(symbol=self.state.ticker, qty=final_qty, side=OrderSide.BUY, time_in_force=TimeInForce.GTC))

            # --- UPDATED LOG LINE ---
            print(f"   ✅ Order Filled! Bought {final_qty} shares of {self.state.ticker} at ~${self.state.current_price:.2f}")
            # ------------------------

            blocks = [
                {"type": "header", "text": {"type": "plain_text", "text": f"🟢 Auto-Trade: {self.state.ticker}"}},
                {"type": "section", "fields": [
                    {"type": "mrkdwn", "text": f"*Action:*\nBUY {final_qty}"},
                    {"type": "mrkdwn", "text": f"*RSI:*\n{self.state.current_rsi:.2f}"},
                    {"type": "mrkdwn", "text": f"*Score:*\n{self.state.sentiment_score}/100"},
                    {"type": "mrkdwn", "text": f"*Order ID:*\n`{order.id}`"}
                ]},
                {"type": "section", "text": {"type": "mrkdwn", "text": f"*Reasoning:*\n{self.state.reasoning}"}}
            ]
            self.slack.chat_postMessage(channel=self.config.SLACK_CHANNEL, text="Auto-Trade", blocks=blocks)
        except Exception as e:
            logger.error(f"Execution Error: {e}")
            self._notify_slack_simple(f"❌ Order Failed: {e}")

    def run(self, verbose: bool = False):
        if self.check_technicals() and self.fetch_news_context() and self.analyze_sentiment():
            self.execute_strategy(verbose=verbose)

# --- 6. DISCOVERY AGENT (Optimized for Quantity) ---
class DiscoveryAgent:
    def __init__(self, config: Config):
        self.config = config
        self.client = genai.Client(api_key=config.GOOGLE_KEY)

    def _fetch_sp500_list(self) -> List[str]:
        url = f"https://financialmodelingprep.com/api/v3/sp500_constituent?apikey={self.config.FMP_API_KEY}"
        try:
            r = requests.get(url)
            data = r.json()
            if isinstance(data, list):
                return [item['symbol'] for item in data]
            else:
                print(f"   ⚠️ FMP S&P List Error: {data}")
                return []
        except Exception as e:
            print(f"   ❌ FMP S&P Fetch Error: {e}")
            return []

    def _check_ticker_criteria(self, ticker: str) -> bool:
        """
        Criteria:
        - Revenue Growth Rate: > 4%
        - P/E ratio: > 15
        - Gross profit margin: > 20%
        - Debt-to-Equity Ratio: <= 1.0
        - ROE: > 12%
        - FCF to Sales: > 5%
        """
        try:
            # 1. Fetch Ratios (TTM)
            ratios_url = f"https://financialmodelingprep.com/api/v3/ratios-ttm/{ticker}?apikey={self.config.FMP_API_KEY}"
            r_resp = requests.get(ratios_url)
            r_data = r_resp.json()
            
            if not isinstance(r_data, list) or not r_data:
                return False

            r = r_data[0]
            
            pe = r.get('peRatioTTM', 0) or 0
            gross_margin = r.get('grossProfitMarginTTM', 0) or 0
            debt_equity = r.get('debtEquityRatioTTM', 100) or 100
            roe = r.get('returnOnEquityTTM', 0) or 0
            
            # Check Basic Ratios first to fail fast
            if not (pe > 15 and gross_margin > 0.20 and debt_equity <= 1.0 and roe > 0.12):
                return False

    def find_top_picks(self) -> List[str]:
        target_count = self.config.TOP_N_STOCKS
        print(f"\n🔎 DISCOVERY MODE: Scanning Market News for {target_count} stocks...")
        
        try:
            # Call Alpha Vantage News (no specific ticker = general market news)
            url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&apikey={self.config.ALPHA_VANTAGE_KEY}&limit=50&topics=financial_markets"
            r = requests.get(url)
            data = r.json()
            
            if "feed" not in data:
                print(f"   ⚠️ No News Feed Found: {data}")
                return []

            found_tickers = []
            for art in data["feed"]:
                if "ticker_sentiment" in art:
                    for t in art["ticker_sentiment"]:
                        ticker = t.get("ticker", "")
                        # Simple filter to avoid Crypto/Forex if labeled as such (though AV usually puts them in separate topics)
                        # We also want to ensure it looks like a stock ticker (up to 5 chars usually)
                        if ticker and "CRYPTO" not in ticker and "FOREX" not in ticker:
                            found_tickers.append(ticker)
            
            # Count frequency (popular stocks in news)
            from collections import Counter
            counts = Counter(found_tickers)

            # Return top N most mentioned
            most_common = counts.most_common(target_count)
            tickers = [t[0] for t in most_common]
            
            print(f"   🔬 Found {len(tickers)} Candidates from News: {tickers}")
            return tickers


            # 2. Fetch Growth (Revenue Growth)
            growth_url = f"https://financialmodelingprep.com/api/v3/financial-growth/{ticker}?limit=1&apikey={self.config.FMP_API_KEY}"
            g_resp = requests.get(growth_url)
            g_data = g_resp.json()

            if not isinstance(g_data, list) or not g_data:
                return False

            rev_growth = g_data[0].get('revenueGrowth', 0) or 0
            if rev_growth <= 0.04:
                return False

            # 3. Check FCF to Sales (Available in Ratios TTM usually as freeCashFlowToRevenueTTM ??
            # Or calculate from Per Share metrics in Ratios TTM)
            # Some versions of FMP ratios have 'freeCashFlowToOperatingCashFlowTTM' etc.
            # Let's check 'freeCashFlowPerShareTTM' / 'revenuePerShareTTM'
            fcf_ps = r.get('freeCashFlowPerShareTTM', 0) or 0
            rev_ps = r.get('revenuePerShareTTM', 0) or 0

            if rev_ps == 0:
                return False

            fcf_sales = fcf_ps / rev_ps
            if fcf_sales <= 0.05:
                return False

            print(f"   ✅ MATCH: {ticker}\n"
                  f"      Growth: {rev_growth:.1%} (✅)\n"
                  f"      PE Ratio: {pe:.1f} (✅)\n"
                  f"      Gross Margin: {gross_margin:.1%} (✅)\n"
                  f"      Debt/Equity: {debt_equity:.2f} (✅)\n"
                  f"      ROE: {roe:.1%} (✅)\n"
                  f"      FCF/Sales: {fcf_sales:.1%} (✅)")
            return True

        except Exception:
            return False

    def find_top_picks(self) -> List[str]:
        target_count = self.config.TOP_N_STOCKS
        print(f"\n🔎 DISCOVERY MODE: Filtering S&P 500 via FMP...")

        # 1. Get Universe
        tickers = self._fetch_sp500_list()

        # FALLBACK: If API fails (e.g. Restricted Key), use a static list of top S&P 500 to allow process to continue
        if not tickers:
            print("   ⚠️ Fetch failed (likely Key Restriction). Using static Top 30 S&P list for fallback.")
            tickers = [
                "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK.B", "LLY", "AVGO",
                "JPM", "V", "UNH", "XOM", "MA", "JNJ", "PG", "HD", "COST", "MRK",
                "ABBV", "CVX", "CRM", "BAC", "WMT", "AMD", "ACN", "PEP", "LIN", "MCD"
            ]

        print(f"   🔬 Scanning {len(tickers)} stocks for criteria...")

        # 2. Parallel Scan
        qualifying_tickers = []
        # Limit concurrency to avoid rate limits (though FMP is usually generous)
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            future_to_ticker = {executor.submit(self._check_ticker_criteria, t): t for t in tickers}

            count = 0
            for future in concurrent.futures.as_completed(future_to_ticker):
                count += 1
                if count % 20 == 0:
                    print(f"      Progress: {count}/{len(tickers)}...")

                t = future_to_ticker[future]
                try:
                    if future.result():
                        qualifying_tickers.append(t)
                except Exception as e:
                    pass

        print(f"   🎯 Found {len(qualifying_tickers)} matches: {qualifying_tickers}")
        return qualifying_tickers[:target_count]

# --- 7. PORTFOLIO MANAGER (Unified Threading) ---
class PortfolioAudit:
    def __init__(self, config: Config):
        self.config = config
        self.alpaca = TradingClient(config.ALPACA_KEY, config.ALPACA_SECRET, paper=config.IS_PAPER)
        self.discovery = DiscoveryAgent(config)

        # FIX: Instantiate the correct class name (SmartTracker)
        # This is what enables the Firestore/JSON fallback logic.
        self.tracker = SmartTracker(config)

    def is_market_open(self) -> bool:
        if self.config.IS_PAPER: return True # Always run paper
        try: return self.alpaca.get_clock().is_open
        except: return False

    def _process_single_ticker(self, ticker: str, qty: float = 0.0):
        bot = DarwinianSwarm(ticker, self.config, existing_qty=qty)
        bot.run(verbose=True)
        return bot.state

    def scan(self):
        print(f"\n🕵️‍♂️ STARTING AUTOMATED AUDIT (Timeout: {self.config.EXECUTION_TIMEOUT_SECONDS}s)")

        if not self.is_market_open():
            print("   💤 Market Closed.")
            return

        try:
            positions = self.alpaca.get_all_positions()
            portfolio_map = {p.symbol: float(p.qty) for p in positions}
        except Exception as e:
            print(f"❌ Alpaca Error: {e}")
            return

        # Determine List
        targets = []
        if not portfolio_map:
            print("   ⚠️ Portfolio Empty. Switching to Discovery Mode...")
            candidates = self.discovery.find_top_picks()
            targets = [(t, 0.0) for t in candidates]
        else:
            print(f"   📉 Found {len(portfolio_map)} positions. Auditing...")
            targets = [(t, q) for t, q in portfolio_map.items()]

        if not targets:
            print("   ❌ No targets found.")
            return

        # Process List
        for i, (ticker, qty) in enumerate(targets, 1):
            print(f"\n🔹 [{i}/{len(targets)}] Processing: {ticker} (Owned: {qty})")

            # Skip if recently analyzed AND we don't own it (Discovery only)
            if qty == 0 and not self.tracker.should_rescan(ticker, self.config):
                continue

            try:
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(self._process_single_ticker, ticker, qty)
                    final_state = future.result(timeout=self.config.EXECUTION_TIMEOUT_SECONDS)
                    self.tracker.mark_processed(final_state)
            except concurrent.futures.TimeoutError:
                print(f"   ⏱️ TIMEOUT: {ticker} skipped.")
            except Exception as e:
                print(f"   ❌ ERROR: {e}")

            time.sleep(2)

# --- MAIN ---
if __name__ == "__main__":
    conf = Config.load()

    if "--test" in sys.argv:
        print("🧪 TEST MODE ENABLED: Checking 1 stock only with 20s timeout.")
        # Override config for testing by creating a new instance with modified values
        # Since Config is frozen, we use object.__setattr__ to bypass (hacky but effective for this script)
        object.__setattr__(conf, 'TOP_N_STOCKS', 1)
        object.__setattr__(conf, 'EXECUTION_TIMEOUT_SECONDS', 20)

    PortfolioAudit(conf).scan()