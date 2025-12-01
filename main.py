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

# --- THIRD PARTY IMPORTS ---
from google import genai
from google.genai import types
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import StockHistoricalDataClient, CryptoHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed

# COLAB DRIVE MOUNT
try:
    from google.colab import drive
    drive.mount('/content/drive')
    # Bump version to v4 to ensure clean slate with new logic
    PERSISTENT_FILE_PATH = "/content/drive/MyDrive/darwinian_processed_stocks_v4.json"
except:
    PERSISTENT_FILE_PATH = "processed_stocks_v4.json"

nest_asyncio.apply()

# --- 1. LOGGING ---
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
    "TRADE_ALLOCATION": 1000.0,
    "MIN_CONFIDENCE": 85,
    "RSI_BUY_THRESHOLD": 40,
    "TOP_N_STOCKS": 20,
    "ANALYSIS_COOLDOWN_HOURS": 24,
    "RSI_PERIOD": 14,
    "DATA_LOOKBACK_DAYS": 45,
    "SLACK_CHANNEL": "D0A1C7TBB5E",
    "EXECUTION_TIMEOUT_SECONDS": 60
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
    TRADE_ALLOCATION: float
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
            env_val = os.getenv(key)
            if env_val: return type_func(env_val)
            return type_func(USER_SETTINGS.get(key))

        secrets = {
            "GOOGLE_KEY": get_secret("GOOGLE_API_KEY"),
            "ALPACA_KEY": get_secret("ALPACA_API_KEY"),
            "ALPACA_SECRET": get_secret("ALPACA_SECRET"),
            "SLACK_TOKEN": get_secret("SLACK_BOT_TOKEN"),
            "SLACK_CHANNEL": get_param("SLACK_CHANNEL", str),
            "IS_PAPER": True,
            "TRADE_ALLOCATION": get_param("TRADE_ALLOCATION", float),
            "MIN_CONFIDENCE": get_param("MIN_CONFIDENCE", int),
            "RSI_BUY_THRESHOLD": get_param("RSI_BUY_THRESHOLD", int),
            "RSI_PERIOD": get_param("RSI_PERIOD", int),
            "DATA_LOOKBACK_DAYS": get_param("DATA_LOOKBACK_DAYS", int),
            "TOP_N_STOCKS": get_param("TOP_N_STOCKS", int),
            "ANALYSIS_COOLDOWN_HOURS": get_param("ANALYSIS_COOLDOWN_HOURS", int),
            "EXECUTION_TIMEOUT_SECONDS": get_param("EXECUTION_TIMEOUT_SECONDS", int)
        }

        missing = [k for k in ["GOOGLE_KEY", "ALPACA_KEY", "ALPACA_SECRET", "SLACK_TOKEN"] if not secrets[k] or secrets[k] == "YOUR_KEY"]
        if missing:
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

# --- 4. TRACKER (UPDATED LOGIC) ---
class TradeTracker:
    def __init__(self, filename=PERSISTENT_FILE_PATH):
        self.filename = filename
        self.processed: Dict[str, dict] = self._load_data()

    def _load_data(self) -> Dict[str, dict]:
        if not os.path.exists(self.filename): return {}
        try:
            with open(self.filename, 'r') as f:
                data = json.load(f)
                return data if isinstance(data, dict) else {}
        except: return {}

    def _save_data(self):
        try:
            with open(self.filename, 'w') as f:
                json.dump(self.processed, f, indent=4)
        except Exception as e:
            print(f"   ❌ Tracker Save Error: {e}")

    def should_rescan(self, ticker: str, config: Config) -> bool:
        """
        Determines if we should scan a stock.
        Returns TRUE if:
        1. It's new (never seen).
        2. Cooldown has expired.
        3. [NEW] Old analysis meets NEW stricter/looser parameters.
        """
        if ticker not in self.processed:
            return True # New stock

        try:
            entry = self.processed[ticker]

            # 1. Cooldown Check
            timestamp_str = entry.get("analysis_timestamp", "")
            if timestamp_str:
                last_time = datetime.fromisoformat(timestamp_str)
                if last_time.tzinfo is None: last_time = last_time.replace(tzinfo=timezone.utc)
                delta = datetime.now(timezone.utc) - last_time
                hours_ago = delta.total_seconds() / 3600

                if hours_ago >= config.ANALYSIS_COOLDOWN_HOURS:
                    print(f"   🔄 Re-scan {ticker}: Cooldown expired ({hours_ago:.1f}h)")
                    return True # Expired, go again

            # 2. Parameter Re-Evaluation (The "New Params" Logic)
            # If the stock failed before, but would pass NOW with new settings, we force a re-scan.
            last_sentiment = entry.get("sentiment_score", 0)
            last_rsi = entry.get("current_rsi", 100)

            passes_new_conf = last_sentiment >= config.MIN_CONFIDENCE
            passes_new_rsi = last_rsi < config.RSI_BUY_THRESHOLD

            if passes_new_conf and passes_new_rsi:
                print(f"   ⚡ Re-evaluating {ticker}: History matches new criteria (Sent:{last_sentiment}, RSI:{last_rsi:.1f})")
                return True

            print(f"   ⏭️ Skipping {ticker}: Analyzed {hours_ago:.1f}h ago (Wait: {config.ANALYSIS_COOLDOWN_HOURS}h)")
            return False

        except Exception as e:
            print(f"   ⚠️ Tracker Error ({e}). Re-scanning.")
            return True

    def mark_processed(self, state: MarketState):
        self.processed[state.ticker] = asdict(state)
        self._save_data()
        print(f"   💾 Saved analysis for {state.ticker}.")

# --- 5. CORE ANALYST AGENT ---
# --- 5. CORE ANALYST AGENT (DEBUG MODE) ---
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

    # --- PIPELINE STEP 1: TECHNICALS ---
    def check_technicals(self) -> bool:
        start = datetime.now(timezone.utc) - timedelta(days=self.config.DATA_LOOKBACK_DAYS)
        try:
            # DEBUG PRINT
            print(f"   📊 [{self.state.ticker}] Requesting Alpaca Data...")
            
            req = StockBarsRequest(
                symbol_or_symbols=self.state.ticker, 
                timeframe=TimeFrame.Hour, 
                start=start, 
                limit=200, 
                feed=DataFeed.IEX # <--- Note: Free accounts MUST use IEX. Paid use SIP.
            )
            bars = self.alpaca_stock_data.get_stock_bars(req)
            
            if not bars.data:
                print(f"   ❌ [{self.state.ticker}] No market data found. (Check: Is market open? Is Ticker valid?)")
                return False
            
            df = bars.df
            if isinstance(df.index, pd.MultiIndex): df = df.reset_index()
            df.columns = [c.lower() for c in df.columns]
            
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).fillna(0)
            loss = (-delta.where(delta < 0, 0)).fillna(0)
            avg_gain = gain.ewm(com=self.config.RSI_PERIOD-1, min_periods=self.config.RSI_PERIOD).mean()
            avg_loss = loss.ewm(com=self.config.RSI_PERIOD-1, min_periods=self.config.RSI_PERIOD).mean()
            rs = avg_gain / avg_loss
            df['rsi'] = 100 - (100 / (1 + rs))

            self.state.current_price = float(df['close'].iloc[-1])
            self.state.current_rsi = float(df['rsi'].iloc[-1])
            print(f"   ✅ [{self.state.ticker}] Technicals OK. Price: {self.state.current_price} RSI: {self.state.current_rsi:.2f}")
            return True
            
        except Exception as e:
            # LOUD ERROR PRINTING
            print(f"   ❌ [{self.state.ticker}] CRASH in Technicals: {str(e)}")
            return False

    # --- PIPELINE STEP 2: NEWS ---
    def fetch_news_context(self) -> bool:
        try:
            print(f"   📰 [{self.state.ticker}] Searching Google News...")
            today = datetime.now().strftime("%Y-%m-%d")
            prompt = f"Find latest news and analyst ratings for {self.state.ticker} as of {today}. Summarize in 3 bullet points."
            
            response = self.gemini.models.generate_content(
                model="gemini-2.5-flash-lite", 
                contents=prompt,
                config=types.GenerateContentConfig(tools=[types.Tool(google_search=types.GoogleSearch())], response_mime_type="text/plain")
            )
            self.state.news_summary = response.text if response.text else "No news."
            print(f"   ✅ [{self.state.ticker}] News found.")
            return True
        except Exception as e:
            # LOUD ERROR PRINTING
            print(f"   ❌ [{self.state.ticker}] CRASH in News: {str(e)}")
            return False

    # --- PIPELINE STEP 3: ANALYSIS ---
    def analyze_sentiment(self) -> bool:
        print(f"   🧠 [{self.state.ticker}] Thinking...")
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
        except Exception as e:
            # LOUD ERROR PRINTING
            print(f"   ❌ [{self.state.ticker}] CRASH in Analysis: {str(e)}")
            return False

    # ... (Keep execute_strategy and run as is) ...
    def execute_strategy(self, verbose: bool = False):
        pass_news = self.state.sentiment_score >= self.config.MIN_CONFIDENCE
        pass_rsi = self.state.current_rsi < self.config.RSI_BUY_THRESHOLD
        is_strong_buy = pass_news and pass_rsi

        if verbose:
            print(f"   🔎 REPORT: {self.state.ticker} | Sent: {self.state.sentiment_score} | RSI: {self.state.current_rsi:.1f} | {'✅ BUY' if is_strong_buy else '⛔ SKIP'}")

        if not is_strong_buy: return

        try:
            qty = int(self.config.TRADE_ALLOCATION / self.state.current_price)
            if qty < 1: return
            
            print(f"   🚀 EXECUTING: BUY {qty} {self.state.ticker}...")
            order = self.alpaca_trade.submit_order(order_data=MarketOrderRequest(symbol=self.state.ticker, qty=qty, side=OrderSide.BUY, time_in_force=TimeInForce.GTC))
            
            pnl_text = "N/A"
            try:
                time.sleep(1)
                pos = self.alpaca_trade.get_open_position(self.state.ticker)
                pnl_text = f"${float(pos.unrealized_pl):.2f}"
            except: pass

            blocks = [
                {"type": "header", "text": {"type": "plain_text", "text": f"🟢 Auto-Trade: {self.state.ticker}"}},
                {"type": "section", "fields": [
                    {"type": "mrkdwn", "text": f"*Action:*\nBUY {qty}"},
                    {"type": "mrkdwn", "text": f"*RSI:*\n{self.state.current_rsi:.2f}"},
                    {"type": "mrkdwn", "text": f"*Score:*\n{self.state.sentiment_score}/100"},
                    {"type": "mrkdwn", "text": f"*Unrealized PnL:*\n{pnl_text}"},
                    {"type": "mrkdwn", "text": f"*Order ID:*\n`{order.id}`"}
                ]},
                {"type": "section", "text": {"type": "mrkdwn", "text": f"*Reasoning:*\n{self.state.reasoning}"}}
            ]
            self.slack.chat_postMessage(channel=self.config.SLACK_CHANNEL, text="Auto-Trade", blocks=blocks)
        except Exception as e:
            print(f"   ❌ Trade Execution Failed: {e}")

    def run(self, verbose: bool = False):
        if not self.check_technicals(): return
        if not self.fetch_news_context(): return
        if not self.analyze_sentiment(): return
        self.execute_strategy(verbose=verbose)

# --- 6. DISCOVERY AGENT ---
class DiscoveryAgent:
    def __init__(self, config: Config):
        self.config = config
        self.client = genai.Client(api_key=config.GOOGLE_KEY)

    def find_top_picks(self) -> List[str]:
        target_count = self.config.TOP_N_STOCKS
        print(f"\n🔎 DISCOVERY MODE: Aggressively hunting for {target_count} stocks...")

        search_prompt = f"""
        Perform a comprehensive Google Search for current "Strong Buy" analyst ratings for S&P 500 stocks as of today.
        TASK: Generate a list of exactly {target_count} stock tickers.
        CRITERIA: Priority to >{self.config.MIN_CONFIDENCE}% Buy ratings. Fallback to "Moderate Buy".
        FORMAT: Return ONLY a list of Ticker Symbols (e.g. NVDA, AMD).
        """

        try:
            search_resp = self.client.models.generate_content(
                model="gemini-2.5-flash-lite", contents=search_prompt,
                config=types.GenerateContentConfig(tools=[types.Tool(google_search=types.GoogleSearch())], response_mime_type="text/plain")
            )
            if not search_resp.text: return []

            extract_prompt = f"Extract exactly {target_count} unique tickers from text. Return JSON list. Text: {search_resp.text}"
            json_resp = self.client.models.generate_content(
                model="gemini-2.5-flash-lite", contents=extract_prompt,
                config=types.GenerateContentConfig(response_mime_type="application/json", response_schema={"type": "ARRAY", "items": {"type": "STRING"}})
            )
            tickers = list(set(json.loads(json_resp.text)))
            print(f"   🔬 Candidates: {tickers}")
            return tickers
        except Exception as e:
            logger.error(f"Discovery Failed: {e}")
            return []

# --- 7. PORTFOLIO MANAGER (WITH RE-EVALUATION LOGIC) ---
class PortfolioAudit:
    def __init__(self, config: Config):
        self.config = config
        self.alpaca = TradingClient(config.ALPACA_KEY, config.ALPACA_SECRET, paper=config.IS_PAPER)
        self.discovery = DiscoveryAgent(config)
        self.tracker = TradeTracker()

    def _process_single_ticker(self, ticker: str):
        bot = DarwinianSwarm(ticker, self.config)
        bot.run(verbose=True)
        return bot.state

    def is_market_open(self) -> bool:
      """Checks if the US Stock Market is currently open."""
      try:
          clock = self.alpaca.get_clock()
          if clock.is_open:
              return True

          # Optional: Log when it opens next
          next_open = clock.next_open.strftime("%Y-%m-%d %H:%M UTC")
          print(f"   💤 Market is CLOSED. Next open: {next_open}")
          return False
      except Exception as e:
          logger.error(f"Market Clock Error: {e}")
          return False # Fail safe: Don't trade if we can't verify

    def scan(self):
        print(f"\n🕵️‍♂️ STARTING AUTOMATED AUDIT (Alloc: ${self.config.TRADE_ALLOCATION})")

        # --- NEW: MARKET HOURS CHECK ---
        # If paper trading, you might want to disable this to test at night
        if not self.config.IS_PAPER and not self.is_market_open():
            return

        # If you WANT to test logic during closed hours (Paper only):
        if self.config.IS_PAPER:
             print("   ⚠️ Paper Mode: Bypassing Market Open Check.")
        # -------------------------------

        try:
            positions = self.alpaca.get_all_positions()

        print(f"\n🕵️‍♂️ STARTING AUTOMATED AUDIT (Timeout: {self.config.EXECUTION_TIMEOUT_SECONDS}s)")
        try: positions = self.alpaca.get_all_positions()
        except: return

        if not positions:
            print("   ⚠️ Portfolio Empty. Discovery Mode Active...")
            candidates = self.discovery.find_top_picks()
            print(f"   🔬 Vetting {len(candidates)} Candidates...")

            for i, ticker in enumerate(candidates, 1):
                print(f"\n🔹 [{i}/{len(candidates)}] Processing: {ticker}")

                # --- NEW LOGIC: Check Tracker via should_rescan ---
                if not self.tracker.should_rescan(ticker, self.config):
                    continue # Skip this one

                try:
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(self._process_single_ticker, ticker)
                        final_state = future.result(timeout=self.config.EXECUTION_TIMEOUT_SECONDS)
                        self.tracker.mark_processed(final_state)
                except concurrent.futures.TimeoutError:
                    print(f"   ⏱️ TIMEOUT: {ticker} took > {self.config.EXECUTION_TIMEOUT_SECONDS}s. Skipping.")
                except Exception as e:
                    print(f"   ❌ CRITICAL FAIL on {ticker}: {e}")

                time.sleep(2)
        else:
            print(f"   📉 Found {len(positions)} positions. Auditing...")
            for i, p in enumerate(positions, 1):
                print(f"\n👉 [{i}/{len(positions)}] Analyzing {p.symbol}...")
                DarwinianSwarm(p.symbol, self.config, existing_qty=float(p.qty)).run(verbose=False)
                time.sleep(1)

if __name__ == "__main__":
    conf = Config.load()
    PortfolioAudit(conf).scan()