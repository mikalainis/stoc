import os
import json
import logging
import sys
import time
import math
import concurrent.futures
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Dict, Union

# --- THIRD PARTY IMPORTS ---
import nest_asyncio
from google import genai
from google.genai import types
from google.cloud import storage # Required for GCS Persistence
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import StockHistoricalDataClient, CryptoHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed

# Patch for async environments
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
    "MIN_CONFIDENCE": 90,
    "RSI_BUY_THRESHOLD": 40,
    "TOP_N_STOCKS": 20,
    "ANALYSIS_COOLDOWN_HOURS": 24,
    "RSI_PERIOD": 14,
    "DATA_LOOKBACK_DAYS": 45,
    "EXECUTION_TIMEOUT_SECONDS": 60,
    # Note: Secrets (API Keys) are loaded from Env Vars
}

# --- 2. CONFIGURATION MANAGEMENT ---
@dataclass(frozen=True)
class Config:
    GOOGLE_KEY: str
    ALPACA_KEY: str
    ALPACA_SECRET: str
    SLACK_TOKEN: str
    SLACK_CHANNEL: str
    IS_PAPER: bool
    GCS_BUCKET_NAME: str # New: For Cloud Memory
    
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
        # Helper to fetch Env or Settings
        def get_param(key, type_func):
            env_val = os.getenv(key)
            if env_val: return type_func(env_val)
            return type_func(USER_SETTINGS.get(key))

        secrets = {
            "GOOGLE_KEY": os.getenv("GOOGLE_API_KEY"),
            "ALPACA_KEY": os.getenv("ALPACA_API_KEY"),
            "ALPACA_SECRET": os.getenv("ALPACA_SECRET"),
            "SLACK_TOKEN": os.getenv("SLACK_BOT_TOKEN"),
            "SLACK_CHANNEL": os.getenv("SLACK_CHANNEL", "D0A1C7TBB5E"),
            "GCS_BUCKET_NAME": os.getenv("GCS_BUCKET_NAME"), # Loaded from Cloud Run Env
            
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

        missing = [k for k in ["GOOGLE_KEY", "ALPACA_KEY", "ALPACA_SECRET", "SLACK_TOKEN"] if not secrets[k]]
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

# --- 4. CLOUD PERSISTENCE TRACKER (GCS ENABLED) ---
class TradeTracker:
    def __init__(self, config: Config, filename="processed_stocks_v4.json"):
        self.filename = filename
        self.bucket_name = config.GCS_BUCKET_NAME
        
        # Initialize Storage Client only if Bucket provided
        self.storage_client = storage.Client() if self.bucket_name else None
        self.processed: Dict[str, dict] = self._load_data()

    def _load_data(self) -> Dict[str, dict]:
        """Loads from GCS Bucket or Local Disk."""
        if self.bucket_name:
            try:
                bucket = self.storage_client.bucket(self.bucket_name)
                blob = bucket.blob(self.filename)
                if not blob.exists(): return {}
                return json.loads(blob.download_as_text())
            except Exception as e:
                logger.error(f"GCS Load Error: {e}")
                return {}
        else:
            # Fallback for local testing
            if not os.path.exists(self.filename): return {}
            try:
                with open(self.filename, 'r') as f: return json.load(f)
            except: return {}

    def _save_data(self):
        """Saves to GCS Bucket or Local Disk."""
        data_str = json.dumps(self.processed, indent=4)
        if self.bucket_name:
            try:
                bucket = self.storage_client.bucket(self.bucket_name)
                blob = bucket.blob(self.filename)
                blob.upload_from_string(data_str)
                print(f"   ☁️ Saved to GCS: {self.bucket_name}")
            except Exception as e:
                logger.error(f"GCS Save Error: {e}")
        else:
            try:
                with open(self.filename, 'w') as f: f.write(data_str)
            except: pass

    def should_rescan(self, ticker: str, config: Config) -> bool:
        if ticker not in self.processed: return True
        try:
            entry = self.processed[ticker]
            # 1. Cooldown Check
            ts = entry.get("analysis_timestamp", "")
            if ts:
                last_time = datetime.fromisoformat(ts).replace(tzinfo=timezone.utc)
                hours_ago = (datetime.now(timezone.utc) - last_time).total_seconds() / 3600
                if hours_ago >= config.ANALYSIS_COOLDOWN_HOURS:
                    return True # Expired

            # 2. Re-evaluation (Did parameters change?)
            last_sent = entry.get("sentiment_score", 0)
            last_rsi = entry.get("current_rsi", 100)
            if last_sent >= config.MIN_CONFIDENCE and last_rsi < config.RSI_BUY_THRESHOLD:
                print(f"   ⚡ Re-evaluating {ticker}: Meets new criteria.")
                return True
                
            return False
        except: return True

    def mark_processed(self, state: MarketState):
        self.processed[state.ticker] = asdict(state)
        self._save_data()

# --- 5. CORE AGENT ---
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

    # Technicals
    def check_technicals(self) -> bool:
        start = datetime.now(timezone.utc) - timedelta(days=self.config.DATA_LOOKBACK_DAYS)
        try:
            req = StockBarsRequest(symbol_or_symbols=self.state.ticker, timeframe=TimeFrame.Hour, start=start, limit=200, feed=DataFeed.IEX)
            bars = self.alpaca_stock_data.get_stock_bars(req)
            if not bars.data: return False
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
            return True
        except: return False

    # News
    def fetch_news_context(self) -> bool:
        try:
            today = datetime.now().strftime("%Y-%m-%d")
            prompt = f"Find latest news and analyst ratings for {self.state.ticker} as of {today}. Summarize in 3 bullet points."
            response = self.gemini.models.generate_content(
                model="gemini-2.5-flash-lite", contents=prompt,
                config=types.GenerateContentConfig(tools=[types.Tool(google_search=types.GoogleSearch())], response_mime_type="text/plain")
            )
            self.state.news_summary = response.text if response.text else "No news."
            return True
        except: return False

    # Analysis
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
            
            # PnL Metrics
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
            logger.error(f"Trade Error: {e}")

    def run(self, verbose: bool = False):
        if self.check_technicals() and self.fetch_news_context() and self.analyze_sentiment():
            self.execute_strategy(verbose=verbose)

# --- 6. DISCOVERY ---
class DiscoveryAgent:
    def __init__(self, config: Config):
        self.config = config
        self.client = genai.Client(api_key=config.GOOGLE_KEY)

    def find_top_picks(self) -> List[str]:
        target = self.config.TOP_N_STOCKS
        print(f"\n🔎 DISCOVERY: Hunting {target} stocks >{self.config.MIN_CONFIDENCE}% Buy...")
        try:
            # 1. Search
            prompt = f"Find 'Strong Buy' S&P 500 stocks. Return exactly {target} tickers. Priority >{self.config.MIN_CONFIDENCE}% Buy ratings."
            search_resp = self.client.models.generate_content(
                model="gemini-2.5-flash-lite", contents=prompt,
                config=types.GenerateContentConfig(tools=[types.Tool(google_search=types.GoogleSearch())], response_mime_type="text/plain")
            )
            if not search_resp.text: return []
            
            # 2. Extract
            extract = f"Extract tickers from text. Return JSON list of {target} strings. Text: {search_resp.text}"
            json_resp = self.client.models.generate_content(
                model="gemini-2.5-flash-lite", contents=extract,
                config=types.GenerateContentConfig(response_mime_type="application/json", response_schema={"type": "ARRAY", "items": {"type": "STRING"}})
            )
            return list(set(json.loads(json_resp.text)))
        except: return []

# --- 7. MANAGER ---
class PortfolioAudit:
    def __init__(self, config: Config):
        self.config = config
        self.alpaca = TradingClient(config.ALPACA_KEY, config.ALPACA_SECRET, paper=config.IS_PAPER)
        self.discovery = DiscoveryAgent(config)
        self.tracker = TradeTracker(config)

    def _process(self, ticker: str):
        bot = DarwinianSwarm(ticker, self.config)
        bot.run(verbose=True)
        return bot.state

    def scan(self):
        print(f"\n🕵️‍♂️ STARTING AUDIT (Timeout: {self.config.EXECUTION_TIMEOUT_SECONDS}s)")
        try: positions = self.alpaca.get_all_positions()
        except: return

        if not positions:
            print("   ⚠️ Portfolio Empty. Discovery Mode...")
            candidates = self.discovery.find_top_picks()
            print(f"   🔬 Vetting {len(candidates)} Candidates...")
            
            for i, ticker in enumerate(candidates, 1):
                print(f"\n🔹 [{i}/{len(candidates)}] Processing: {ticker}")
                
                if not self.tracker.should_rescan(ticker, self.config): continue

                try:
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(self._process, ticker)
                        final_state = future.result(timeout=self.config.EXECUTION_TIMEOUT_SECONDS)
                        self.tracker.mark_processed(final_state)
                except concurrent.futures.TimeoutError:
                    print(f"   ⏱️ TIMEOUT: {ticker} skipped.")
                except Exception as e:
                    print(f"   ❌ ERROR: {e}")
                time.sleep(2)
        else:
            print(f"   📉 Found {len(positions)} positions. Auditing...")
            for p in positions:
                print(f"\n👉 Analyzing {p.symbol}...")
                DarwinianSwarm(p.symbol, self.config, existing_qty=float(p.qty)).run(verbose=False)
                time.sleep(1)

if __name__ == "__main__":
    conf = Config.load()
    PortfolioAudit(conf).scan()