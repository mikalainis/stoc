import os
import pandas as pd
import numpy as np
import json
import logging
import sys
import time
import concurrent.futures
import traceback
import requests
import random
from io import StringIO # Required for the Wiki Fix
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Dict, Any, Union

# --- THIRD PARTY IMPORTS ---
import yfinance as yf
import google.auth
import google.auth.exceptions

# Graceful imports
try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None

try:
    from google.cloud import firestore
except ImportError:
    firestore = None

from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOrdersRequest
from alpaca.trading.enums import QueryOrderStatus
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

# --- 1. LOGGING CONFIGURATION ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("DarwinianSwarm")

# --- 2. AUTHENTICATION & CONFIG ---
def setup_google_auth():
    print("\n🔐 AUTHENTICATION CHECK")
    try:
        from google.colab import auth
        print("   ☁️  Detected Google Colab. Authenticating...")
        auth.authenticate_user()
        print("   ✅ Colab Authentication Successful.")
    except ImportError:
        print("   🖥️  Local Environment detected (Non-Colab).")
    except Exception as e:
        print(f"   ⚠️  Auth Warning: {e}")

@dataclass(frozen=True)
class Config:
    # ---------------------------------------------------------
    # 1. REQUIRED KEYS (Must come first, no defaults)
    # ---------------------------------------------------------
    GOOGLE_KEY: str
    ALPACA_KEY: str
    ALPACA_SECRET: str
    SLACK_TOKEN: str
    SLACK_CHANNEL: str
    IS_PAPER: bool
    ALPHA_VANTAGE_KEY: str

    # 🚨 EMAIL FIELDS HAVE BEEN REMOVED FROM HERE

    # ---------------------------------------------------------
    # 2. OPTIONAL SETTINGS (Must come last, with defaults)
    # ---------------------------------------------------------
    TRADE_ALLOCATION: float = 3000.0
    MAX_POS_PERCENT: float = 0.05
    RSI_BUY_THRESHOLD: int = 35
    RSI_SELL_THRESHOLD: int = 70
    RSI_PERIOD: int = 14
    DATA_LOOKBACK_DAYS: int = 45
    MIN_CONFIDENCE: int = 85
    ANALYSIS_COOLDOWN_HOURS: int = 24
    TOP_N_STOCKS: int = 40
    
    # Fundamentals
    MIN_GROWTH_RATE: float = 0.04
    MIN_PE_RATIO: float = 15.0
    MIN_GROSS_MARGIN: float = 0.20
    MAX_DEBT_EQUITY: float = 1.0
    MIN_ROE: float = 0.12

    # Sell Logic
    STOP_LOSS_PCT: float = 0.08
    MIN_AI_HOLD_SCORE: int = 40

    @classmethod
    def load(cls) -> "Config":
        def get_secret(key, default=None):
            try:
                from google.colab import userdata
                return userdata.get(key)
            except: pass
            return os.getenv(key, default)

        # 1. Load Keys
        google = get_secret("GOOGLE_API_KEY")
        alpaca = get_secret("ALPACA_API_KEY")
        alpaca_sec = get_secret("ALPACA_SECRET")
        slack_token = get_secret("SLACK_BOT_TOKEN")

        # 2. Load Optional Keys
        fmp = get_secret("FMP_API_KEY", "demo")
        alpha = get_secret("ALPHA_VANTAGE_KEY", "demo")
        channel = get_secret("SLACK_CHANNEL", "#general")

        # TEST MODE SECRET BYPASS
        if "--test" in sys.argv:
            google = google or "dummy_google"
            alpaca = alpaca or "dummy_alpaca"
            alpaca_sec = alpaca_sec or "dummy_secret"
            slack_token = slack_token or "dummy_slack"

        if not google or not alpaca or not alpaca_sec:
            raise ValueError("❌ CRITICAL: Missing Keys in Colab Secrets.")

        return cls(
                    GOOGLE_KEY=google,
                    ALPACA_KEY=alpaca,
                    ALPACA_SECRET=alpaca_sec,
                    SLACK_TOKEN=slack_token,
                    SLACK_CHANNEL=channel,
                    IS_PAPER=True,
                    ALPHA_VANTAGE_KEY=alpha,
                    # FMP line is GONE

                    STOP_LOSS_PCT=float(os.getenv("STOP_LOSS_PCT", 0.08)),
                    RSI_SELL_THRESHOLD=int(os.getenv("RSI_SELL_THRESHOLD", 70)),
                    TOP_N_STOCKS=int(os.getenv("TOP_N_STOCKS", 40))
                )

# --- 3. DISCOVERY AGENT (The Wiki Fix) ---
class DiscoveryAgent:
    def __init__(self, config: Config):
        self.config = config

    def get_universe(self) -> List[str]:
        # TEST MODE OVERRIDE
        if "--test" in sys.argv:
            print("\n🌌 DISCOVERY AGENT: Test Mode - Returning Single Stock (AAPL).")
            return ["AAPL"]

        print("\n🌌 DISCOVERY AGENT: Fetching S&P 500 Universe (Wikipedia)...")

        # 1. Base Layer (Empty - relies purely on Config now)
        universe = set()

        # 2. Dynamic Layer (S&P 500)
        sp500 = self._fetch_sp500_tickers()

        if sp500:
            # Use EXACTLY the number from Config
            target_count = self.config.TOP_N_STOCKS
            
            # Ensure we don't crash if target > available
            picks = random.sample(sp500, min(len(sp500), target_count))
            
            universe.update(picks)
            print(f"   🎲 Added {len(picks)} random S&P 500 picks (Target: {target_count}).")

        return list(universe)

    def _fetch_sp500_tickers(self) -> List[str]:
        try:
            url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
            r = requests.get(url, headers=headers)
            tables = pd.read_html(StringIO(r.text))
            tickers = tables[0]['Symbol'].tolist()
            return [t.replace('.', '-') for t in tickers]
        except Exception as e:
            print(f"   ⚠️ S&P 500 Fetch Error: {e}")
            # Fallback only if Wiki fails
            return ["SPY", "QQQ", "IWM"]

# --- 4. STATE & PERSISTENCE ---
@dataclass
class MarketState:
    ticker: str
    analysis_timestamp: str = ""
    decision: str = "HOLD"
    sentiment_score: int = 0
    reasoning: str = ""
    current_price: float = 0.0
    current_rsi: float = 0.0
    position_qty: float = 0.0

    # Fundamentals
    pe_ratio: float = 0.0
    growth_rate: float = 0.0
    gross_margin: float = 0.0
    debt_equity: float = 0.0
    roe: float = 0.0
    fundamentals_valid: bool = False

class HybridTracker:
    def __init__(self, use_cloud: bool = True, local_file: str = "processed_stocks.json"):
        self.use_cloud = False
        self.local_file = local_file
        self.local_data = self._load_local()
        
        if use_cloud and firestore:
            print("   🔗 Connecting to Database...")
            try:
                self.db = firestore.Client(database="stocs")
                self.collection = self.db.collection("darwinian_analysis")
                self.collection.limit(1).get()
                self.use_cloud = True
                print("   🔥 Connected to Firestore.")
            except Exception:
                print("   ⚠️ Cloud Connect Failed. Using Local JSON.")

    def _load_local(self) -> Dict:
        if not os.path.exists(self.local_file): return {}
        try:
            with open(self.local_file, 'r') as f: return json.load(f)
        except: return {}

    def _save_local(self):
        def default_serializer(obj):
            if isinstance(obj, (datetime)): return obj.isoformat()
            return str(obj)
        with open(self.local_file, 'w') as f:
            json.dump(self.local_data, f, indent=4, default=default_serializer)

    def should_rescan(self, ticker: str, cooldown_hours: int) -> bool:
        last_ts = ""
        if self.use_cloud:
            try:
                doc = self.collection.document(ticker).get()
                if doc.exists: last_ts = doc.to_dict().get("analysis_timestamp", "")
            except: pass
        else:
            if ticker in self.local_data:
                last_ts = self.local_data[ticker].get("analysis_timestamp", "")
        
        if not last_ts: return True
        try:
            last_dt = datetime.fromisoformat(last_ts).replace(tzinfo=timezone.utc)
            delta = (datetime.now(timezone.utc) - last_dt).total_seconds() / 3600
            return delta >= cooldown_hours
        except: return True

    def save_state(self, state: MarketState):
        record = asdict(state)
        if self.use_cloud:
            try:
                cloud_rec = record.copy()
                cloud_rec["updated_at"] = firestore.SERVER_TIMESTAMP
                self.collection.document(state.ticker).set(cloud_rec, merge=True)
            except Exception: pass

        record["updated_at"] = datetime.now(timezone.utc).isoformat()
        self.local_data[state.ticker] = record
        self._save_local()

# --- 5. CORE ANALYST ---
# --- 5. ANALYST (Updated with Pro RSI) ---
class DarwinianAnalyst:
    def __init__(self, ticker: str, config: Config, qty: float):
        self.ticker = ticker.upper()
        self.config = config
        self.state = MarketState(ticker=self.ticker)
        self.alpaca_data = StockHistoricalDataClient(config.ALPACA_KEY, config.ALPACA_SECRET)
        if genai: self.gemini = genai.Client(api_key=config.GOOGLE_KEY)

    def _calculate_rsi(self, series: pd.Series, period: int) -> pd.Series:
        """
        Helper: Calculates RSI using Wilder's Smoothing (Industry Standard).
        """
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).fillna(0)
        loss = (-delta.where(delta < 0, 0)).fillna(0)

        # Wilder's Smoothing: alpha = 1/n, which equals com = n - 1 in Pandas
        avg_gain = gain.ewm(com=period-1, min_periods=period).mean()
        avg_loss = loss.ewm(com=period-1, min_periods=period).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def fetch_technicals(self) -> bool:
        # 1. Fetch Data
        start = datetime.now(timezone.utc) - timedelta(days=self.config.DATA_LOOKBACK_DAYS)
        try:
            req = StockBarsRequest(symbol_or_symbols=self.ticker, timeframe=TimeFrame.Hour, start=start, limit=200)
            bars = self.alpaca_data.get_stock_bars(req)
            if not bars.data: return False
            
            df = bars.df
            if isinstance(df.index, pd.MultiIndex): df = df.reset_index()
            df.columns = [c.lower() for c in df.columns]

            # 2. Calculate RSI (Using the Pro Helper)
            # We pass the 'close' column and the period from config
            df['rsi'] = self._calculate_rsi(df['close'], self.config.RSI_PERIOD)

            # 3. Store Results
            self.state.current_price = float(df['close'].iloc[-1])
            self.state.current_rsi = float(df['rsi'].iloc[-1])
            return True
        except Exception as e:
            print(f"      ⚠️ Technicals Error: {e}")
            return False

    def fetch_fundamentals(self) -> bool:
        try:
            print(f"      (Fetching fundamentals via Yahoo Finance for {self.ticker}...)")
            stock = yf.Ticker(self.ticker)
            info = stock.info
            self.state.pe_ratio = info.get('trailingPE', 0) or info.get('forwardPE', 0) or 0
            self.state.growth_rate = info.get('revenueGrowth', 0) or 0
            self.state.gross_margin = info.get('grossMargins', 0) or 0
            self.state.roe = info.get('returnOnEquity', 0) or 0
            self.state.debt_equity = info.get('debtToEquity', 0) / 100.0 if info.get('debtToEquity') else 0

            if self.state.pe_ratio == 0 and self.state.growth_rate == 0: return False
            self.state.fundamentals_valid = True
            return True
        except Exception as e:
            print(f"   ❌ News Error: {e}")
            return False

    def fetch_news(self) -> bool:
        try:
            url = f"https://news.google.com/rss/search?q={self.ticker}+stock&hl=en-US&gl=US&ceid=US:en"
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
            resp = requests.get(url, headers=headers, timeout=5)

            # Simple XML parse
            from xml.etree import ElementTree as ET
            from html import unescape

            root = ET.fromstring(resp.content)
            items = root.findall('.//item')[:3]
            summary = [f"- {unescape(i.find('title').text)}" for i in items]

            if not summary:
                self.state.news_summary = "No news found."
                return True

            self.state.news_summary = "\n".join(summary)
            print(f"      📰 Found {len(summary)} news articles.")
            return True
        except Exception as e:
            print(f"      ⚠️ News Error: {e}")
            self.state.news_summary = "Error fetching news."
            return False

    def run_ai(self) -> bool:
        if not genai: return False

        if not self.state.news_summary or self.state.news_summary in ["No news found.", "Error fetching news."]:
            print("      🛡️ No News Data. Defaulting to HOLD (Score 50).")
            self.state.sentiment_score = 50
            self.state.decision = "HOLD"
            self.state.reasoning = "Insufficient Data"
            return True

        prompt = f"""
        Analyze {self.ticker}. Price: {self.state.current_price}, RSI: {self.state.current_rsi}.
        News: {self.state.news_summary}
        Fundamentals: PE {self.state.pe_ratio:.1f}, Growth {self.state.growth_rate:.1%}.
        Goal: Buy if RSI < {self.config.RSI_BUY_THRESHOLD} AND Good News/Fundamentals.
        Output JSON: {{ "sentiment_score": int(0-100), "decision": "BUY/HOLD", "reasoning": "short summary" }}
        """
        try:
            resp = self.gemini.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt,
                config=types.GenerateContentConfig(response_mime_type="application/json")
            )
            d = json.loads(resp.text)
            self.state.sentiment_score = int(d.get("sentiment_score", 50))
            self.state.decision = d.get("decision", "HOLD")
            self.state.reasoning = d.get("reasoning", "")
            return True
        except Exception as e:
            print(f"      ⚠️ AI Error: {e}. Defaulting to 50.")
            self.state.sentiment_score = 50
            self.state.decision = "HOLD"
            return False

    def print_vetting_report(self):
        s = self.state
        cfg = self.config
        def check(condition, val, suffix=""): return f"{val}{suffix} (✅)" if condition else f"{val}{suffix} (❌)"
        print(f"\n  🔎 VETTING REPORT: {s.ticker}")
        print(f"      {'-'*48}")
        if s.fundamentals_valid:
            print(f"      Growth:       {check(s.growth_rate > cfg.MIN_GROWTH_RATE, f'{s.growth_rate:.1%}')}")
            print(f"      PE Ratio:     {check(s.pe_ratio > cfg.MIN_PE_RATIO, f'{s.pe_ratio:.1f}')}")
        print(f"      1. Score (>{cfg.MIN_CONFIDENCE}):       {'✅' if s.sentiment_score>=cfg.MIN_CONFIDENCE else '❌'} ({s.sentiment_score})")
        print(f"      2. RSI (<{cfg.RSI_BUY_THRESHOLD}):            {'✅' if s.current_rsi<cfg.RSI_BUY_THRESHOLD else '❌'} ({s.current_rsi:.2f})")
        print(f"      {'-'*48}")
        print(f"      {'🎯 APPROVED' if (s.sentiment_score>=cfg.MIN_CONFIDENCE and s.current_rsi<cfg.RSI_BUY_THRESHOLD) else '✋ REJECTED'}")

class MacroAnalyst:
    """
    The 'MacroAnalyst' determines the current Market Regime (Bull, Bear, or Neutral).

    LOGIC:
    1. Fetches S&P 500 (SPY) price history to determine the long-term trend.
    2. Fetches Volatility Index (VIX) to measure market fear.
    3. Dynamically overrides the frozen 'Config' parameters to adapt to the regime.

    REGIMES:
    - 🐂 BULL: SPY > 200-Day SMA AND VIX < 20. (Risk On)
    - 🐻 BEAR: SPY < 200-Day SMA OR VIX > 30. (Risk Off)
    - 🦀 NEUTRAL: Sideways market. (Standard Settings)
    """
    def __init__(self, config: Config):
        self.config = config

    def analyze_and_adapt(self) -> str:
        print("\n🌍 MACRO ANALYSIS: Checking Market Regime...")

        try:
            # 1. FETCH DATA
            # We need 1 year (252 trading days) to calculate the 200-day Moving Average (SMA).
            # We need the VIX to check for immediate panic.
            spy = yf.Ticker("SPY").history(period="1y")
            vix = yf.Ticker("^VIX").history(period="5d")
            
            if spy.empty or vix.empty:
                print("   ⚠️ Macro Data Unavailable (Network Error?). Defaulting to NEUTRAL.")
                return "NEUTRAL"

            # 2. CALCULATE METRICS
            current_spy = spy['Close'].iloc[-1]
            # The 200-day SMA is the "Line in the Sand" for institutional investors.
            # Above = Bull Market. Below = Bear Market.
            sma_200 = spy['Close'].rolling(window=200).mean().iloc[-1]
            
            current_vix = vix['Close'].iloc[-1]
            
            # 3. DEFINE LOGIC GATES
            is_uptrend = current_spy > sma_200      # Is price above long-term average?
            is_extreme_fear = current_vix > 30      # Is VIX screaming panic?
            is_calm = current_vix < 20              # Is VIX signaling complacency/safety?

            print(f"   📊 SPY: ${current_spy:.2f} (200-SMA: ${sma_200:.2f}) | VIX: {current_vix:.2f}")

            # 4. DETERMINE REGIME
            
            # --- SCENARIO A: BEAR / DEFENSIVE ---
            # If the trend is broken OR fear is extreme, we must protect capital.
            if not is_uptrend or is_extreme_fear:
                return self._activate_bear_mode(current_vix, current_spy, sma_200)
            
            # --- SCENARIO B: BULL / AGGRESSIVE ---
            # If trend is up AND markets are calm, we can be greedy.
            elif is_uptrend and is_calm:
                return self._activate_bull_mode()
            
            # --- SCENARIO C: NEUTRAL ---
            # Sideways chop. Keep standard settings.
            else:
                print("   ⚖️  Regime: NEUTRAL. Trend is up but Volatility is elevated. Keeping Defaults.")
                return "NEUTRAL"

        except Exception as e:
            print(f"   ⚠️ Macro Analysis Failed: {e}. Keeping Defaults.")
            return "ERROR"

    def _activate_bear_mode(self, vix, spy, sma):
        reason = "VIX Panic (>30)" if vix > 30 else "SPY Downtrend (<SMA200)"
        print(f"   🐻 Regime: BEAR / DEFENSIVE ({reason})")
        print("      👉 ACTION: Tightening Stops, Reducing Size, Requiring Deep Dips.")

        # LOGIC:
        # 1. RSI < 25: Don't buy the dip unless it's a CRASH. Regular dips keep dipping in a bear market.
        # 2. Stop Loss 5%: Cut losers fast. Do not hold bags.
        # 3. Allocation $1500: Keep cash on the sidelines.

        # We use object.__setattr__ to bypass the 'frozen=True' protection of the Config class
        object.__setattr__(self.config, 'RSI_BUY_THRESHOLD', 25)
        object.__setattr__(self.config, 'STOP_LOSS_PCT', 0.05)
        object.__setattr__(self.config, 'MIN_CONFIDENCE', 90) # Be picky
        object.__setattr__(self.config, 'TRADE_ALLOCATION', self.config.TRADE_ALLOCATION * 0.5)

        print(f"      [UPDATED] RSI Buy < 25 | Stop Loss 5% | Alloc ${self.config.TRADE_ALLOCATION:.0f}")
        return "BEAR"

    def _activate_bull_mode(self):
        print("   🐂 Regime: BULL / AGGRESSIVE (Trend Up + Low Volatility)")
        print("      👉 ACTION: Loosening Stops, Buying Shallower Dips.")

        # LOGIC:
        # 1. RSI < 45: In a strong bull market, stocks rarely hit 30. We buy sooner (FOMO mode).
        # 2. Stop Loss 12%: Give winning stocks room to breathe (volatility is upside).
        # 3. PE Ratio: Ignore high valuations (momentum matters more).

        object.__setattr__(self.config, 'RSI_BUY_THRESHOLD', 45)
        object.__setattr__(self.config, 'STOP_LOSS_PCT', 0.12)
        object.__setattr__(self.config, 'MIN_PE_RATIO', 5.0) # Lower quality filter to catch growth runners

        print(f"      [UPDATED] RSI Buy < 45 | Stop Loss 12% | PE Filter Relaxed")
        return "BULL"

# --- 7. MANAGER ---
class PortfolioManager:
    def __init__(self, config: Config):
        self.config = config
        self.alpaca = TradingClient(config.ALPACA_KEY, config.ALPACA_SECRET, paper=config.IS_PAPER)
        self.tracker = HybridTracker(use_cloud=True)
        self.slack = WebClient(token=config.SLACK_TOKEN)
        self.discovery = DiscoveryAgent(config)
        self.scan_results: List[Dict] = []
        self.start_time = datetime.now()

    def process(self, ticker: str, position_data: Dict, force: bool = False):
        # 1. Unpack Position Data
        qty_held = position_data.get('qty', 0.0)
        entry_price = position_data.get('entry', 0.0)
        is_owned = qty_held > 0
        final_qty = qty_held

        # 2. Cooldown Check
        if not is_owned and not force and not self.tracker.should_rescan(ticker, self.config.ANALYSIS_COOLDOWN_HOURS):
            return

        # 3. Initialize Record
        result_entry = {
            "Ticker": ticker, "Status": "Owned" if is_owned else "New", "Action": "SKIP",
            "Reason": "Cooldown/Error", "Price": 0.0, "RSI": 0.0, "Score": 0, "PL": "-",
            "Qty": 0, "TotalPos": "$0.00", "Growth": "-", "PE": "-", "Margin": "-", "Debt": "-", "ROE": "-"
        }

        try:
            print(f"\n🔹 Processing: {ticker} (Owned: {qty_held} @ ${entry_price:.2f})")
            agent = DarwinianAnalyst(ticker, self.config, qty_held)
            
            # --- PIPELINE: Technicals ---
            if not agent.fetch_technicals():
                result_entry["Reason"] = "Tech Data Missing"
                self.scan_results.append(result_entry)
                return

            curr_price = agent.state.current_price
            result_entry.update({"Price": round(curr_price, 2), "RSI": round(agent.state.current_rsi, 2)})

            # --- SELL LOGIC ---
            if is_owned:
                pl_pct = (curr_price - entry_price) / entry_price if entry_price > 0 else 0
                result_entry["PL"] = f"{pl_pct:.1%}"

                sell_reason = ""
                if pl_pct < -self.config.STOP_LOSS_PCT:
                    sell_reason = f"Stop Loss ({pl_pct:.1%})"
                elif agent.state.current_rsi > self.config.RSI_SELL_THRESHOLD and pl_pct > 0.01:
                    sell_reason = f"Take Profit (RSI {agent.state.current_rsi:.0f})"

                if sell_reason:
                    print(f"   📉 EXECUTE SELL: {ticker} ({sell_reason})")
                    self.execute_sell(ticker, qty_held, sell_reason)
                    result_entry.update({
                        "Action": "EXECUTED_SELL",
                        "Reason": sell_reason,
                        "Qty": qty_held,
                        "TotalPos": "$0.00"
                    })
                    self.scan_results.append(result_entry)
                    return

            # Skip Logic
            if not is_owned and agent.state.current_rsi > 60:
                result_entry["Reason"] = "High RSI (Skipped)"
                self.scan_results.append(result_entry)
                return

            # --- DEEP ANALYSIS ---
            agent.fetch_fundamentals()
            agent.fetch_news()
            agent.run_ai()

            s = agent.state
            if s.fundamentals_valid:
                result_entry.update({
                    "Growth": f"{s.growth_rate:.1%}", "PE": f"{s.pe_ratio:.1f}",
                    "Margin": f"{s.gross_margin:.1%}", "Debt": f"{s.debt_equity:.2f}", "ROE": f"{s.roe:.1%}"
                })

            # Check AI Exit
            if is_owned and s.sentiment_score < self.config.MIN_AI_HOLD_SCORE:
                sell_reason = f"AI Score Drop ({s.sentiment_score})"
                print(f"   📉 EXECUTE SELL: {ticker} ({sell_reason})")
                self.execute_sell(ticker, qty_held, sell_reason)
                result_entry.update({
                    "Action": "EXECUTED_SELL", "Reason": sell_reason,
                    "Qty": qty_held, "Score": int(s.sentiment_score)
                })
                self.scan_results.append(result_entry)
                return

            # --- BUY LOGIC ---
            if not is_owned and s.decision == "BUY" and s.current_rsi < self.config.RSI_BUY_THRESHOLD:
                qty_bought = self.execute_buy(agent.state)
                if qty_bought > 0:
                    final_qty = qty_held + qty_bought
                    result_entry.update({"Action": "EXECUTED_BUY", "Qty": qty_bought})

            total_val = final_qty * curr_price
            result_entry.update({
                "TotalPos": f"${total_val:,.2f}",
                "Action": result_entry.get("Action", s.decision if not is_owned else "HOLD"),
                "Score": int(s.sentiment_score),
                "Reason": s.reasoning
            })

            agent.print_vetting_report()
            self.tracker.save_state(agent.state)

        except Exception as e:
            result_entry["Reason"] = f"Error: {str(e)[:20]}"
            print(f"   ❌ Process Error: {e}")
            traceback.print_exc()
        
        self.scan_results.append(result_entry)

    # --- EXECUTION METHODS ---
    def execute_buy(self, state: MarketState) -> float:
        try:
            self.smart_cancel(state.ticker, OrderSide.BUY)
            acct = self.alpaca.get_account()
            amt = min(float(acct.buying_power), self.config.TRADE_ALLOCATION)
            qty = int(amt // state.current_price)
            if qty > 0:
                print(f"   🚀 ORDER SENT: Buy {qty} {state.ticker}")
                self.alpaca.submit_order(MarketOrderRequest(symbol=state.ticker, qty=qty, side=OrderSide.BUY, time_in_force=TimeInForce.GTC))
                return qty
            return 0.0
        except Exception as e:
            print(f"   ❌ Buy Error: {e}")
            return 0.0

    def execute_sell(self, ticker: str, qty: float, reason: str):
        try:
            self.smart_cancel(ticker, OrderSide.SELL)
            print(f"   📉 SELL ORDER: {qty} {ticker}")
            self.alpaca.submit_order(MarketOrderRequest(symbol=ticker, qty=qty, side=OrderSide.SELL, time_in_force=TimeInForce.GTC))
        except Exception as e:
            print(f"   ❌ Sell Error: {e}")

    def smart_cancel(self, ticker: str, side_to_cancel: OrderSide):
        try:
            req = GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[ticker])
            open_orders = self.alpaca.get_orders(req)
            for o in open_orders:
                if o.side == side_to_cancel:
                    print(f"      🔓 Cancelling conflicting {o.side} order {o.id}")
                    self.alpaca.cancel_order_by_id(o.id)
            if open_orders: time.sleep(1)
        except Exception as e:
            print(f"   ⚠️ Smart Cancel Warning: {e}")

    # --- RUN LOOP ---
    def run(self):
        try:
            alpaca_pos = self.alpaca.get_all_positions()
            positions = {}
            for p in alpaca_pos:
                positions[p.symbol] = {"qty": float(p.qty), "entry": float(p.avg_entry_price)}

            targets = self.discovery.get_universe()
            all_targets = set(targets + list(positions.keys()))

            print(f"📋 Final Scan List: {len(all_targets)} stocks")

            for t in all_targets:
                pos_data = positions.get(t, {"qty": 0.0, "entry": 0.0})
                self.process(t, pos_data, force=True)

        finally:
            self.generate_console_summary()
            self.send_slack_summary()
            self.send_report_to_slack() # <--- NEW ATTACHMENT FUNCTION

    # --- REPORTING ---
    def generate_console_summary(self):
        if not self.scan_results: return
        print("\n" + "="*155)
        print("📊 MARKET SCAN SUMMARY")
        print("="*155)
        df = pd.DataFrame(self.scan_results)
        df.insert(0, '#', range(1, len(df) + 1))
        
        cols = ["#", "Ticker", "Status", "Action", "Qty", "Price", "TotalPos", "PL", "Score", "Growth", "PE", "Margin", "Reason"]
        final_cols = [c for c in cols if c in df.columns]

        print(df[final_cols].to_string(index=False))
        print("="*155 + "\n")

    def send_slack_summary(self):
        """Sends the quick summary blocks."""
        if not self.scan_results: return

        buys = [x for x in self.scan_results if "BUY" in x['Action'] and "EXECUTED" in x['Action']]
        sells = [x for x in self.scan_results if "SELL" in x['Action']]

        blocks = [
            {"type": "header", "text": {"type": "plain_text", "text": "📊 Darwinian Daily Briefing"}},
            {"type": "section", "text": {"type": "mrkdwn", "text": f"🚀 *Bought:* {len(buys)}  |  📉 *Sold:* {len(sells)}"}},
            {"type": "divider"}
        ]

        if sells:
            msg = "*📉 SELLS:*\n" + "\n".join([f"• *{s['Ticker']}* (Sold {s['Qty']}) | Profit: {s['PL']} | {s['Reason']}" for s in sells])
            blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": msg}})

        if buys:
            msg = "*🚀 BUYS:*\n" + "\n".join([f"• *{b['Ticker']}* (Buy {b['Qty']} @ ${b['Price']}) | Score: {b['Score']}" for b in buys])
            blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": msg}})

        try:
            self.slack.chat_postMessage(channel=self.config.SLACK_CHANNEL, text="Daily Scan Summary", blocks=blocks)
        except Exception as e:
            print(f"   ⚠️ Slack Summary Error: {e}")

    # --- 🆕 NEW: CSV ATTACHMENT FUNCTION ---
    def send_report_to_slack(self):
        """Generates a CSV and uploads it as a file attachment to Slack."""
        if not self.scan_results: return

        print("   📤 Uploading CSV report to Slack...")

        try:
            # 1. Create CSV in Memory
            df = pd.DataFrame(self.scan_results)

            # Select and Order Columns
            cols = ["Ticker", "Status", "Action", "Qty", "Price", "TotalPos", "PL", "Score",
                    "Growth", "PE", "Margin", "Debt", "ROE", "Reason"]
            final_cols = [c for c in cols if c in df.columns]

            csv_buffer = StringIO()
            df[final_cols].to_csv(csv_buffer, index=False)
            csv_content = csv_buffer.getvalue()

            # 2. Upload File (The Modern Way)
            timestamp = datetime.now().strftime('%Y-%m-%d')
            filename = f"Darwinian_Report_{timestamp}.csv"
            
            self.slack.files_upload_v2(
                channel=self.config.SLACK_CHANNEL,
                title=filename,
                filename=filename,
                content=csv_content,
                initial_comment="📎 *Detailed Execution Log (CSV)*"
            )
            print("   ✅ CSV uploaded successfully.")

        except SlackApiError as e:
            print(f"   ⚠️ Slack Upload Error: {e.response['error']}")
            if e.response['error'] == 'missing_scope':
                print("      👉 ACTION REQUIRED: Add 'files:write' scope to your Slack App.")
        except Exception as e:
            print(f"   ❌ Report Generation Error: {e}")

# --- 7. MAIN EXECUTION ---
# --- 7. MAIN EXECUTION ---
if __name__ == "__main__":
    try:
        setup_google_auth()

        # 1. Load Base Config (The Static Settings)
        cfg = Config.load()

        # TEST MODE OVERRIDE
        if "--test" in sys.argv:
            print("🧪 TEST MODE ENABLED: Checking 1 stock only with 20s timeout.")
            # Override config for testing by using object.__setattr__ to bypass frozen state
            object.__setattr__(cfg, 'TOP_N_STOCKS', 1)
            # We don't need EXECUTION_TIMEOUT_SECONDS in this new code structure,
            # but we ensure only 1 stock is picked.

        # 2. Run Macro Analysis (The Dynamic Adjustment)
        # This will modify 'cfg' in-place based on SPY/VIX
        # Skip macro analysis in test mode if data fetching is an issue, or let it run to verify it.
        # We'll let it run but catch errors if test mode.
        try:
            macro = MacroAnalyst(cfg)
            regime = macro.analyze_and_adapt()
        except Exception as e:
            if "--test" in sys.argv:
                print(f"⚠️ Macro Analysis Failed in Test Mode (Expected if keys invalid): {e}")
                regime = "TEST"
            else:
                raise e

        # 3. Start the Swarm with the Optimized Config
        print(f"\n✅ Configuration Optimized. Starting Swarm in {regime} Mode...")
        PortfolioManager(cfg).run()

    except KeyboardInterrupt:
        print("\n🛑 Stopped by User.")
    except Exception:
        print("\n❌ FATAL ERROR:")
        traceback.print_exc()
