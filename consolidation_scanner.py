"""
consolidation_scanner.py
=========================
Bullish Consolidation / Base-Formation Scanner — SINGLE FILE VERSION.

Scans a stock universe on three INDEPENDENT timeframes (Daily, Hourly,
15-Minute) for bullish consolidation setups that could precede a breakout,
and writes a dark trading-dashboard HTML report + CSVs + a log file.

This is a technical screener only — not financial advice.

--------------------------------------------------------------------------
INSTALL
    pip install yfinance pandas numpy jinja2

RUN
    python consolidation_scanner.py                    # daily + hourly + 15m
    python consolidation_scanner.py --timeframe daily   # only daily
    python consolidation_scanner.py --timeframe hourly  # only hourly
    python consolidation_scanner.py --timeframe 15m     # only 15-minute

OUTPUT (written to ./output/)
    stock_consolidation_report.html   <- open this in a browser, no server needed
    (CSV files and scanner.log are no longer written; progress/log messages
     print to the console only.)

EDIT YOUR STOCK LIST
    Scroll down to the "STOCK UNIVERSE" section near the top of this file.
    STOCKS is auto-built from SECTOR_MAP (individual constituents only —
    broad-market/sector index ETFs like SPY/QQQ/DIA and the sector SPDRs
    are excluded). To change the universe, edit SECTOR_MAP — nothing else
    in the file needs to change.
--------------------------------------------------------------------------
"""

import argparse
import csv
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from jinja2 import Template

logger = logging.getLogger("scanner")


# ===========================================================================
# 1. STOCK UNIVERSE  <-- EDIT THIS SECTION TO CHANGE WHAT GETS SCANNED
# ===========================================================================

# Sector map for the individual constituent stocks (used for grouping in the
# report / dashboard). Keys are tickers, values are the sector SPDR ETF.
SECTOR_MAP = {
    **{s: "XLK" for s in [
        # Technology (16)
        "NVDA", "MSFT", "AAPL", "AVGO", "AMD", "ORCL", "ADBE", "PANW",
        "NOW", "SNPS", "CRM", "CSCO", "INTC", "QCOM", "AMAT", "LRCX",
        # Extras: SMCI
        "SMCI",
    ]},
    **{s: "XLC" for s in [
        # Communication Services (12)
        "GOOGL", "GOOG", "META", "NFLX", "CMCSA", "DIS",
        "TMUS", "VZ", "T", "CHTR", "SPOT", "RBLX",
    ]},
    **{s: "XLY" for s in [
        # Consumer Discretionary (13 — COST moved to Staples)
        "AMZN", "TSLA", "HD", "MCD", "TJX", "BKNG",
        "LOW", "SBUX", "NKE", "MAR", "ROST", "EBAY", "LULU",
    ]},
    **{s: "XLP" for s in [
        # Consumer Staples (10 — COST kept here as primary)
        "WMT", "PG", "KO", "PEP", "COST", "PM", "MO", "MDLZ", "CL", "MNST",
    ]},
    **{s: "XLV" for s in [
        # Health Care (16)
        "LLY", "UNH", "JNJ", "MRK", "ABBV", "TMO", "AMGN", "BMY",
        "GILD", "ISRG", "VRTX", "CVS", "CI", "MDT", "SYK", "REGN",
    ]},
    **{s: "XLF" for s in [
        # Financials (16)
        "JPM", "BAC", "MS", "GS", "V", "MA", "AXP", "BLK",
        "SPGI", "C", "WFC", "SCHW", "COF", "PGR", "CB", "MRSH",
        # Extras: HOOD, SOFI
        "HOOD", "SOFI",
    ]},
    **{s: "XLI" for s in [
        # Industrials (15)
        "GE", "CAT", "UNP", "HON", "LMT", "UPS", "RTX", "DE",
        "FDX", "BA", "GEV", "ETN", "ADP", "FAST", "CTAS",
    ]},
    **{s: "XLE" for s in [
        # Energy (12 — NEE, SO, DUK, CEG, VST kept here as listed)
        "XOM", "CVX", "COP", "NEE", "SO", "DUK", "CEG", "VST",
        "SLB", "EOG", "KMI", "PSX",
    ]},
    **{s: "XLB" for s in [
        # Materials (8)
        "LIN", "FCX", "SHW", "NEM", "APD", "ECL", "NUE", "DOW",
    ]},
    **{s: "XLRE" for s in [
        # Real Estate (10)
        "PLD", "AMT", "EQIX", "DLR", "WELL", "SPG", "PSA", "O", "CBRE", "VTR",
    ]},
    **{s: "XLU" for s in [
        # Utilities (7 — SO, DUK, NEE already assigned to XLE above)
        "EXC", "XEL", "AEP", "SRE", "D", "PEG", "WEC",
    ]},
}

# STOCKS = every constituent from SECTOR_MAP, de-duplicated while preserving
# order. (Broad-market/sector-ETF index tickers such as SPY/QQQ/DIA and the
# sector SPDRs are intentionally excluded from the scan universe.)
STOCKS = list(dict.fromkeys(list(SECTOR_MAP.keys())))

# Readable label for each sector-SPDR "index" ticker used in SECTOR_MAP, shown
# in the report's "Index" column (next to Company) so it's clear at a glance
# which sector index a stock belongs to.
SECTOR_INDEX_LABELS = {
    "XLK": "Technology (XLK)",
    "XLC": "Communication Svcs (XLC)",
    "XLY": "Consumer Discretionary (XLY)",
    "XLP": "Consumer Staples (XLP)",
    "XLV": "Health Care (XLV)",
    "XLF": "Financials (XLF)",
    "XLI": "Industrials (XLI)",
    "XLE": "Energy (XLE)",
    "XLB": "Materials (XLB)",
    "XLRE": "Real Estate (XLRE)",
    "XLU": "Utilities (XLU)",
}

# Emoji shown next to each sector header in the Telegram card.
SECTOR_EMOJI = {
    "XLK": "\U0001F4BB",   # 💻 Technology
    "XLC": "\U0001F4E1",   # 📡 Communication Services
    "XLY": "\U0001F6CD",   # 🛍️ Consumer Discretionary
    "XLP": "\U0001F96B",   # 🥫 Consumer Staples
    "XLV": "\U0001F489",   # 💉 Health Care
    "XLF": "\U0001F3E6",   # 🏦 Financials
    "XLI": "\U0001F3ED",   # 🏭 Industrials
    "XLE": "\u26A1",       # ⚡ Energy
    "XLB": "\U0001F3ED",   # 🏭 Materials
    "XLRE": "\U0001F3E2",  # 🏢 Real Estate
    "XLU": "\u26A1",       # ⚡ Utilities
}

# Icon + short label shown per consolidation classification in the Telegram
# card. Only classes that can reach the Telegram min-score (>=60) matter.
CLASSIFICATION_DISPLAY = {
    "STRONG CONSOLIDATION": ("\U0001F535", "STRONG"),  # 🔵 STRONG
    "GOOD CONSOLIDATION": ("\U0001F7E2", "GOOD"),       # 🟢 GOOD
    "WATCH": ("\U0001F7E1", "WATCH"),                   # 🟡 WATCH
}

# Optional: readable company names for the report. Any symbol missing here
# just falls back to showing the raw ticker — nothing breaks.
COMPANY_NAMES = {
    # --- Technology ---
    "NVDA": "NVIDIA Corporation", "MSFT": "Microsoft Corporation", "AAPL": "Apple Inc.",
    "AVGO": "Broadcom Inc.", "AMD": "Advanced Micro Devices", "ORCL": "Oracle Corporation",
    "ADBE": "Adobe Inc.", "PANW": "Palo Alto Networks", "NOW": "ServiceNow Inc.",
    "SNPS": "Synopsys Inc.", "CRM": "Salesforce Inc.", "CSCO": "Cisco Systems",
    "INTC": "Intel Corporation", "QCOM": "Qualcomm Inc.", "AMAT": "Applied Materials",
    "LRCX": "Lam Research", "SMCI": "Super Micro Computer",
    # --- Communication Services ---
    "GOOGL": "Alphabet Inc. (Class A)", "GOOG": "Alphabet Inc. (Class C)",
    "META": "Meta Platforms", "NFLX": "Netflix Inc.", "CMCSA": "Comcast Corporation",
    "DIS": "The Walt Disney Company", "TMUS": "T-Mobile US", "VZ": "Verizon Communications",
    "T": "AT&T Inc.", "CHTR": "Charter Communications", "SPOT": "Spotify Technology",
    "RBLX": "Roblox Corporation",
    # --- Consumer Discretionary ---
    "AMZN": "Amazon.com Inc.", "TSLA": "Tesla Inc.", "HD": "The Home Depot",
    "MCD": "McDonald's Corporation", "TJX": "TJX Companies", "BKNG": "Booking Holdings",
    "LOW": "Lowe's Companies", "SBUX": "Starbucks Corporation", "NKE": "Nike Inc.",
    "MAR": "Marriott International", "ROST": "Ross Stores", "EBAY": "eBay Inc.",
    "LULU": "Lululemon Athletica",
    # --- Consumer Staples ---
    "WMT": "Walmart Inc.", "PG": "Procter & Gamble", "KO": "Coca-Cola Company",
    "PEP": "PepsiCo Inc.", "COST": "Costco Wholesale", "PM": "Philip Morris International",
    "MO": "Altria Group", "MDLZ": "Mondelez International", "CL": "Colgate-Palmolive",
    "MNST": "Monster Beverage",
    # --- Health Care ---
    "LLY": "Eli Lilly and Company", "UNH": "UnitedHealth Group", "JNJ": "Johnson & Johnson",
    "MRK": "Merck & Co.", "ABBV": "AbbVie Inc.", "TMO": "Thermo Fisher Scientific",
    "AMGN": "Amgen Inc.", "BMY": "Bristol-Myers Squibb", "GILD": "Gilead Sciences",
    "ISRG": "Intuitive Surgical", "VRTX": "Vertex Pharmaceuticals", "CVS": "CVS Health",
    "CI": "The Cigna Group", "MDT": "Medtronic plc", "SYK": "Stryker Corporation",
    "REGN": "Regeneron Pharmaceuticals",
    # --- Financials ---
    "JPM": "JPMorgan Chase", "BAC": "Bank of America", "MS": "Morgan Stanley",
    "GS": "Goldman Sachs Group", "V": "Visa Inc.", "MA": "Mastercard Inc.",
    "AXP": "American Express", "BLK": "BlackRock Inc.", "SPGI": "S&P Global",
    "C": "Citigroup Inc.", "WFC": "Wells Fargo & Co.", "SCHW": "Charles Schwab Corp.",
    "COF": "Capital One Financial", "PGR": "Progressive Corporation", "CB": "Chubb Limited",
    "MRSH": "Marsh McLennan", "HOOD": "Robinhood Markets", "SOFI": "SoFi Technologies",
    # --- Industrials ---
    "GE": "GE Aerospace", "CAT": "Caterpillar Inc.", "UNP": "Union Pacific Corp.",
    "HON": "Honeywell International", "LMT": "Lockheed Martin", "UPS": "United Parcel Service",
    "RTX": "RTX Corporation", "DE": "Deere & Company", "FDX": "FedEx Corporation",
    "BA": "Boeing Company", "GEV": "GE Vernova", "ETN": "Eaton Corporation",
    "ADP": "Automatic Data Processing", "FAST": "Fastenal Company", "CTAS": "Cintas Corporation",
    # --- Energy / Utilities-in-Energy ---
    "XOM": "Exxon Mobil Corp.", "CVX": "Chevron Corporation", "COP": "ConocoPhillips",
    "NEE": "NextEra Energy", "SO": "The Southern Company", "DUK": "Duke Energy",
    "CEG": "Constellation Energy", "VST": "Vistra Corp.", "SLB": "SLB (Schlumberger)",
    "EOG": "EOG Resources", "KMI": "Kinder Morgan", "PSX": "Phillips 66",
    # --- Materials ---
    "LIN": "Linde plc", "FCX": "Freeport-McMoRan", "SHW": "Sherwin-Williams",
    "NEM": "Newmont Corporation", "APD": "Air Products and Chemicals", "ECL": "Ecolab Inc.",
    "NUE": "Nucor Corporation", "DOW": "Dow Inc.",
    # --- Real Estate ---
    "PLD": "Prologis Inc.", "AMT": "American Tower Corp.", "EQIX": "Equinix Inc.",
    "DLR": "Digital Realty Trust", "WELL": "Welltower Inc.", "SPG": "Simon Property Group",
    "PSA": "Public Storage", "O": "Realty Income Corp.", "CBRE": "CBRE Group",
    "VTR": "Ventas Inc.",
    # --- Utilities ---
    "EXC": "Exelon Corporation", "XEL": "Xcel Energy", "AEP": "American Electric Power",
    "SRE": "Sempra", "D": "Dominion Energy", "PEG": "Public Service Enterprise Group",
    "WEC": "WEC Energy Group",
}


# ===========================================================================
# 2. CONFIGURATION — every threshold/weight lives here
# ===========================================================================

@dataclass
class TimeframeConfig:
    label: str
    interval: str
    period: str
    consolidation_candles: int
    pre_move_candles: int
    min_score: int
    max_range_percent: float
    atr_period: int = 14
    rsi_period: int = 14
    adx_period: int = 14
    volume_period: int = 20
    ema_fast: int = 20
    ema_mid: int = 50
    ema_slow: int = 200
    breakout_volume_multiplier: float = 1.5
    min_pre_move_percent: float = 3.0
    rsi_low: float = 45.0
    rsi_high: float = 65.0


# Yahoo Finance intraday history limits (not bypassable):
#   15m candles -> ~last 60 days only | 1h candles -> ~last 730 days only
# "period" below respects those limits with headroom.

DAILY_CONFIG = TimeframeConfig(
    label="Daily", interval="1d", period="1y",
    consolidation_candles=20, pre_move_candles=20,
    min_score=60, max_range_percent=8.0,
)

HOURLY_CONFIG = TimeframeConfig(
    label="Hourly", interval="1h", period="60d",
    consolidation_candles=20, pre_move_candles=20,
    min_score=60, max_range_percent=6.0,
)

MIN15_CONFIG = TimeframeConfig(
    label="15 Minute", interval="15m", period="30d",
    consolidation_candles=20, pre_move_candles=20,
    min_score=60, max_range_percent=5.0,
)

SCORE_WEIGHTS = {
    "ema20_above_ema50": 15,
    "price_above_ema20": 10,
    "prior_upswing": 15,
    "tight_range": 15,
    "atr_contraction": 10,
    "bb_contraction": 10,
    "volume_contraction": 10,
    "rsi_sweet_spot": 5,
    "breakout_proximity": 10,
}
assert sum(SCORE_WEIGHTS.values()) == 100, "SCORE_WEIGHTS must sum to 100"

SCORE_BANDS = [
    (80, 100, "STRONG CONSOLIDATION"),
    (70, 79, "GOOD CONSOLIDATION"),
    (60, 69, "WATCH"),
    (50, 59, "WEAK"),
    (0, 49, "IGNORE"),
]

NEAR_BREAKOUT_MAX_DISTANCE_PCT = 2.0

MAX_WORKERS = 12  # shared pool now handles daily+hourly+15m together (see run_all_scans),
                  # so this covers up to len(STOCKS) * 3 concurrent download tasks.
                  # yfinance/Yahoo can rate-limit or throttle at very high concurrency -
                  # if you see a spike in "ERROR" / retry-exhausted rows in scanner.log,
                  # dial this back down (e.g. 6-8).
REQUEST_RETRIES = 3
RETRY_BACKOFF_SECONDS = 2.0
INTER_REQUEST_PAUSE_SECONDS = 0.2

# Anchor the output folder to THIS script's own location (not the current
# working directory the user happens to run `python` from), so
# `output/` always lands right next to consolidation_scanner.py no matter
# where you invoke it from.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
HTML_REPORT_FILENAME = "stock_consolidation_report.html"

# Market label shown in the Telegram summary header (the config file below
# is shared with the India script, so each script tags its own label).
MARKET_LABEL = "USA"

# --- Telegram config --------------------------------------------------------
# Preferred (CI / GitHub Actions): environment variables
#     TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, TELEGRAM_ENABLED, TELEGRAM_TOP_N_PER_INDEX
# Fallback (local/dev only): a telegram_config.py file placed next to this
# script (shared with consolidation_scanner_india.py). Environment variables
# always take priority over telegram_config.py when both are present.
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
try:
    import telegram_config as _tg_cfg
except ImportError:
    _tg_cfg = None


def _env_or_attr(env_name: str, attr_name: str, default):
    val = os.environ.get(env_name)
    if val is not None and val != "":
        return val
    if _tg_cfg is not None:
        return getattr(_tg_cfg, attr_name, default)
    return default


def _env_bool(env_name: str, attr_name: str, default: bool) -> bool:
    val = os.environ.get(env_name)
    if val is not None and val != "":
        return val.strip().lower() in ("1", "true", "yes", "on")
    if _tg_cfg is not None:
        return getattr(_tg_cfg, attr_name, default)
    return default


def _env_int(env_name: str, attr_name: str, default: int) -> int:
    val = os.environ.get(env_name)
    if val is not None and val != "":
        try:
            return int(val)
        except ValueError:
            pass
    if _tg_cfg is not None:
        return getattr(_tg_cfg, attr_name, default)
    return default


TELEGRAM_BOT_TOKEN = _env_or_attr("TELEGRAM_BOT_TOKEN", "TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = _env_or_attr("TELEGRAM_CHAT_ID", "TELEGRAM_CHAT_ID", "")
# If not explicitly set, default to "on" whenever both a token and chat id
# are available (env vars or telegram_config.py), off otherwise.
TELEGRAM_ENABLED = _env_bool(
    "TELEGRAM_ENABLED", "TELEGRAM_ENABLED", bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID)
)
TELEGRAM_TOP_N_PER_INDEX = _env_int("TELEGRAM_TOP_N_PER_INDEX", "TELEGRAM_TOP_N_PER_INDEX", 3)
LOG_FILENAME = "scanner.log"
CSV_FILENAMES = {
    "daily": "daily_consolidation.csv",
    "hourly": "hourly_consolidation.csv",
    "15m": "15m_consolidation.csv",
}


# ===========================================================================
# 3. INDICATORS
# ===========================================================================

def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period, min_periods=period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def atr_percent(df: pd.DataFrame, period: int = 14) -> pd.Series:
    return atr(df, period) / df["Close"] * 100


def bollinger_bands(series: pd.Series, period: int = 20, num_std: float = 2.0):
    mid = sma(series, period)
    std = series.rolling(window=period, min_periods=period).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    width_pct = (upper - lower) / mid * 100
    return upper, mid, lower, width_pct


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    result = 100 - (100 / (1 + rs))
    result = result.where(avg_loss != 0, 100)
    return result


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low = df["High"], df["Low"]
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = pd.Series(plus_dm, index=df.index)
    minus_dm = pd.Series(minus_dm, index=df.index)
    tr_atr = atr(df, period)
    plus_di = 100 * (plus_dm.ewm(alpha=1 / period, adjust=False, min_periods=period).mean() / tr_atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1 / period, adjust=False, min_periods=period).mean() / tr_atr)
    dx = (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan) * 100
    return dx.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def relative_volume(df: pd.DataFrame, period: int = 20) -> pd.Series:
    vol_sma = sma(df["Volume"], period)
    return df["Volume"] / vol_sma.replace(0, np.nan)


def add_all_indicators(df: pd.DataFrame, tf: TimeframeConfig) -> pd.DataFrame:
    out = df.copy()
    out["EMA_FAST"] = ema(out["Close"], tf.ema_fast)
    out["EMA_MID"] = ema(out["Close"], tf.ema_mid)
    out["EMA_SLOW"] = ema(out["Close"], tf.ema_slow) if len(out) >= tf.ema_slow else np.nan
    out["ATR"] = atr(out, tf.atr_period)
    out["ATR_PCT"] = atr_percent(out, tf.atr_period)
    _, out["BB_MID"], _, out["BB_WIDTH_PCT"] = bollinger_bands(out["Close"], period=20)
    out["RSI"] = rsi(out["Close"], tf.rsi_period)
    out["ADX"] = adx(out, tf.adx_period)
    out["VOL_SMA"] = sma(out["Volume"], tf.volume_period)
    out["REL_VOL"] = relative_volume(out, tf.volume_period)
    return out


# ===========================================================================
# 4. DATA FETCHING (retries, in-memory cache, never raises)
# ===========================================================================

_CACHE: dict = {}


def fetch_data(symbol: str, tf: TimeframeConfig) -> Optional[pd.DataFrame]:
    cache_key = (symbol, tf.interval, tf.period)
    if cache_key in _CACHE:
        return _CACHE[cache_key]

    last_error = None
    for attempt in range(1, REQUEST_RETRIES + 1):
        try:
            df = yf.Ticker(symbol).history(period=tf.period, interval=tf.interval, auto_adjust=True)
            time.sleep(INTER_REQUEST_PAUSE_SECONDS)

            if df is None or df.empty:
                raise ValueError("empty dataframe returned")

            df = df.dropna(how="all")
            required_cols = {"Open", "High", "Low", "Close", "Volume"}
            if not required_cols.issubset(df.columns):
                raise ValueError(f"missing expected columns, got {list(df.columns)}")

            # Yahoo/yfinance sometimes appends a same-session bar before that
            # session's OHLC data has fully settled (very common in the
            # minutes right after market close, or under throttling) — its
            # Close (and often Open/High/Low) come back NaN even though the
            # row itself isn't dropped by dropna(how="all") above. Trim any
            # such incomplete rows off the TAIL so the scanner falls back to
            # the last fully-formed candle instead of failing the whole
            # symbol with a misleading "NaN/short data" error.
            trimmed = 0
            while len(df) and pd.isna(df["Close"].iloc[-1]):
                df = df.iloc[:-1]
                trimmed += 1
            if trimmed:
                logger.debug("Trimmed %d incomplete trailing bar(s) for %s [%s/%s]",
                            trimmed, symbol, tf.interval, tf.period)
            if df.empty:
                raise ValueError("no rows with a valid Close after trimming incomplete tail")

            _CACHE[cache_key] = df
            return df

        except Exception as e:  # noqa: BLE001 - isolate failures per stock, never crash the run
            last_error = e
            logger.warning("Attempt %d/%d failed for %s [%s/%s]: %s",
                            attempt, REQUEST_RETRIES, symbol, tf.interval, tf.period, e)
            if attempt < REQUEST_RETRIES:
                time.sleep(RETRY_BACKOFF_SECONDS * attempt)

    logger.error("Giving up on %s [%s/%s] after %d attempts: %s",
                 symbol, tf.interval, tf.period, REQUEST_RETRIES, last_error)
    return None


# ===========================================================================
# 5. CONSOLIDATION METRICS
# ===========================================================================

@dataclass
class ConsolidationMetrics:
    current_price: float
    range_high: float
    range_low: float
    range_percent: float
    breakout_distance_percent: float
    ema_fast: Optional[float]
    ema_mid: Optional[float]
    ema_slow: Optional[float]
    rsi: Optional[float]
    adx_now: Optional[float]
    adx_prior: Optional[float]
    atr_pct_now: Optional[float]
    atr_pct_prior: Optional[float]
    bb_width_now: Optional[float]
    bb_width_prior: Optional[float]
    rel_volume: Optional[float]
    consolidation_volume_avg: float
    pre_move_volume_avg: float
    pre_consolidation_return_pct: float
    swing_high: float
    swing_low: float
    support: float
    resistance: float
    prev_close: float
    prev_volume: float
    volume_sma: float


def compute_consolidation_metrics(df: pd.DataFrame, tf: TimeframeConfig) -> Optional[ConsolidationMetrics]:
    needed_rows = tf.consolidation_candles + tf.pre_move_candles + 6
    if len(df) < needed_rows:
        return None

    # The base/resistance box is built from the N candles BEFORE the current
    # one — the current candle is the potential breakout bar and must be
    # excluded, otherwise its own high would always define (and swallow)
    # the resistance level, making a true breakout impossible to detect.
    window = df.iloc[-(tf.consolidation_candles + 1):-1]
    last_row = df.iloc[-1]
    if pd.isna(last_row["Close"]):
        return None

    current_price = float(last_row["Close"])
    range_high = float(window["High"].max())
    range_low = float(window["Low"].min())
    if range_low <= 0:
        return None

    range_percent = (range_high - range_low) / range_low * 100
    breakout_distance_percent = max(0.0, (range_high - current_price) / current_price * 100)

    pre_move_slice = df.iloc[-(tf.consolidation_candles + tf.pre_move_candles + 1):-(tf.consolidation_candles + 1)]
    if pre_move_slice.empty:
        pre_consolidation_return_pct = 0.0
        pre_move_volume_avg = float(window["Volume"].mean())
    else:
        price_before = float(pre_move_slice["Close"].iloc[0])
        price_at_consolidation_start = float(window["Close"].iloc[0])
        pre_consolidation_return_pct = (
            (price_at_consolidation_start - price_before) / price_before * 100
            if price_before > 0 else 0.0
        )
        pre_move_volume_avg = float(pre_move_slice["Volume"].mean())

    consolidation_volume_avg = float(window["Volume"].mean())

    half = max(1, tf.consolidation_candles // 2)
    first_half, second_half = window.iloc[:half], window.iloc[half:]

    def _safe_mean(series):
        v = series.dropna()
        return float(v.mean()) if not v.empty else None

    atr_pct_prior = _safe_mean(first_half["ATR_PCT"])
    atr_pct_now = _safe_mean(second_half["ATR_PCT"])
    adx_prior = _safe_mean(first_half["ADX"])
    adx_now = _safe_mean(second_half["ADX"])
    bb_width_prior = _safe_mean(first_half["BB_WIDTH_PCT"])
    bb_width_now = _safe_mean(second_half["BB_WIDTH_PCT"])

    swing_high_val = float(df["High"].tail(max(10, tf.consolidation_candles)).max())
    swing_low_val = float(df["Low"].tail(max(10, tf.consolidation_candles)).min())
    prev_close = float(df["Close"].iloc[-2]) if len(df) >= 2 else current_price
    prev_volume = float(df["Volume"].iloc[-2]) if len(df) >= 2 else float(last_row["Volume"])

    def _val(col):
        v = last_row.get(col)
        return float(v) if pd.notna(v) else None

    return ConsolidationMetrics(
        current_price=current_price, range_high=range_high, range_low=range_low,
        range_percent=range_percent, breakout_distance_percent=breakout_distance_percent,
        ema_fast=_val("EMA_FAST"), ema_mid=_val("EMA_MID"), ema_slow=_val("EMA_SLOW"),
        rsi=_val("RSI"), adx_now=adx_now, adx_prior=adx_prior,
        atr_pct_now=atr_pct_now, atr_pct_prior=atr_pct_prior,
        bb_width_now=bb_width_now, bb_width_prior=bb_width_prior,
        rel_volume=_val("REL_VOL"),
        consolidation_volume_avg=consolidation_volume_avg, pre_move_volume_avg=pre_move_volume_avg,
        pre_consolidation_return_pct=pre_consolidation_return_pct,
        swing_high=swing_high_val, swing_low=swing_low_val,
        support=range_low, resistance=range_high,
        prev_close=prev_close, prev_volume=prev_volume,
        volume_sma=_val("VOL_SMA") or consolidation_volume_avg,
    )


def classify_breakout_status(m: ConsolidationMetrics, tf: TimeframeConfig) -> str:
    broke_out = (
        m.current_price > m.range_high
        and m.rel_volume is not None and m.rel_volume > tf.breakout_volume_multiplier
        and m.current_price > m.prev_close
    )
    if broke_out:
        return "BREAKOUT"
    if m.breakout_distance_percent <= NEAR_BREAKOUT_MAX_DISTANCE_PCT:
        return "NEAR BREAKOUT"
    if m.range_percent <= tf.max_range_percent:
        return "WATCH"
    return "WEAK CONSOLIDATION"


# ===========================================================================
# 6. SCORING (additive — no single indicator can veto a stock)
# ===========================================================================

@dataclass
class ScoreResult:
    score: int
    classification: str
    trend_score: int
    consolidation_score: int
    volume_score: int
    momentum_score: int
    breakout_score: int
    reasons: list = field(default_factory=list)


def _classify(score: int) -> str:
    for low, high, label in SCORE_BANDS:
        if low <= score <= high:
            return label
    return "IGNORE"


def compute_score(m: ConsolidationMetrics, tf: TimeframeConfig) -> ScoreResult:
    reasons = []
    trend_score = consolidation_score = volume_score = momentum_score = breakout_score = 0

    if m.ema_fast is not None and m.ema_mid is not None and m.ema_fast > m.ema_mid:
        trend_score += SCORE_WEIGHTS["ema20_above_ema50"]
        reasons.append(f"EMA{tf.ema_fast} above EMA{tf.ema_mid}")

    if m.ema_fast is not None and m.current_price > m.ema_fast:
        trend_score += SCORE_WEIGHTS["price_above_ema20"]
        reasons.append(f"Price above EMA{tf.ema_fast}")

    if m.pre_consolidation_return_pct >= tf.min_pre_move_percent:
        consolidation_score += SCORE_WEIGHTS["prior_upswing"]
        reasons.append(f"Previous {m.pre_consolidation_return_pct:.1f}% upswing before consolidating")

    if m.range_percent <= tf.max_range_percent:
        tightness = max(0.0, 1 - (m.range_percent / tf.max_range_percent))
        pts = round(SCORE_WEIGHTS["tight_range"] * (0.5 + 0.5 * tightness))
        consolidation_score += pts
        reasons.append(f"Tight {m.range_percent:.1f}% consolidation range")

    if m.atr_pct_now is not None and m.atr_pct_prior is not None and m.atr_pct_now < m.atr_pct_prior:
        consolidation_score += SCORE_WEIGHTS["atr_contraction"]
        reasons.append("ATR% contracting during consolidation")

    if m.bb_width_now is not None and m.bb_width_prior is not None and m.bb_width_now < m.bb_width_prior:
        consolidation_score += SCORE_WEIGHTS["bb_contraction"]
        reasons.append("Bollinger Band width narrowing")

    if m.pre_move_volume_avg > 0 and m.consolidation_volume_avg < m.pre_move_volume_avg:
        volume_score += SCORE_WEIGHTS["volume_contraction"]
        ratio = m.consolidation_volume_avg / m.pre_move_volume_avg
        reasons.append(f"Volume declining during consolidation ({ratio:.2f}x prior)")

    if m.rsi is not None and tf.rsi_low <= m.rsi <= tf.rsi_high:
        momentum_score += SCORE_WEIGHTS["rsi_sweet_spot"]
        reasons.append(f"RSI {m.rsi:.0f} in bullish-neutral zone")

    if m.adx_now is not None and m.adx_prior is not None and m.adx_now < m.adx_prior:
        reasons.append("ADX declining (losing directional strength -> consolidating)")

    d = m.breakout_distance_percent
    if d <= 1:
        breakout_score += SCORE_WEIGHTS["breakout_proximity"]
        reasons.append(f"Only {d:.1f}% below resistance (excellent)")
    elif d <= 2:
        breakout_score += round(SCORE_WEIGHTS["breakout_proximity"] * 0.8)
        reasons.append(f"Only {d:.1f}% below resistance (very good)")
    elif d <= 3:
        breakout_score += round(SCORE_WEIGHTS["breakout_proximity"] * 0.6)
        reasons.append(f"{d:.1f}% below resistance (good)")
    elif d <= 5:
        breakout_score += round(SCORE_WEIGHTS["breakout_proximity"] * 0.35)
        reasons.append(f"{d:.1f}% below resistance (moderate)")
    else:
        reasons.append(f"{d:.1f}% below resistance (far from breakout)")

    total = max(0, min(100, trend_score + consolidation_score + volume_score + momentum_score + breakout_score))

    return ScoreResult(
        score=total, classification=_classify(total),
        trend_score=trend_score, consolidation_score=consolidation_score,
        volume_score=volume_score, momentum_score=momentum_score, breakout_score=breakout_score,
        reasons=reasons,
    )


# ===========================================================================
# 7. PER-STOCK PIPELINE  (fetch -> indicators -> metrics -> score)
# ===========================================================================

@dataclass
class StockReport:
    symbol: str
    company: str
    sector: str
    timeframe: str
    status: str
    error_message: Optional[str] = None
    score: Optional[int] = None
    classification: Optional[str] = None
    breakout_status: Optional[str] = None
    price: Optional[float] = None
    range_percent: Optional[float] = None
    breakout_distance_percent: Optional[float] = None
    rsi: Optional[float] = None
    adx: Optional[float] = None
    atr_pct: Optional[float] = None
    rel_volume: Optional[float] = None
    ema_fast: Optional[float] = None
    ema_mid: Optional[float] = None
    ema_slow: Optional[float] = None
    support: Optional[float] = None
    resistance: Optional[float] = None
    pre_upswing_percent: Optional[float] = None
    trend_score: int = 0
    consolidation_score: int = 0
    volume_score: int = 0
    momentum_score: int = 0
    breakout_score: int = 0
    reasons: list = field(default_factory=list)


def _error_report(symbol: str, timeframe: str, message: str) -> StockReport:
    logger.warning("ERROR analyzing %s [%s]: %s", symbol, timeframe, message)
    return StockReport(symbol=symbol, company=COMPANY_NAMES.get(symbol, symbol),
                        sector=SECTOR_INDEX_LABELS.get(SECTOR_MAP.get(symbol, ""), SECTOR_MAP.get(symbol, "")),
                        timeframe=timeframe, status="ERROR", error_message=message)


def analyze_stock(symbol: str, tf: TimeframeConfig, timeframe_key: str) -> StockReport:
    """Full pipeline for one stock on one timeframe. Never raises."""
    try:
        df = fetch_data(symbol, tf)
        if df is None:
            return _error_report(symbol, timeframe_key, "no data returned (download failed)")

        min_needed = tf.consolidation_candles + tf.pre_move_candles + max(tf.ema_slow, 20) // 5
        if len(df) < min_needed:
            return _error_report(symbol, timeframe_key,
                                  f"insufficient history ({len(df)} candles, need ~{min_needed})")

        enriched = add_all_indicators(df, tf)
        metrics = compute_consolidation_metrics(enriched, tf)
        if metrics is None:
            return _error_report(symbol, timeframe_key, "could not compute consolidation metrics (NaN/short data)")

        score_result = compute_score(metrics, tf)
        breakout_status = classify_breakout_status(metrics, tf)

        return StockReport(
            symbol=symbol, company=COMPANY_NAMES.get(symbol, symbol),
            sector=SECTOR_INDEX_LABELS.get(SECTOR_MAP.get(symbol, ""), SECTOR_MAP.get(symbol, "")),
            timeframe=timeframe_key, status="OK",
            score=score_result.score, classification=score_result.classification, breakout_status=breakout_status,
            price=metrics.current_price, range_percent=metrics.range_percent,
            breakout_distance_percent=metrics.breakout_distance_percent,
            rsi=metrics.rsi, adx=metrics.adx_now, atr_pct=metrics.atr_pct_now, rel_volume=metrics.rel_volume,
            ema_fast=metrics.ema_fast, ema_mid=metrics.ema_mid, ema_slow=metrics.ema_slow,
            support=metrics.support, resistance=metrics.resistance,
            pre_upswing_percent=metrics.pre_consolidation_return_pct,
            trend_score=score_result.trend_score, consolidation_score=score_result.consolidation_score,
            volume_score=score_result.volume_score, momentum_score=score_result.momentum_score,
            breakout_score=score_result.breakout_score, reasons=score_result.reasons,
        )
    except Exception as e:  # noqa: BLE001 - last line of defense, must never crash the run
        return _error_report(symbol, timeframe_key, f"unexpected error: {e}")


# ===========================================================================
# 8. THREE INDEPENDENT SCANNERS
#    Each uses ITS OWN TimeframeConfig — a stock's daily result has no path
#    to affect its hourly or 15-minute result. They share the calculation
#    pipeline above (DRY) but their pass/fail outcomes are fully decoupled.
# ===========================================================================

def scan_daily(symbol: str) -> StockReport:
    return analyze_stock(symbol, DAILY_CONFIG, "daily")


def scan_hourly(symbol: str) -> StockReport:
    return analyze_stock(symbol, HOURLY_CONFIG, "hourly")


def scan_15min(symbol: str) -> StockReport:
    return analyze_stock(symbol, MIN15_CONFIG, "15m")


def _run_scan(scan_fn, timeframe_label: str, min_score: int, stocks: list) -> list:
    start = time.time()
    logger.info("Starting %s scanner...", timeframe_label)
    logger.info("Stocks: %d", len(stocks))

    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(scan_fn, s): s for s in stocks}
        for future in as_completed(futures):
            results.append(future.result())

    ok = [r for r in results if r.status == "OK"]
    errors = [r for r in results if r.status == "ERROR"]
    qualified = [r for r in ok if r.score is not None and r.score >= min_score]

    logger.info("%s successful downloads: %d, failed: %d", timeframe_label, len(ok), len(errors))
    logger.info("%s qualified (score >= %d): %d", timeframe_label, min_score, len(qualified))
    logger.info("%s scan completed in %.1fs", timeframe_label, time.time() - start)
    return results


def run_daily_scan(stocks=None):
    return _run_scan(scan_daily, "DAILY", DAILY_CONFIG.min_score, stocks or STOCKS)


def run_hourly_scan(stocks=None):
    return _run_scan(scan_hourly, "HOURLY", HOURLY_CONFIG.min_score, stocks or STOCKS)


def run_15m_scan(stocks=None):
    return _run_scan(scan_15min, "15MIN", MIN15_CONFIG.min_score, stocks or STOCKS)


# ---------------------------------------------------------------------------
# COMBINED RUNNER — used for `--timeframe all` (the default).
#
# The three scans above are each internally parallel (one thread pool per
# timeframe), but calling them one after another still means: finish all
# 152 daily downloads -> THEN start hourly -> THEN start 15m. That serializes
# three independent, network-bound jobs for no reason and roughly triples
# the wall-clock time.
#
# Since daily/hourly/15m truly don't depend on each other, this instead
# throws every (symbol, timeframe) pair — up to len(stocks) * 3 tasks — into
# ONE shared thread pool, so all three timeframes download concurrently.
# ---------------------------------------------------------------------------

_TF_LABELS = {"daily": ("DAILY", DAILY_CONFIG), "hourly": ("HOURLY", HOURLY_CONFIG), "15m": ("15MIN", MIN15_CONFIG)}


def run_all_scans(stocks=None) -> dict:
    stocks = stocks or STOCKS
    tasks = [
        (symbol, tf, key)
        for key, (_, tf) in _TF_LABELS.items()
        for symbol in stocks
    ]

    start = time.time()
    logger.info("Starting combined scan (daily + hourly + 15m in parallel)...")
    logger.info("Stocks: %d  |  Total tasks: %d  |  Workers: %d", len(stocks), len(tasks), MAX_WORKERS)

    results_by_tf = {"daily": [], "hourly": [], "15m": []}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(analyze_stock, symbol, tf, key): key for symbol, tf, key in tasks}
        for future in as_completed(futures):
            key = futures[future]
            results_by_tf[key].append(future.result())

    for key, (label, tf) in _TF_LABELS.items():
        results = results_by_tf[key]
        ok = [r for r in results if r.status == "OK"]
        errors = [r for r in results if r.status == "ERROR"]
        qualified = [r for r in ok if r.score is not None and r.score >= tf.min_score]
        logger.info("%s successful downloads: %d, failed: %d", label, len(ok), len(errors))
        logger.info("%s qualified (score >= %d): %d", label, tf.min_score, len(qualified))

    logger.info("Combined scan completed in %.1fs", time.time() - start)
    return results_by_tf


# ===========================================================================
# 9. REPORT GENERATION (HTML dashboard + CSVs)
# ===========================================================================

STATUS_COLOR = {
    "BREAKOUT": "#22C55E", "NEAR BREAKOUT": "#38BDF8",
    "WATCH": "#F59E0B", "WEAK CONSOLIDATION": "#94A3B8",
}
CLASS_COLOR = {
    "STRONG CONSOLIDATION": "#22C55E", "GOOD CONSOLIDATION": "#38BDF8",
    "WATCH": "#F59E0B", "WEAK": "#F87171", "IGNORE": "#64748B",
}


def _fmt(v, decimals=2):
    if v is None:
        return "--"
    try:
        return f"{v:.{decimals}f}"
    except (TypeError, ValueError):
        return str(v)


def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def write_csv(timeframe_key: str, results: list):
    ensure_output_dir()
    path = os.path.join(OUTPUT_DIR, CSV_FILENAMES[timeframe_key])
    qualified = sorted(
        [r for r in results if r.status == "OK" and r.score is not None],
        key=lambda r: (-r.score, r.breakout_distance_percent or 999),
    )
    fieldnames = [
        "symbol", "company", "score", "classification", "breakout_status",
        "price", "range_percent", "breakout_distance_percent", "rsi", "adx",
        "atr_pct", "rel_volume", "ema_fast", "ema_mid", "ema_slow",
        "support", "resistance", "pre_upswing_percent", "reasons",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in qualified:
            row = {k: getattr(r, k) for k in fieldnames if k != "reasons"}
            row["reasons"] = "; ".join(r.reasons)
            writer.writerow(row)
    return path


TEMPLATE = Template(r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Bullish Consolidation Scanner</title>
<style>
  :root {
    --bg: #0B1220; --panel: #121A2B; --panel-border: #223049;
    --text: #E7ECF3; --muted: #8CA0BE; --accent: #38BDF8;
    --green: #22C55E; --amber: #F59E0B; --red: #F87171; --grey: #64748B;
  }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--bg); color: var(--text);
         font-family: -apple-system, "Segoe UI", Inter, sans-serif; line-height: 1.5; }
  .num { font-family: "SFMono-Regular", "IBM Plex Mono", Menlo, monospace; }
  header { padding: 28px 32px 20px; border-bottom: 1px solid var(--panel-border); }
  header h1 { margin: 0 0 4px; font-size: 22px; font-weight: 700; letter-spacing: -0.01em; }
  header .meta { color: var(--muted); font-size: 13px; }
  .summary { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
             gap: 12px; padding: 20px 32px 4px; }
  .card { background: var(--panel); border: 1px solid var(--panel-border); border-radius: 10px; padding: 14px 16px; }
  .card .value { font-size: 26px; font-weight: 700; font-family: "SFMono-Regular", "IBM Plex Mono", Menlo, monospace; }
  .card .label { color: var(--muted); font-size: 12px; margin-top: 2px; }
  section.tf-section { padding: 28px 32px; }
  section.tf-section h2 { font-size: 16px; margin: 0 0 12px; display: flex; align-items: center; gap: 8px; }
  section.tf-section h2 .count { color: var(--muted); font-weight: 400; font-size: 13px; }
  table { width: 100%; border-collapse: collapse; background: var(--panel); border: 1px solid var(--panel-border);
          border-radius: 10px; overflow: hidden; font-size: 13px; }
  thead th { text-align: left; padding: 10px 12px; background: #16203540; color: var(--muted);
             font-weight: 600; border-bottom: 1px solid var(--panel-border); cursor: pointer; white-space: nowrap; }
  thead th:hover { color: var(--text); }
  tbody td { padding: 9px 12px; border-bottom: 1px solid #1a2540; white-space: nowrap; }
  tbody tr:last-child td { border-bottom: none; }
  tbody tr:hover { background: #16213a; cursor: pointer; }
  .badge { display: inline-block; padding: 3px 9px; border-radius: 999px; font-size: 11px; font-weight: 600; color: #0B1220; }
  .score-bar-wrap { width: 70px; height: 6px; background: #1e2b45; border-radius: 4px; overflow: hidden;
                    display: inline-block; vertical-align: middle; margin-right: 8px; }
  .score-bar { height: 100%; }
  .empty { color: var(--muted); font-size: 13px; padding: 16px; }
  /* --- Symbol search box --- */
  .search-wrap { padding: 18px 32px 0; }
  .search-box { position: relative; max-width: 320px; }
  .search-box input { width: 100%; box-sizing: border-box; background: var(--panel);
             border: 1px solid var(--panel-border); color: var(--text); font-size: 13px;
             padding: 9px 34px 9px 34px; border-radius: 8px; outline: none;
             transition: border-color .15s; }
  .search-box input:focus { border-color: var(--accent); }
  .search-box input::placeholder { color: var(--muted); }
  .search-box .icon { position: absolute; left: 11px; top: 50%; transform: translateY(-50%);
             color: var(--muted); font-size: 13px; pointer-events: none; }
  .search-box .clear-btn { position: absolute; right: 8px; top: 50%; transform: translateY(-50%);
             color: var(--muted); background: none; border: none; cursor: pointer; font-size: 15px;
             line-height: 1; padding: 2px 4px; display: none; }
  .search-box .clear-btn:hover { color: var(--text); }
  .search-box.has-value .clear-btn { display: block; }
  .search-empty { color: var(--muted); font-size: 13px; padding: 12px 0; display: none; }
  /* --- Top-level timeframe tabs --- */
  .tabs { display: flex; gap: 6px; padding: 18px 32px 0; flex-wrap: wrap; }
  .tab-btn { background: var(--panel); border: 1px solid var(--panel-border); color: var(--muted);
             padding: 8px 16px; border-radius: 8px 8px 0 0; font-size: 13px; font-weight: 600;
             cursor: pointer; transition: color .15s, background .15s; }
  .tab-btn:hover { color: var(--text); }
  .tab-btn.active { color: var(--bg); background: var(--accent); border-color: var(--accent); }
  /* --- Colored timeframe badge next to section title --- */
  .tf-dot { display: inline-block; width: 9px; height: 9px; border-radius: 50%; flex-shrink: 0; }
  /* --- Quick filter chips --- */
  .chips { display: flex; gap: 8px; flex-wrap: wrap; margin: 0 0 14px; }
  .chip { background: #16203540; border: 1px solid var(--panel-border); color: var(--muted);
          padding: 5px 12px; border-radius: 999px; font-size: 12px; font-weight: 600; cursor: pointer;
          transition: color .15s, background .15s, border-color .15s; }
  .chip:hover { color: var(--text); }
  .chip.active { color: var(--bg); background: var(--text); border-color: var(--text); }
  /* --- Flame icon for high-score / breakout-proximity tickers --- */
  .flame { margin-left: 5px; filter: drop-shadow(0 0 3px rgba(245, 158, 11, .7)); }
  /* --- Row glow for tickers within 1% of breakout --- */
  tbody tr.row-glow { background: linear-gradient(90deg, rgba(56,189,248,.13), rgba(34,197,94,.08) 70%);
                       box-shadow: inset 3px 0 0 0 var(--accent); }
  tbody tr.row-glow:hover { background: linear-gradient(90deg, rgba(56,189,248,.2), rgba(34,197,94,.13) 70%); }
  /* --- Sort direction arrow in table headers --- */
  thead th .arrow { margin-left: 4px; color: var(--accent); font-size: 10px; }
  thead th[title] { text-decoration: underline dotted var(--panel-border) 1px; text-underline-offset: 3px; }
  .modal-backdrop { display: none; position: fixed; inset: 0; background: rgba(5, 9, 17, 0.72);
                     align-items: center; justify-content: center; z-index: 50; }
  .modal-backdrop.open { display: flex; }
  .modal { background: var(--panel); border: 1px solid var(--panel-border); border-radius: 12px;
           width: min(560px, 92vw); max-height: 84vh; overflow-y: auto; padding: 24px; }
  .modal h3 { margin: 0 0 2px; font-size: 18px; }
  .modal .sub { color: var(--muted); font-size: 13px; margin-bottom: 16px; }
  .modal .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px 20px; margin-bottom: 18px; }
  .modal .grid .k { color: var(--muted); font-size: 11px; }
  .modal .grid .v { font-family: "SFMono-Regular", "IBM Plex Mono", Menlo, monospace; font-size: 14px; }
  .modal .reasons { list-style: none; padding: 0; margin: 0; }
  .modal .reasons li { padding: 6px 0; border-bottom: 1px solid #1a2540; font-size: 13px; color: #C9D4E4; }
  .modal .close { float: right; cursor: pointer; color: var(--muted); font-size: 20px; line-height: 1; }
  footer { padding: 20px 32px 40px; color: var(--muted); font-size: 12px; }
</style>
</head>
<body>
<header>
  <h1>Bullish Consolidation Scanner</h1>
  <div class="meta">Generated {{ generated_at }} &middot; Technical screen only, not financial advice</div>
</header>
<div class="summary">
  <div class="card"><div class="value">{{ summary.daily }}</div><div class="label">Daily setups</div></div>
  <div class="card"><div class="value">{{ summary.hourly }}</div><div class="label">Hourly setups</div></div>
  <div class="card"><div class="value">{{ summary.min15 }}</div><div class="label">15M setups</div></div>
  <div class="card"><div class="value" style="color:var(--green)">🔥 {{ summary.strong }}</div><div class="label">Strong setups</div></div>
  <div class="card"><div class="value" style="color:var(--accent)">⚡ {{ summary.near_breakout }}</div><div class="label">Near breakout</div></div>
  <div class="card"><div class="value" style="color:var(--green)">🚀 {{ summary.breakouts }}</div><div class="label">Breakouts</div></div>
</div>
<div class="search-wrap">
  <div class="search-box" id="searchBox">
    <span class="icon">&#128269;</span>
    <input type="text" id="symbolSearch" placeholder="Search symbol (e.g. AAPL)" autocomplete="off" spellcheck="false">
    <button type="button" class="clear-btn" id="searchClear" title="Clear">&times;</button>
  </div>
</div>
<div class="tabs" id="tfTabs">
  <button class="tab-btn active" data-target="all">All Timeframes</button>
  {% for sec in sections %}
  <button class="tab-btn" data-target="{{ sec.key }}">{{ sec.tab_label }}</button>
  {% endfor %}
</div>
{% for sec in sections %}
<section class="tf-section" id="section-{{ sec.key }}" data-section-key="{{ sec.key }}">
  <h2><span class="tf-dot" style="background:{{ sec.badge_color }}"></span>{{ sec.title }} <span class="count">{{ sec.rows|length }} qualifying</span></h2>
  {% if sec.rows %}
  <div class="chips" data-chips data-section="{{ sec.key }}">
    <button class="chip active" data-filter="all">All ({{ sec.rows|length }})</button>
    <button class="chip" data-filter="strong-consolidation">🔥 Strong ({{ sec.n_strong }})</button>
    <button class="chip" data-filter="near-breakout">⚡ Near Breakout ({{ sec.n_near }})</button>
    <button class="chip" data-filter="breakout">🚀 Breakout ({{ sec.n_breakout }})</button>
    <button class="chip" data-filter="watch">👁 Watch ({{ sec.n_watch }})</button>
  </div>
  <table data-table>
    <thead><tr>
      <th>Rank</th><th>Symbol</th><th>Company</th><th>Index</th><th>Score</th><th>Status</th>
      <th>Price</th><th>Range %</th>
      <th title="Distance from current price to resistance (range high). Under 1% gets highlighted as an imminent setup.">Breakout Dist %</th>
      <th title="Relative Strength Index. 45-65 is this scanner's bullish-neutral sweet spot; above 70 is overbought, below 30 is oversold.">RSI</th>
      <th title="Average Directional Index. A falling ADX during the base means the stock is losing directional strength, i.e. consolidating.">ADX</th>
      <th title="Average True Range as % of price. A falling ATR% during the base means volatility is contracting, a bullish tightening signal.">ATR %</th>
      <th title="Current volume vs its N-period average. Above 1.5x on a breakout day is this scanner's volume-confirmation threshold.">Rel Vol</th>
      <th>EMA20</th><th>EMA50</th>
      <th>Support</th><th>Resistance</th><th>Prior Move %</th>
    </tr></thead>
    <tbody>
      {% for r in sec.rows %}
      <tr data-detail="{{ loop.index0 }}" data-section="{{ sec.key }}" data-tags="{{ r.class_slug }} {{ r.status_slug }}" data-symbol="{{ r.symbol }}" class="{{ r.row_class }}">
        <td>{{ loop.index }}</td>
        <td><strong>{{ r.symbol }}</strong>{% if r.flame %}<span class="flame" title="High score or breakout proximity">🔥</span>{% endif %}</td>
        <td>{{ r.company }}</td>
        <td>{{ r.sector }}</td>
        <td class="num"><span class="score-bar-wrap"><span class="score-bar" style="width:{{ r.score }}%; background:{{ r.class_color }};"></span></span>{{ r.score }}</td>
        <td><span class="badge" style="background:{{ r.status_color }}">{{ r.breakout_status }}</span></td>
        <td class="num">{{ r.price }}</td>
        <td class="num">{{ r.range_percent }}</td>
        <td class="num">{{ r.breakout_distance_percent }}</td>
        <td class="num">{{ r.rsi }}</td>
        <td class="num">{{ r.adx }}</td>
        <td class="num">{{ r.atr_pct }}</td>
        <td class="num">{{ r.rel_volume }}</td>
        <td class="num">{{ r.ema_fast }}</td>
        <td class="num">{{ r.ema_mid }}</td>
        <td class="num">{{ r.support }}</td>
        <td class="num">{{ r.resistance }}</td>
        <td class="num">{{ r.pre_upswing_percent }}</td>
      </tr>
      {% endfor %}
    </tbody>
  </table>
  <div class="search-empty" data-empty-search="{{ sec.key }}">No matches for that symbol on this timeframe.</div>
  {% else %}
  <div class="empty">No stocks currently qualify on this timeframe (min score {{ sec.min_score }}).</div>
  {% endif %}
</section>
{% endfor %}
<footer>
  Technical conditions only &mdash; this report does not constitute financial advice.
  Daily / Hourly / 15-Minute scans run independently; a stock may appear in one, several, or none.
</footer>
<div class="modal-backdrop" id="modalBackdrop"><div class="modal" id="modalBody"></div></div>
<script>
  const DATA = {{ detail_json | safe }};
  document.querySelectorAll('tbody tr[data-detail]').forEach(tr => {
    tr.addEventListener('click', () => {
      const key = tr.dataset.section + ':' + tr.dataset.detail;
      const d = DATA[key];
      if (!d) return;
      const reasons = d.reasons.map(r => `<li>${r}</li>`).join('');
      document.getElementById('modalBody').innerHTML = `
        <span class="close" id="modalClose">&times;</span>
        <h3>${d.symbol}</h3>
        <div class="sub">${d.company} &middot; ${d.sector} &middot; ${d.timeframe}</div>
        <div class="grid">
          <div><span class="k">Current Price</span><span class="v">${d.price}</span></div>
          <div><span class="k">Score</span><span class="v">${d.score} (${d.classification})</span></div>
          <div><span class="k">Status</span><span class="v">${d.breakout_status}</span></div>
          <div><span class="k">Trend</span><span class="v">EMA20 ${d.ema_fast} / EMA50 ${d.ema_mid} / EMA200 ${d.ema_slow}</span></div>
          <div><span class="k">Consolidation Range</span><span class="v">${d.range_percent}%</span></div>
          <div><span class="k">Support / Resistance</span><span class="v">${d.support} / ${d.resistance}</span></div>
          <div><span class="k">RSI</span><span class="v">${d.rsi}</span></div>
          <div><span class="k">ADX</span><span class="v">${d.adx}</span></div>
          <div><span class="k">ATR %</span><span class="v">${d.atr_pct}</span></div>
          <div><span class="k">Relative Volume</span><span class="v">${d.rel_volume}</span></div>
          <div><span class="k">Prior Upswing</span><span class="v">${d.pre_upswing_percent}%</span></div>
          <div><span class="k">Breakout Distance</span><span class="v">${d.breakout_distance_percent}%</span></div>
        </div>
        <div class="k" style="margin-bottom:8px;">Why this stock qualified</div>
        <ul class="reasons">${reasons}</ul>
      `;
      document.getElementById('modalBackdrop').classList.add('open');
      document.getElementById('modalClose').addEventListener('click', () => {
        document.getElementById('modalBackdrop').classList.remove('open');
      });
    });
  });
  document.getElementById('modalBackdrop').addEventListener('click', (e) => {
    if (e.target.id === 'modalBackdrop') e.target.classList.remove('open');
  });
  document.querySelectorAll('table[data-table]').forEach(table => {
    const headers = table.querySelectorAll('th');
    headers.forEach((th, idx) => {
      let asc = true;
      th.addEventListener('click', () => {
        const tbody = table.querySelector('tbody');
        const rows = Array.from(tbody.querySelectorAll('tr'));
        rows.sort((a, b) => {
          const av = a.children[idx].innerText.trim();
          const bv = b.children[idx].innerText.trim();
          const an = parseFloat(av.replace(/[^0-9.\-]/g, ''));
          const bn = parseFloat(bv.replace(/[^0-9.\-]/g, ''));
          const bothNumeric = !isNaN(an) && !isNaN(bn);
          if (bothNumeric) return asc ? an - bn : bn - an;
          return asc ? av.localeCompare(bv) : bv.localeCompare(av);
        });
        rows.forEach(r => tbody.appendChild(r));
        // Sort-direction arrow: clear every header in this table, then mark the active one.
        headers.forEach(h => { h.querySelector('.arrow')?.remove(); });
        const arrow = document.createElement('span');
        arrow.className = 'arrow';
        arrow.textContent = asc ? '\u25B2' : '\u25BC';
        th.appendChild(arrow);
        asc = !asc;
      });
    });
  });
  // --- Top-level timeframe tabs: show only the selected section (or all) ---
  document.querySelectorAll('#tfTabs .tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('#tfTabs .tab-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      const target = btn.dataset.target;
      document.querySelectorAll('section.tf-section').forEach(sec => {
        sec.style.display = (target === 'all' || sec.dataset.sectionKey === target) ? '' : 'none';
      });
    });
  });
  // --- Per-section quick filter chips (Strong / Near Breakout / Breakout / Watch)
  //     combined with the symbol search box below, applied together (AND). ---
  function activeChipFilter(sectionKey) {
    const chipBar = document.querySelector(`[data-chips][data-section="${sectionKey}"]`);
    const active = chipBar ? chipBar.querySelector('.chip.active') : null;
    return active ? active.dataset.filter : 'all';
  }

  function applySectionFilters(sectionKey) {
    const query = (document.getElementById('symbolSearch').value || '').trim().toUpperCase();
    const chipFilter = activeChipFilter(sectionKey);
    const rows = document.querySelectorAll(`tbody tr[data-section="${sectionKey}"]`);
    let visibleCount = 0;
    rows.forEach(tr => {
      const tags = (tr.dataset.tags || '').split(' ');
      const tagOk = (chipFilter === 'all' || tags.includes(chipFilter));
      const symOk = (query === '' || (tr.dataset.symbol || '').toUpperCase().includes(query));
      const show = tagOk && symOk;
      tr.style.display = show ? '' : 'none';
      if (show) visibleCount++;
    });
    const emptyMsg = document.querySelector(`[data-empty-search="${sectionKey}"]`);
    if (emptyMsg) emptyMsg.style.display = (visibleCount === 0 && rows.length > 0) ? '' : 'none';
  }

  const allSectionKeys = Array.from(document.querySelectorAll('section.tf-section'))
    .map(sec => sec.dataset.sectionKey);

  function applyAllSectionFilters() {
    allSectionKeys.forEach(applySectionFilters);
  }

  document.querySelectorAll('[data-chips]').forEach(chipBar => {
    const sectionKey = chipBar.dataset.section;
    chipBar.querySelectorAll('.chip').forEach(chip => {
      chip.addEventListener('click', () => {
        chipBar.querySelectorAll('.chip').forEach(c => c.classList.remove('active'));
        chip.classList.add('active');
        applySectionFilters(sectionKey);
      });
    });
  });

  // --- Symbol search box: type a ticker to show only matching rows, across
  //     every timeframe section at once (regardless of the active chip). ---
  const searchInput = document.getElementById('symbolSearch');
  const searchBox = document.getElementById('searchBox');
  const searchClear = document.getElementById('searchClear');
  searchInput.addEventListener('input', () => {
    searchBox.classList.toggle('has-value', searchInput.value.trim() !== '');
    applyAllSectionFilters();
  });
  searchClear.addEventListener('click', () => {
    searchInput.value = '';
    searchBox.classList.remove('has-value');
    applyAllSectionFilters();
    searchInput.focus();
  });
</script>
</body>
</html>
""")


def _slug(text: Optional[str]) -> str:
    """'NEAR BREAKOUT' -> 'near-breakout' — used as a filter-chip/row data tag."""
    return (text or "na").strip().lower().replace(" ", "-")


def _row_dict(r: StockReport):
    is_flame = (r.score is not None and r.score >= 80) or r.breakout_status in ("BREAKOUT", "NEAR BREAKOUT")
    is_glow = r.breakout_distance_percent is not None and r.breakout_distance_percent < 1
    return {
        "symbol": r.symbol, "company": r.company, "sector": r.sector, "score": r.score,
        "class_color": CLASS_COLOR.get(r.classification, "#64748B"),
        "status_color": STATUS_COLOR.get(r.breakout_status, "#64748B"),
        "breakout_status": r.breakout_status,
        "classification": r.classification,
        "class_slug": _slug(r.classification),
        "status_slug": _slug(r.breakout_status),
        "row_class": "row-glow" if is_glow else "",
        "flame": is_flame,
        "price": _fmt(r.price), "range_percent": _fmt(r.range_percent),
        "breakout_distance_percent": _fmt(r.breakout_distance_percent),
        "rsi": _fmt(r.rsi, 0), "adx": _fmt(r.adx, 0), "atr_pct": _fmt(r.atr_pct),
        "rel_volume": _fmt(r.rel_volume), "ema_fast": _fmt(r.ema_fast), "ema_mid": _fmt(r.ema_mid),
        "ema_slow": _fmt(r.ema_slow), "support": _fmt(r.support), "resistance": _fmt(r.resistance),
        "pre_upswing_percent": _fmt(r.pre_upswing_percent),
    }


# Distinct color per timeframe, used for the tab-adjacent dot and section badge.
TF_BADGE_COLOR = {"daily": "#38BDF8", "hourly": "#A78BFA", "15m": "#2DD4BF"}
TF_TAB_LABEL = {"daily": "Daily", "hourly": "Hourly", "15m": "15 Min"}


def generate_html_report(results_by_tf: dict):
    ensure_output_dir()
    section_defs = [
        ("daily", "Daily Consolidation", DAILY_CONFIG.min_score),
        ("hourly", "Hourly Consolidation", HOURLY_CONFIG.min_score),
        ("15m", "15 Minute Consolidation", MIN15_CONFIG.min_score),
    ]

    sections = []
    detail_json = {}
    summary = {"daily": 0, "hourly": 0, "min15": 0, "strong": 0, "near_breakout": 0, "breakouts": 0}

    for key, title, min_score in section_defs:
        results = results_by_tf.get(key, [])
        qualified = sorted(
            [r for r in results if r.status == "OK" and r.score is not None and r.score >= min_score],
            key=lambda r: (-r.score, r.breakout_distance_percent if r.breakout_distance_percent is not None else 999),
        )
        rows = [_row_dict(r) for r in qualified]
        sections.append({
            "key": key, "title": title, "rows": rows, "min_score": min_score,
            "badge_color": TF_BADGE_COLOR.get(key, "#64748B"),
            "tab_label": TF_TAB_LABEL.get(key, key),
            "n_strong": sum(1 for r in qualified if r.classification == "STRONG CONSOLIDATION"),
            "n_near": sum(1 for r in qualified if r.breakout_status == "NEAR BREAKOUT"),
            "n_breakout": sum(1 for r in qualified if r.breakout_status == "BREAKOUT"),
            "n_watch": sum(1 for r in qualified if r.classification == "WATCH" or r.breakout_status == "WATCH"),
        })

        if key == "daily":
            summary["daily"] = len(qualified)
        elif key == "hourly":
            summary["hourly"] = len(qualified)
        else:
            summary["min15"] = len(qualified)

        for i, r in enumerate(qualified):
            summary["strong"] += 1 if r.classification == "STRONG CONSOLIDATION" else 0
            summary["near_breakout"] += 1 if r.breakout_status == "NEAR BREAKOUT" else 0
            summary["breakouts"] += 1 if r.breakout_status == "BREAKOUT" else 0
            detail_json[f"{key}:{i}"] = {
                "symbol": r.symbol, "company": r.company, "sector": r.sector, "timeframe": title,
                "price": _fmt(r.price), "score": r.score, "classification": r.classification,
                "breakout_status": r.breakout_status,
                "ema_fast": _fmt(r.ema_fast), "ema_mid": _fmt(r.ema_mid), "ema_slow": _fmt(r.ema_slow),
                "range_percent": _fmt(r.range_percent), "support": _fmt(r.support), "resistance": _fmt(r.resistance),
                "rsi": _fmt(r.rsi, 0), "adx": _fmt(r.adx, 0), "atr_pct": _fmt(r.atr_pct),
                "rel_volume": _fmt(r.rel_volume), "pre_upswing_percent": _fmt(r.pre_upswing_percent),
                "breakout_distance_percent": _fmt(r.breakout_distance_percent),
                "reasons": r.reasons,
            }

    html = TEMPLATE.render(
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        summary=summary, sections=sections, detail_json=json.dumps(detail_json),
    )
    path = os.path.join(OUTPUT_DIR, HTML_REPORT_FILENAME)
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    return path


# ===========================================================================
# TELEGRAM SUMMARY (Daily timeframe only — top N per Index)
# ===========================================================================

def _escape_html(text: str) -> str:
    return (text or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


TELEGRAM_CARD_DIVIDER = "\u2501" * 18  # ━━━━━━━━━━━━━━━━━━


def build_daily_telegram_message(results_by_tf: dict) -> Optional[str]:
    """Group the DAILY report's qualifying rows by Index (sector) and keep
    only the top TELEGRAM_TOP_N_PER_INDEX (by score) within each index.
    Renders a premium "card" layout (see TELEGRAM_CARD_DIVIDER). Returns
    None if there's nothing qualifying to send (message is skipped).
    """
    daily_results = results_by_tf.get("daily", [])
    qualified = sorted(
        [r for r in daily_results if r.status == "OK" and r.score is not None and r.score >= DAILY_CONFIG.min_score],
        key=lambda r: (-r.score, r.breakout_distance_percent if r.breakout_distance_percent is not None else 999),
    )
    if not qualified:
        return None

    # Overall rank = this stock's position in the FULL daily report (the
    # same "Rank" column shown in the HTML table), not a per-index counter.
    rank_by_symbol = {r.symbol: rank for rank, r in enumerate(qualified, start=1)}

    # Walk the already best-first-sorted list and keep the first N seen per
    # index -> that's the top N by score for that index, in order.
    by_index: dict = {}
    for r in qualified:
        idx_label = r.sector or "Other"
        bucket = by_index.setdefault(idx_label, [])
        if len(bucket) < TELEGRAM_TOP_N_PER_INDEX:
            bucket.append(r)

    date_str = datetime.now().strftime("%Y-%m-%d %H:%M")

    lines = [
        TELEGRAM_CARD_DIVIDER,
        f"\U0001F4CA {_escape_html(MARKET_LABEL)} Daily Consolidation",
        f"\U0001F4C5 {date_str}",
        TELEGRAM_CARD_DIVIDER,
    ]
    for idx_label, rows in by_index.items():
        raw_code = SECTOR_MAP.get(rows[0].symbol, "")
        emoji = SECTOR_EMOJI.get(raw_code, "\U0001F4E6")  # 📦 fallback
        lines.append(f"{emoji} {_escape_html(idx_label)}")
        for r in rows:
            icon, short_status = CLASSIFICATION_DISPLAY.get(r.classification or "", ("\u26AA", r.classification or ""))
            lines.append(f"\u2022 #{rank_by_symbol[r.symbol]} {_escape_html(r.symbol)} \u2014 {r.score} \u2014 {icon} {short_status}")
        lines.append("")  # blank line between sector blocks

    return "\n".join(lines).strip()


def send_telegram_message(text: str) -> bool:
    """Post `text` to the configured Telegram chat. Never raises — logs and
    returns False on any failure so it can't crash the scan run."""
    if not TELEGRAM_ENABLED:
        logger.info("Telegram sending is disabled (set TELEGRAM_ENABLED=true, or via telegram_config.py) — skipping.")
        return False
    if not TELEGRAM_BOT_TOKEN or "PUT_YOUR" in TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID or "PUT_YOUR" in str(TELEGRAM_CHAT_ID):
        logger.warning("Telegram bot token / chat id not configured (env vars or telegram_config.py) — skipping send.")
        return False

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }
    try:
        resp = requests.post(url, data=payload, timeout=15)
        if resp.status_code == 200 and resp.json().get("ok"):
            logger.info("Telegram summary sent successfully.")
            return True
        logger.error("Telegram send failed (%s): %s", resp.status_code, resp.text[:300])
        return False
    except Exception as e:  # noqa: BLE001 - never let a notification failure kill the run
        logger.error("Telegram send raised an exception: %s", e)
        return False


# ===========================================================================
# 10. MAIN ENTRY POINT
# ===========================================================================

def setup_logging():
    ensure_output_dir()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger("scanner")


def main():
    parser = argparse.ArgumentParser(description="Bullish consolidation scanner (single-file version)")
    parser.add_argument("--timeframe", choices=["daily", "hourly", "15m", "all"], default="all",
                         help="Which scanner(s) to run (default: all)")
    args = parser.parse_args()

    global logger
    logger = setup_logging()
    overall_start = time.time()
    logger.info("=" * 60)
    logger.info("Bullish Consolidation Scanner starting (timeframe=%s)", args.timeframe)
    logger.info("Universe: %d stocks", len(STOCKS))

    results_by_tf = {"daily": [], "hourly": [], "15m": []}
    run_all = args.timeframe == "all"

    if run_all:
        # All 3 timeframes downloaded concurrently in ONE shared thread pool
        # (see run_all_scans) instead of daily -> hourly -> 15m sequentially.
        results_by_tf = run_all_scans()
    elif args.timeframe == "daily":
        results_by_tf["daily"] = run_daily_scan()
    elif args.timeframe == "hourly":
        results_by_tf["hourly"] = run_hourly_scan()
    elif args.timeframe == "15m":
        results_by_tf["15m"] = run_15m_scan()

    report_path = generate_html_report(results_by_tf)

    logger.info("HTML report written to %s", report_path)

    if results_by_tf.get("daily"):
        telegram_text = build_daily_telegram_message(results_by_tf)
        if telegram_text:
            send_telegram_message(telegram_text)
        else:
            logger.info("No daily-timeframe stocks qualified — nothing to send to Telegram.")

    logger.info("Total execution time: %.1fs", time.time() - overall_start)
    logger.info("=" * 60)

    print(f"\nDone. Open {report_path} in your browser to view the dashboard.")


if __name__ == "__main__":
    main()
