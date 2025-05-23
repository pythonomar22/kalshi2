# /random/backtest/live_backtest_config.py

from pathlib import Path
import datetime as dt
from datetime import timezone
import logging

# --- Initial logging setup for this config file itself ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger_live_config = logging.getLogger("live_backtest_config")

# --- Core Paths ---
BASE_PROJECT_DIR = Path("/Users/omarabul-hassan/Desktop/projects/kalshi") # Your project root
RANDOM_DIR = BASE_PROJECT_DIR / "random"
LIVE_BACKTEST_DIR = RANDOM_DIR / "backtest"

# --- Data Directories for "Live" Collected Data ---
LIVE_KALSHI_DATA_DIR = RANDOM_DIR / "market_data_logs"
LIVE_BINANCE_DATA_DIR = RANDOM_DIR / "binance_market_data_logs"
LIVE_MARKET_OUTCOMES_CSV = LIVE_BACKTEST_DIR / "live_sessions_market_outcomes.csv"

logger_live_config.info(f"Live Kalshi data dir: {LIVE_KALSHI_DATA_DIR}")
logger_live_config.info(f"Live Binance data dir: {LIVE_BINANCE_DATA_DIR}")
logger_live_config.info(f"Live market outcomes CSV: {LIVE_MARKET_OUTCOMES_CSV}")

# --- Model & Scaler Paths (from historical training) ---
MODEL_TYPE_SUFFIX = "no_vol_oi"
MODEL_TYPE_BASE = "logreg_per_minute"
HISTORICAL_TRAINED_MODELS_BASE_DIR = BASE_PROJECT_DIR / "notebooks" / "trained_models"

MODEL_DIR_NAME = f"{MODEL_TYPE_BASE}_{MODEL_TYPE_SUFFIX}"
MODEL_DIR = HISTORICAL_TRAINED_MODELS_BASE_DIR / MODEL_DIR_NAME
MODEL_PATH = MODEL_DIR / f"{MODEL_DIR_NAME}_model.joblib"
SCALER_PATH = MODEL_DIR / f"{MODEL_DIR_NAME}_scaler.joblib"
FEATURE_NAMES_PATH = MODEL_DIR / f"{MODEL_DIR_NAME}_feature_names.json"

logger_live_config.info(f"Using model from directory: {MODEL_DIR}")
logger_live_config.info(f"Model path: {MODEL_PATH}")
logger_live_config.info(f"Scaler path: {SCALER_PATH}")
logger_live_config.info(f"Feature names path: {FEATURE_NAMES_PATH}")

# --- Mapping from Kalshi Session (from ticker) to Binance File ---
SESSION_TO_BINANCE_FILE_MAP = {
    "25MAY1920": "btcusdt_kline_1m.csv", "25MAY2015": "btcusdt_2kline_1m.csv",
    "25MAY2016": "btcusdt_3kline_1m.csv", "25MAY2017": "btcusdt_4kline_1m.csv",
    "25MAY2018": "btcusdt_5kline_1m.csv", "25MAY2019": "btcusdt_6kline_1m.csv",
    "25MAY2117": "btcusdt_7kline_1m.csv", "25MAY2118": "btcusdt_8kline_1m.csv",
    "25MAY2119": "btcusdt_9kline_1m.csv", "25MAY2120": "btcusdt_10kline_1m.csv",
    "25MAY2121": "btcusdt_11kline_1m.csv", "25MAY2122": "btcusdt_12kline_1m.csv",
    "25MAY2123": "btcusdt_13kline_1m.csv", "25MAY2219": "btcusdt_14kline_1m.csv",
    "25MAY2221": "btcusdt_15kline_1m.csv", "25MAY2222": "btcusdt_16kline_1m.csv",
    "25MAY2223": "btcusdt_17kline_1m.csv", "25MAY2300": "btcusdt_18kline_1m.csv"
}
logger_live_config.info(f"Loaded {len(SESSION_TO_BINANCE_FILE_MAP)} session-to-Binance file mappings.")

# --- Trading Parameters ---
PROBABILITY_THRESHOLD_YES = 0.96 # Minimum probability to consider a trade
PROBABILITY_THRESHOLD_NO = 0.96
ONE_BET_PER_KALSHI_MARKET = False

logger_live_config.info(f"Trading Thresholds: Min P(Yes) > {PROBABILITY_THRESHOLD_YES} for BUY_YES; Min P(No) > {PROBABILITY_THRESHOLD_NO} for BUY_NO")
logger_live_config.info(f"ONE_BET_PER_KALSHI_MARKET set to: {ONE_BET_PER_KALSHI_MARKET}")

# --- Kelly Criterion Sizing Parameters ---
USE_KELLY_CRITERION = True # Set to True to enable Kelly sizing
INITIAL_CAPITAL_CENTS = 50000 # Example: $5000, in cents
KELLY_FRACTION = 0.5 # Use 10% of full Kelly suggestion (fractional Kelly)
MAX_PCT_CAPITAL_PER_TRADE = 0.2 # Max 5% of current capital on a single trade, regardless of Kelly
MIN_CONTRACTS_TO_TRADE = 1
MAX_CONTRACTS_TO_TRADE = 100 # Max contracts per single trade execution

logger_live_config.info(f"USE_KELLY_CRITERION: {USE_KELLY_CRITERION}")
if USE_KELLY_CRITERION:
    logger_live_config.info(f"  Initial Capital: {INITIAL_CAPITAL_CENTS / 100:.2f} USD")
    logger_live_config.info(f"  Kelly Fraction: {KELLY_FRACTION}")
    logger_live_config.info(f"  Max % Capital per Trade: {MAX_PCT_CAPITAL_PER_TRADE*100:.1f}%")
    logger_live_config.info(f"  Min Contracts per Trade: {MIN_CONTRACTS_TO_TRADE}")
    logger_live_config.info(f"  Max Contracts per Trade: {MAX_CONTRACTS_TO_TRADE}")


# --- Feature Engineering Parameters (for live data) ---
MIN_MINUTES_BEFORE_RESOLUTION_FOR_DECISION = 1
LAG_WINDOWS_MINUTES = [1, 3, 5, 10, 15, 30]
ROLLING_WINDOWS_MINUTES = [5, 15, 30]

logger_live_config.info(f"Live Feature Eng: Decisions up to T-{MIN_MINUTES_BEFORE_RESOLUTION_FOR_DECISION}m.")
logger_live_config.info(f"Live Feature Eng: Lag windows: {LAG_WINDOWS_MINUTES} mins.")
logger_live_config.info(f"Live Feature Eng: Rolling windows: {ROLLING_WINDOWS_MINUTES} mins.")

# --- Logging for the Backtest Run ---
LOG_DIR = LIVE_BACKTEST_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
logger_live_config.info(f"Live backtest trade logs will be saved in: {LOG_DIR}")

# --- Timezone for live Kalshi data timestamps before conversion to UTC ---
KALSHI_RAW_DATA_TIMEZONE = "America/Los_Angeles"
logger_live_config.info(f"Assuming raw Kalshi data timestamps are in: {KALSHI_RAW_DATA_TIMEZONE}")