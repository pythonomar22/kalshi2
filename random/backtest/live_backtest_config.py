# /Users/omarabul-hassan/Desktop/projects/kalshi/random/backtest/live_backtest_config.py

from pathlib import Path
import datetime as dt
from datetime import timezone
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger_config = logging.getLogger("live_backtest_config")

# --- Core Paths ---
BASE_PROJECT_DIR = Path("/Users/omarabul-hassan/Desktop/projects/kalshi")
RANDOM_DIR = BASE_PROJECT_DIR / "random"
LIVE_BACKTEST_DIR = RANDOM_DIR / "backtest"

LIVE_KALSHI_DATA_DIR = RANDOM_DIR / "market_data_logs"
LIVE_BINANCE_DATA_DIR = RANDOM_DIR / "binance_market_data_logs"
LIVE_FEATURES_DIR = LIVE_BACKTEST_DIR / "features_live" 
LOG_DIR = LIVE_BACKTEST_DIR / "logs" # Log directory for this live backtester
LIVE_FEATURES_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# --- Market Outcomes File (from fetch.py) ---
LIVE_MARKET_OUTCOMES_CSV = LIVE_BACKTEST_DIR / "live_sessions_market_outcomes.csv"
logger_config.info(f"Live market outcomes expected from: {LIVE_MARKET_OUTCOMES_CSV}")

# --- Mapping from Kalshi session (e.g., 25MAY1920) to Binance file ---
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
logger_config.info(f"Loaded {len(SESSION_TO_BINANCE_FILE_MAP)} session-to-Binance file mappings.")

# --- Model & Scaler Paths (Using the NEW retrained model) ---
# These point to your newly trained model from the historical data.
# *** MODIFIED TO POINT TO THE NEW MODEL VERSION ***
MODEL_VERSION_SUFFIX = "no_vol_oi" 
MODEL_TYPE_BASE = "logreg_per_minute"

MODEL_TYPE_DIR_NAME = f"{MODEL_TYPE_BASE}_{MODEL_VERSION_SUFFIX}"
MODEL_FILENAME_BASE = f"{MODEL_TYPE_BASE}_{MODEL_VERSION_SUFFIX}" # Filenames also include the suffix

# Base directory where all 'trained_models' subdirectories are located
TRAINED_MODELS_BASE_DIR = BASE_PROJECT_DIR / "notebooks" / "trained_models" 

MODEL_DIR = TRAINED_MODELS_BASE_DIR / MODEL_TYPE_DIR_NAME # e.g., .../trained_models/logreg_per_minute_no_vol_oi
MODEL_PATH = MODEL_DIR / f"{MODEL_FILENAME_BASE}_model.joblib"
SCALER_PATH = MODEL_DIR / f"{MODEL_FILENAME_BASE}_scaler.joblib"
FEATURE_NAMES_PATH = MODEL_DIR / f"{MODEL_FILENAME_BASE}_feature_names.json"

logger_config.info(f"Using NEW model from directory: {MODEL_DIR}")
logger_config.info(f"Model path: {MODEL_PATH}")
logger_config.info(f"Scaler path: {SCALER_PATH}")
logger_config.info(f"Feature names path: {FEATURE_NAMES_PATH}")


# --- Trading Parameters (same as historical for now) ---
PROBABILITY_THRESHOLD_YES = 0.60
PROBABILITY_THRESHOLD_NO = 0.60
logger_config.info(f"Trading Thresholds: P(Yes) > {PROBABILITY_THRESHOLD_YES} for BUY_YES; P(No) > {PROBABILITY_THRESHOLD_NO} for BUY_NO")

# --- Feature Engineering Constants for Live Data ---
MIN_MINUTES_BEFORE_RESOLUTION_FOR_DECISION = 1
LAG_WINDOWS_MINUTES = [1, 3, 5, 10, 15, 30]
ROLLING_WINDOWS_MINUTES = [5, 15, 30]

# --- Output Feature File Name for Live Data ---
LATEST_LIVE_FEATURES_CSV_GLOB_PATTERN = str(LIVE_FEATURES_DIR / "kalshi_live_decision_features_*.csv")
logger_config.info(f"Live feature files will be searched using pattern: {LATEST_LIVE_FEATURES_CSV_GLOB_PATTERN}")

# --- Other Backtest Parameters ---
INITIAL_CAPITAL = 50000 
ONE_BET_PER_KALSHI_MARKET = False 
logger_config.info(f"ONE_BET_PER_KALSHI_MARKET set to: {ONE_BET_PER_KALSHI_MARKET}")