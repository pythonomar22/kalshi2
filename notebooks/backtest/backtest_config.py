# /notebooks/backtest/backtest_config.py

from pathlib import Path
import datetime as dt
from datetime import timezone
import logging # Import logging

# --- Initial logging setup for this config file itself ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger_config = logging.getLogger("backtest_config") 

# --- Core Paths ---
BASE_PROJECT_DIR = Path("/Users/omarabul-hassan/Desktop/projects/kalshi")
NOTEBOOKS_DIR = BASE_PROJECT_DIR / "notebooks"
FEATURES_DIR = NOTEBOOKS_DIR / "features"
TRAINED_MODELS_BASE_DIR = NOTEBOOKS_DIR / "trained_models"
BACKTEST_DIR = NOTEBOOKS_DIR / "backtest"
LOG_DIR = BACKTEST_DIR / "logs" # Log directory for this historical backtester

# --- Model & Scaler Paths (for per_minute model) ---
# *** THIS IS THE MODEL YOU INTEND TO USE FOR THIS HISTORICAL BACKTEST RUN ***
# Change MODEL_TYPE to "logreg_per_minute" if you want to run the original model
# Change MODEL_TYPE to "logreg_per_minute_no_vol_oi" if you want to run the newly retrained model
MODEL_TYPE_SUFFIX = "no_vol_oi" # Or "" for the original model
MODEL_TYPE_BASE = "logreg_per_minute"

if MODEL_TYPE_SUFFIX:
    MODEL_TYPE_DIR_NAME = f"{MODEL_TYPE_BASE}_{MODEL_TYPE_SUFFIX}"
    MODEL_FILENAME_BASE = f"{MODEL_TYPE_BASE}_{MODEL_TYPE_SUFFIX}"
else:
    MODEL_TYPE_DIR_NAME = MODEL_TYPE_BASE
    MODEL_FILENAME_BASE = MODEL_TYPE_BASE

MODEL_DIR = TRAINED_MODELS_BASE_DIR / MODEL_TYPE_DIR_NAME
MODEL_PATH = MODEL_DIR / f"{MODEL_FILENAME_BASE}_model.joblib"
SCALER_PATH = MODEL_DIR / f"{MODEL_FILENAME_BASE}_scaler.joblib"
FEATURE_NAMES_PATH = MODEL_DIR / f"{MODEL_FILENAME_BASE}_feature_names.json"

logger_config.info(f"Using model from directory: {MODEL_DIR}")
logger_config.info(f"Model path: {MODEL_PATH}")
logger_config.info(f"Scaler path: {SCALER_PATH}")
logger_config.info(f"Feature names path: {FEATURE_NAMES_PATH}")


# --- Market Resolution Period for Selecting Eligible Markets ---
MARKET_RESOLUTION_START_DATE_STR = "2025-05-09"
MARKET_RESOLUTION_END_DATE_STR = "2025-05-15"

MARKET_RESOLUTION_START_TS = int(dt.datetime.strptime(MARKET_RESOLUTION_START_DATE_STR + " 00:00:00", "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc).timestamp())
MARKET_RESOLUTION_END_TS = int(dt.datetime.strptime(MARKET_RESOLUTION_END_DATE_STR + " 23:59:59", "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc).timestamp())
logger_config.info(f"Eligible markets for backtest: Resolving from {MARKET_RESOLUTION_START_DATE_STR} to {MARKET_RESOLUTION_END_DATE_STR}")

# --- Actual Decision-Making Period (Calendar Days) ---
DECISION_MAKING_START_DATE_STR = "2025-05-09"
DECISION_MAKING_END_DATE_STR = "2025-05-15"

DECISION_MAKING_START_TS = int(dt.datetime.strptime(DECISION_MAKING_START_DATE_STR + " 00:00:00", "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc).timestamp())
DECISION_MAKING_END_TS = int(dt.datetime.strptime(DECISION_MAKING_END_DATE_STR + " 23:59:59", "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc).timestamp())
logger_config.info(f"Decision-making simulation: For decisions occurring from {DECISION_MAKING_START_DATE_STR} to {DECISION_MAKING_END_DATE_STR}")

# --- Trading Parameters ---
PROBABILITY_THRESHOLD_YES = 0.60
PROBABILITY_THRESHOLD_NO = 0.60  
logger_config.info(f"Trading Thresholds: P(Yes) > {PROBABILITY_THRESHOLD_YES} for BUY_YES; P(No) > {PROBABILITY_THRESHOLD_NO} for BUY_NO")

# --- Logging ---
LOG_DIR.mkdir(parents=True, exist_ok=True)

# --- Data Loading ---
LATEST_FEATURES_CSV_GLOB_PATTERN = str(FEATURES_DIR / "kalshi_per_minute_decision_features_*.csv")
logger_config.info(f"Feature files will be searched using pattern: {LATEST_FEATURES_CSV_GLOB_PATTERN}")

# --- Other ---
INITIAL_CAPITAL = 50000 
ONE_BET_PER_KALSHI_MARKET = False 
logger_config.info(f"ONE_BET_PER_KALSHI_MARKET set to: {ONE_BET_PER_KALSHI_MARKET}")