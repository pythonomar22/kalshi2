# /notebooks/backtest/backtest_config.py

from pathlib import Path
import datetime as dt
from datetime import timezone
import logging # Import logging

# --- Initial logging setup for this config file itself ---
# This ensures that if this file is imported and its constants are accessed,
# the logging messages within it (like the date ranges) are shown.
# The main notebook/script might reconfigure logging later.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger_config = logging.getLogger("backtest_config") # Specific logger for this file

# --- Core Paths ---
BASE_PROJECT_DIR = Path("/Users/omarabul-hassan/Desktop/projects/kalshi")
NOTEBOOKS_DIR = BASE_PROJECT_DIR / "notebooks"
FEATURES_DIR = NOTEBOOKS_DIR / "features"
TRAINED_MODELS_BASE_DIR = NOTEBOOKS_DIR / "trained_models"
BACKTEST_DIR = NOTEBOOKS_DIR / "backtest"
LOG_DIR = BACKTEST_DIR / "logs"

# --- Model & Scaler Paths (for per_minute model) ---
MODEL_TYPE = "logreg_per_minute" 
MODEL_DIR = TRAINED_MODELS_BASE_DIR / MODEL_TYPE
MODEL_PATH = MODEL_DIR / "logreg_per_minute_model.joblib"
SCALER_PATH = MODEL_DIR / "logreg_per_minute_scaler.joblib"
FEATURE_NAMES_PATH = MODEL_DIR / "logreg_per_minute_feature_names.json"

# --- Market Resolution Period for Selecting Eligible Markets ---
# Markets must RESOLVE within this window to be considered.
MARKET_RESOLUTION_START_DATE_STR = "2025-05-09"
MARKET_RESOLUTION_END_DATE_STR = "2025-05-15"

MARKET_RESOLUTION_START_TS = int(dt.datetime.strptime(MARKET_RESOLUTION_START_DATE_STR + " 00:00:00", "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc).timestamp())
MARKET_RESOLUTION_END_TS = int(dt.datetime.strptime(MARKET_RESOLUTION_END_DATE_STR + " 23:59:59", "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc).timestamp())
logger_config.info(f"Eligible markets for backtest: Resolving from {MARKET_RESOLUTION_START_DATE_STR} to {MARKET_RESOLUTION_END_DATE_STR}")

# --- Actual Decision-Making Period (Calendar Days) ---
# We will only simulate making decisions if the decision_timestamp_s falls within these calendar days.
DECISION_MAKING_START_DATE_STR = "2025-05-09"
DECISION_MAKING_END_DATE_STR = "2025-05-15"

DECISION_MAKING_START_TS = int(dt.datetime.strptime(DECISION_MAKING_START_DATE_STR + " 00:00:00", "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc).timestamp())
DECISION_MAKING_END_TS = int(dt.datetime.strptime(DECISION_MAKING_END_DATE_STR + " 23:59:59", "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc).timestamp())
logger_config.info(f"Decision-making simulation: For decisions occurring from {DECISION_MAKING_START_DATE_STR} to {DECISION_MAKING_END_DATE_STR}")


# --- Trading Parameters ---
PROBABILITY_THRESHOLD_YES = 0.60 # Example: if P(Yes) > 0.60, consider BUY_YES
PROBABILITY_THRESHOLD_NO = 0.60  # Example: if P(No) > 0.60 (i.e., P(Yes) < 0.40), consider BUY_NO
logger_config.info(f"Trading Thresholds: P(Yes) > {PROBABILITY_THRESHOLD_YES} for BUY_YES; P(No) > {PROBABILITY_THRESHOLD_NO} for BUY_NO")

# --- Bet Sizing & Cost Assumption ---
# (Conceptual for now)

# --- Logging ---
LOG_DIR.mkdir(parents=True, exist_ok=True) # Ensure log directory exists

# --- Data Loading ---
LATEST_FEATURES_CSV_GLOB_PATTERN = str(FEATURES_DIR / "kalshi_per_minute_decision_features_*.csv")
logger_config.info(f"Feature files will be searched using pattern: {LATEST_FEATURES_CSV_GLOB_PATTERN}")

# --- Other ---
INITIAL_CAPITAL = 50000 
ONE_BET_PER_KALSHI_MARKET = False 
logger_config.info(f"ONE_BET_PER_KALSHI_MARKET set to: {ONE_BET_PER_KALSHI_MARKET}")