# /paper/config.py

from pathlib import Path
import logging
import os
from dotenv import load_dotenv

dotenv_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=dotenv_path)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger_cfg = logging.getLogger("paper_config")

BASE_PROJECT_DIR = Path("/Users/omarabul-hassan/Desktop/projects/kalshi")
PAPER_TRADING_DIR = BASE_PROJECT_DIR / "paper"
LOG_DIR = PAPER_TRADING_DIR / "logs"
TEMP_PREFILL_DATA_DIR = PAPER_TRADING_DIR / "temp_prefill_data"

IS_DEMO_MODE = os.getenv("KALSHI_DEMO_MODE", "false").lower() == "true"

if IS_DEMO_MODE:
    logger_cfg.info("KALSHI: Paper trading in DEMO mode.")
    KALSHI_API_KEY_ID = os.getenv("KALSHI_DEMO_API_KEY_ID")
    KALSHI_PRIVATE_KEY_PATH_STR = os.getenv("KALSHI_DEMO_PRIVATE_KEY_PATH")
    KALSHI_WS_BASE_URL = os.getenv("KALSHI_DEMO_WS_BASE_URL", "wss://demo-api.kalshi.co/trade-api/ws/v2")
    KALSHI_API_BASE_URL = os.getenv("KALSHI_DEMO_API_BASE_URL", "https://demo-api.kalshi.co")
else:
    logger_cfg.info("KALSHI: Paper trading in PRODUCTION mode (with paper money).")
    KALSHI_API_KEY_ID = os.getenv("KALSHI_API_KEY_ID")
    KALSHI_PRIVATE_KEY_PATH_STR = os.getenv("KALSHI_PRIVATE_KEY_PATH")
    KALSHI_WS_BASE_URL = os.getenv("KALSHI_PROD_WS_BASE_URL", "wss://api.elections.kalshi.com/trade-api/ws/v2")
    KALSHI_API_BASE_URL = os.getenv("KALSHI_PROD_API_BASE_URL", "https://api.elections.kalshi.com")

if not KALSHI_API_KEY_ID or not KALSHI_PRIVATE_KEY_PATH_STR:
    logger_cfg.critical("Kalshi API Key ID or Private Key Path not found in .env or config.")
    raise ValueError("Missing Kalshi API credentials")
_resolved_key_path = Path(KALSHI_PRIVATE_KEY_PATH_STR).expanduser().resolve()
if not _resolved_key_path.exists():
    logger_cfg.critical(f"Kalshi private key file not found at: {_resolved_key_path}")
    raise FileNotFoundError(f"Kalshi private key file not found at: {_resolved_key_path}")
KALSHI_PRIVATE_KEY_PATH = str(_resolved_key_path)

# --- Binance Configuration ---
BINANCE_WS_ENDPOINT = "wss://data-stream.binance.vision/stream" 
BINANCE_STREAM_SUBSCRIPTIONS = ["btcusdt@kline_1m"] 

# *** UPDATED: Using HashKey API for pre-fill due to Binance geo-restrictions ***
# BINANCE_REST_API_KLINES_URL = "https://api.binance.com/api/v3/klines" # Geo-restricted
HASHKEY_API_KLINES_URL = "https://api-glb.hashkey.com/quote/v1/klines" # New prefill source

BINANCE_PREFILL_SYMBOL = "BTCUSDT" # Symbol for HashKey (check if it uses BTCUSDT or BTCUSDT-PERPETUAL)
                                   # The example uses BTCUSDT-PERPETUAL, but Kalshi refers to spot.
                                   # For consistency, let's try to find a SPOT BTCUSDT on HashKey if available,
                                   # or be aware if we're using a perpetuals feed.
                                   # The sample response has "ETHUSDT", "BTCUSDT-PERPETUAL".
                                   # Let's assume "BTCUSDT" is the spot symbol on HashKey if available.
                                   # If not, we might have to use BTCUSDT-PERPETUAL or find another source.
                                   # For now, we'll try with BTCUSDT and see.
HASHKEY_PREFILL_SYMBOL = "BTCUSDT" # Adjust if Hashkey uses a different spot BTC ticker like "BTC/USDT" or just "BTCUSDT"
                                   # The API example showed ETHUSDT and BTCUSDT-PERPETUAL.
                                   # Let's assume we can try "BTCUSDT" for spot.
HASHKEY_PREFILL_INTERVAL = "1min" # Hashkey uses "1min" not "1m"

# --- Model & Scaler Paths ---
MODEL_TYPE_SUFFIX = "no_vol_oi" 
MODEL_TYPE_BASE = "logreg_per_minute"
HISTORICAL_TRAINED_MODELS_BASE_DIR = BASE_PROJECT_DIR / "notebooks" / "trained_models"
MODEL_DIR_NAME = f"{MODEL_TYPE_BASE}_{MODEL_TYPE_SUFFIX}" if MODEL_TYPE_SUFFIX else MODEL_TYPE_BASE
MODEL_DIR = HISTORICAL_TRAINED_MODELS_BASE_DIR / MODEL_DIR_NAME
MODEL_PATH = MODEL_DIR / f"{MODEL_DIR_NAME}_model.joblib"
SCALER_PATH = MODEL_DIR / f"{MODEL_DIR_NAME}_scaler.joblib"
FEATURE_NAMES_PATH = MODEL_DIR / f"{MODEL_DIR_NAME}_feature_names.json"
logger_cfg.info(f"Using model from directory: {MODEL_DIR}")

# --- Trading Parameters ---
PROBABILITY_THRESHOLD_YES = 0.70 
PROBABILITY_THRESHOLD_NO = 0.70  
ONE_BET_PER_KALSHI_MARKET = False 

# --- Kelly Criterion Sizing Parameters ---
USE_KELLY_CRITERION = True
INITIAL_PAPER_CAPITAL_CENTS = 50000 
KELLY_FRACTION = 0.10 
MAX_PCT_CAPITAL_PER_TRADE = 0.05 
MIN_CONTRACTS_TO_TRADE = 1
MAX_CONTRACTS_TO_TRADE = 100 
logger_cfg.info(f"USE_KELLY_CRITERION for paper trading: {USE_KELLY_CRITERION}")

# --- Feature Engineering Parameters ---
MIN_MINUTES_BEFORE_RESOLUTION_FOR_DECISION = 1 
LAG_WINDOWS_MINUTES = [1, 3, 5, 10, 15, 30]
ROLLING_WINDOWS_MINUTES = [5, 15, 30]

# --- Kalshi Market Discovery ---
TARGET_EVENT_SERIES_TICKER = "KXBTCD" 
MARKET_DISCOVERY_INTERVAL_SECONDS = 60 * 5 
MAX_MARKETS_TO_MONITOR = 10 
NUM_NTM_MARKETS_PER_EVENT = 5 
KALSHI_STRIKE_INCREMENT = 250 

# --- Bot Operation Parameters ---
TRADING_LOOP_INTERVAL_SECONDS = 60 

LOG_DIR.mkdir(parents=True, exist_ok=True)
# TEMP_PREFILL_DATA_DIR.mkdir(parents=True, exist_ok=True) # No longer needed if not downloading zips
logger_cfg.info(f"Target Kalshi event series for discovery: {TARGET_EVENT_SERIES_TICKER}")
logger_cfg.info(f"Will select ~{NUM_NTM_MARKETS_PER_EVENT} NTM markets per active event.")