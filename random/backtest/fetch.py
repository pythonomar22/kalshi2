# fetch_live_session_outcomes.py
import requests
import os
import time
import datetime as dt
from datetime import timezone, timedelta
import base64
import json
from pathlib import Path
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import padding, rsa
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
import logging
import re

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s.%(funcName)s:%(lineno)d - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("FetchLiveOutcomes")

# --- Configuration ---
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / '.env') # Assuming .env is in project root (parent of 'random')

KALSHI_API_KEY_ID = os.getenv("KALSHI_API_KEY_ID")
KALSHI_PRIVATE_KEY_PATH = os.getenv("KALSHI_PRIVATE_KEY_PATH")
KALSHI_BASE_URL = ""

IS_DEMO_MODE = os.getenv("KALSHI_DEMO_MODE", "false").lower() == "true"
if IS_DEMO_MODE:
    logger.info("KALSHI: Running in DEMO mode.")
    KALSHI_BASE_URL = "https://demo-api.kalshi.co"
    KALSHI_DEMO_API_KEY_ID = os.getenv("KALSHI_DEMO_API_KEY_ID")
    KALSHI_DEMO_PRIVATE_KEY_PATH = os.getenv("KALSHI_DEMO_PRIVATE_KEY_PATH")
    if KALSHI_DEMO_API_KEY_ID: KALSHI_API_KEY_ID = KALSHI_DEMO_API_KEY_ID
    if KALSHI_DEMO_PRIVATE_KEY_PATH: KALSHI_PRIVATE_KEY_PATH = KALSHI_DEMO_PRIVATE_KEY_PATH
else:
    logger.info("KALSHI: Running in PRODUCTION mode.")
    KALSHI_BASE_URL = "https://api.elections.kalshi.com" # Or your preferred prod URL

# Directory containing your live Kalshi CSVs
# Assumes this script is run from 'random/' or its path is adjusted.
# If running from 'random/live_backtester', then parent is 'random'
LIVE_KALSHI_CSV_DIR = Path(__file__).resolve().parent.parent / "market_data_logs"
OUTPUT_CSV_FILENAME = "live_sessions_market_outcomes.csv"
OUTPUT_CSV_PATH = Path(__file__).resolve().parent / OUTPUT_CSV_FILENAME # Save in live_backtester dir

API_DELAY_SECONDS = 0.2 # Be respectful to the API

# --- Auth Functions ---
private_key_global = None

def load_private_key_once(file_path: str) -> rsa.RSAPrivateKey | None:
    global private_key_global
    if private_key_global:
        return private_key_global
    if not file_path:
        logger.error("Private key file path is not provided.")
        return None
    try:
        expanded_path = Path(file_path).expanduser().resolve()
        if not expanded_path.exists():
            logger.error(f"Private key file does not exist at resolved path: {expanded_path}")
            return None
        with open(expanded_path, "rb") as key_file:
            private_key_global = serialization.load_pem_private_key(key_file.read(), password=None)
        logger.info(f"Private key loaded successfully from {expanded_path}")
        return private_key_global
    except Exception as e:
        logger.error(f"Error loading private key from {file_path}: {e}")
        return None

def sign_pss_text(private_key: rsa.RSAPrivateKey, text: str) -> str | None:
    if not private_key: return None
    message = text.encode('utf-8')
    try:
        signature = private_key.sign(
            message, padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH),
            hashes.SHA256()
        )
        return base64.b64encode(signature).decode('utf-8')
    except Exception as e:
        logger.error(f"Error during signing: {e}")
        return None

def get_kalshi_auth_headers_for_get(path: str) -> dict | None:
    if not private_key_global:
        logger.error("Global private_key_global not loaded.")
        return None
    if not KALSHI_API_KEY_ID:
        logger.error("Global KALSHI_API_KEY_ID not set.")
        return None
        
    timestamp_ms_str = str(int(time.time() * 1000))
    if not path.startswith('/'): path = '/' + path
    message_to_sign = timestamp_ms_str + "GET" + path
    signature = sign_pss_text(private_key_global, message_to_sign)
    if signature is None: return None
    
    return {
        'accept': 'application/json',
        'KALSHI-ACCESS-KEY': KALSHI_API_KEY_ID,
        'KALSHI-ACCESS-SIGNATURE': signature,
        'KALSHI-ACCESS-TIMESTAMP': timestamp_ms_str
    }

def kalshi_api_get(endpoint_path: str, params: dict = None, timeout: int = 20) -> dict | None:
    if not private_key_global: return None
    if not KALSHI_BASE_URL: return None
    if not endpoint_path.startswith('/'): endpoint_path = '/' + endpoint_path
        
    full_url = f"{KALSHI_BASE_URL}{endpoint_path}"
    auth_headers = get_kalshi_auth_headers_for_get(endpoint_path)
    if not auth_headers: return None

    try:
        response = requests.get(full_url, headers=auth_headers, params=params, timeout=timeout)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as http_err:
        logger.error(f"HTTP error: {http_err} | Status: {response.status_code if 'response' in locals() else 'N/A'} | URL: {full_url} | Response: {response.text if 'response' in locals() else 'N/A'}")
    # ... (other specific request exceptions from your NTM downloader) ...
    except Exception as e:
        logger.error(f"Generic error for {full_url}: {e}")
    return None

def parse_kalshi_market_ticker_basic(ticker_string: str):
    """Extracts strike price from a market ticker string."""
    match = re.search(r"-T(\d+\.?\d*)$", ticker_string)
    if match:
        return float(match.group(1))
    return None

def main():
    logger.info("Starting script to fetch outcomes for live-collected Kalshi markets.")
    if not (KALSHI_API_KEY_ID and KALSHI_PRIVATE_KEY_PATH):
        logger.critical("CRITICAL: Kalshi API credentials not set in .env. Exiting.")
        return
    
    load_private_key_once(KALSHI_PRIVATE_KEY_PATH)
    if not private_key_global:
        logger.critical("Failed to load private key. Exiting.")
        return

    if not LIVE_KALSHI_CSV_DIR.exists():
        logger.error(f"Live Kalshi CSV directory not found: {LIVE_KALSHI_CSV_DIR}")
        return

    market_tickers = set()
    for csv_file in LIVE_KALSHI_CSV_DIR.glob("KXBTCD-*.csv"):
        market_tickers.add(csv_file.stem)
    
    if not market_tickers:
        logger.info("No Kalshi market CSV files found to process.")
        return

    logger.info(f"Found {len(market_tickers)} unique market tickers to fetch outcomes for.")
    
    all_outcomes_data = []
    for ticker in tqdm(sorted(list(market_tickers)), desc="Fetching Market Details"):
        logger.debug(f"Fetching details for {ticker}...")
        api_path = f"/trade-api/v2/markets/{ticker}"
        market_data_response = kalshi_api_get(api_path)
        time.sleep(API_DELAY_SECONDS)

        if market_data_response and "market" in market_data_response:
            market_info = market_data_response["market"]
            strike_price = parse_kalshi_market_ticker_basic(ticker) # Get strike from ticker name as fallback/confirmation
            
            # Ensure 'result' is present, it might be null if market is not yet settled/finalized.
            # Only include markets that have a definitive result.
            if market_info.get("status") in ["settled", "finalized"] and market_info.get("result"):
                all_outcomes_data.append({
                    "market_ticker": market_info.get("ticker"),
                    "result": market_info.get("result"), # 'yes' or 'no'
                    "status": market_info.get("status"),
                    "open_time_iso": market_info.get("open_time"),
                    "close_time_iso": market_info.get("close_time"), # This is the resolution time
                    "strike_price": strike_price if strike_price is not None else market_info.get("strike"), # Prefer parsed
                })
            else:
                logger.warning(f"Market {ticker} is not settled/finalized or has no result. Status: {market_info.get('status')}, Result: {market_info.get('result')}. Skipping.")
        else:
            logger.error(f"Failed to fetch valid data for market: {ticker}")

    if all_outcomes_data:
        df_outcomes = pd.DataFrame(all_outcomes_data)
        df_outcomes.to_csv(OUTPUT_CSV_PATH, index=False)
        logger.info(f"Successfully saved {len(df_outcomes)} market outcomes to: {OUTPUT_CSV_PATH}")
        print(f"\nOutcomes data saved to: {OUTPUT_CSV_PATH.resolve()}")
        print("Sample of outcomes data:")
        print(df_outcomes.head().to_string())
    else:
        logger.info("No settled market outcomes were fetched.")

if __name__ == "__main__":
    main()