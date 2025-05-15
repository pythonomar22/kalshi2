import requests
import os
import time
import datetime as dt # Use dt to avoid conflict with datetime module name
from datetime import timezone # Explicitly import timezone
import base64
import json
from pathlib import Path
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import padding, rsa
import pandas as pd
from tqdm import tqdm # For progress bar
from dotenv import load_dotenv
import logging

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
load_dotenv()
KALSHI_API_KEY_ID = os.getenv("KALSHI_API_KEY_ID")
KALSHI_PRIVATE_KEY_PATH = os.getenv("KALSHI_PRIVATE_KEY_PATH")

IS_DEMO_MODE = os.getenv("KALSHI_DEMO_MODE", "false").lower() == "true"
if IS_DEMO_MODE:
    logger.info("KALSHI: Running in DEMO mode.")
    KALSHI_BASE_URL = "https://demo-api.kalshi.co"
    KALSHI_DEMO_API_KEY_ID = os.getenv("KALSHI_DEMO_API_KEY_ID", KALSHI_API_KEY_ID)
    KALSHI_DEMO_PRIVATE_KEY_PATH = os.getenv("KALSHI_DEMO_PRIVATE_KEY_PATH", KALSHI_PRIVATE_KEY_PATH)
    if KALSHI_DEMO_API_KEY_ID != KALSHI_API_KEY_ID : KALSHI_API_KEY_ID = KALSHI_DEMO_API_KEY_ID
    if KALSHI_DEMO_PRIVATE_KEY_PATH != KALSHI_PRIVATE_KEY_PATH : KALSHI_PRIVATE_KEY_PATH = KALSHI_DEMO_PRIVATE_KEY_PATH
else:
    logger.info("KALSHI: Running in PRODUCTION mode.")
    KALSHI_BASE_URL = "https://api.elections.kalshi.com"


# --- Target Historical Market ---
TARGET_SERIES_TICKER = "KXBTCD" # Bitcoin Series
TARGET_MARKET_TICKER = "KXBTCD-25MAY1500-T102999.99" # BTC > $103,000 @ 12AM EDT May 15

# Determine market open/close times in UTC
# Market closes: May 15, 2025 · 12:00 AM EDT (EDT is UTC-4)
# 12:00 AM EDT May 15 = 00:00 EDT May 15 = 04:00 UTC May 15
MARKET_CLOSE_DT_UTC = dt.datetime(2025, 5, 15, 4, 0, 0, tzinfo=timezone.utc)
# Market opens 1 hour before close (standard for these hourly contracts)
MARKET_OPEN_DT_UTC = MARKET_CLOSE_DT_UTC - dt.timedelta(hours=1)

# --- Data Fetching Parameters ---
PERIOD_INTERVAL_MINUTES = 1 # 1-minute candlesticks
MAX_PERIODS_PER_REQUEST = 4900 # Kalshi's limit

# --- Output Configuration ---
OUTPUT_DIR = Path("./kalshi_historical_data")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Auth Functions ---
def load_private_key(file_path: str) -> rsa.RSAPrivateKey | None:
    try:
        with open(file_path, "rb") as key_file:
            private_key = serialization.load_pem_private_key(key_file.read(), password=None)
        return private_key
    except FileNotFoundError:
        logger.error(f"Private key file not found: {file_path}")
        return None
    except Exception as e:
        logger.error(f"Error loading private key from {file_path}: {e}")
        return None

def sign_pss_text(private_key: rsa.RSAPrivateKey, text: str) -> str | None:
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

def get_kalshi_auth_headers(method: str, path: str, private_key: rsa.RSAPrivateKey, key_id: str) -> dict | None:
    timestamp_ms_str = str(int(time.time() * 1000))
    message_to_sign = timestamp_ms_str + method.upper() + path
    signature = sign_pss_text(private_key, message_to_sign)
    if signature is None: return None
    return {
        'accept': 'application/json',
        'KALSHI-ACCESS-KEY': key_id,
        'KALSHI-ACCESS-SIGNATURE': signature,
        'KALSHI-ACCESS-TIMESTAMP': timestamp_ms_str
    }

# --- Data Fetching Logic ---
def fetch_and_parse_kalshi_candlesticks(
    series_ticker: str,
    market_ticker: str,
    start_dt_utc: dt.datetime,
    end_dt_utc: dt.datetime,
    period_minutes: int,
    private_key: rsa.RSAPrivateKey,
    key_id: str
) -> pd.DataFrame | None:

    all_candlesticks_processed = []
    start_ts_s = int(start_dt_utc.timestamp())
    # For Kalshi, end_ts for candlesticks is inclusive of the period it represents
    # If market closes at 04:00:00, the last candle is for 03:59:00-03:59:59, ending at 04:00:00
    # So we want to query up to and including the candle that ends at MARKET_CLOSE_DT_UTC
    end_ts_s = int(end_dt_utc.timestamp())
    current_start_ts = start_ts_s

    logger.info(f"Fetching candlesticks for {market_ticker}")
    logger.info(f"Target Period: {start_dt_utc.strftime('%Y-%m-%d %H:%M:%S %Z')} to {end_dt_utc.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    logger.info(f"Querying Timestamps (epoch seconds): start_ts={start_ts_s}, end_ts={end_ts_s}")


    total_duration_seconds = max(0, end_ts_s - start_ts_s)
    # Number of intervals is (total duration / interval duration). Add 1 if start_ts exactly aligns with a period start.
    total_expected_intervals = (total_duration_seconds // (period_minutes * 60)) +1 if total_duration_seconds >=0 else 0


    # Ensure we log the first candlestick structure only once per script run
    if 'first_candle_logged_this_run' not in fetch_and_parse_kalshi_candlesticks.__dict__:
        fetch_and_parse_kalshi_candlesticks.first_candle_logged_this_run = False


    with tqdm(total=total_expected_intervals, desc=f"Fetching {market_ticker}") as pbar:
        while current_start_ts <= end_ts_s: # Use <= for end_ts to include the last candle
            chunk_end_ts = min(end_ts_s, current_start_ts + (MAX_PERIODS_PER_REQUEST -1) * period_minutes * 60)
            
            api_path = f"/trade-api/v2/series/{series_ticker}/markets/{market_ticker}/candlesticks"
            params = {
                "start_ts": current_start_ts, # Start of the window to look for candles
                "end_ts": chunk_end_ts,       # End of the window to look for candles
                "period_interval": period_minutes
            }
            logger.debug(f"Requesting chunk: start_ts={current_start_ts}, end_ts={chunk_end_ts}")

            headers = get_kalshi_auth_headers("GET", api_path, private_key, key_id)
            if headers is None:
                logger.error("Failed to generate authentication headers.")
                return None

            try:
                response = requests.get(f"{KALSHI_BASE_URL}{api_path}", headers=headers, params=params)
                response.raise_for_status()
                api_response_data = response.json()
                
                if not fetch_and_parse_kalshi_candlesticks.first_candle_logged_this_run and api_response_data.get("candlesticks"):
                    logger.info("Structure of the first candlestick received from API for this run:")
                    logger.info(json.dumps(api_response_data["candlesticks"][0], indent=2))
                    fetch_and_parse_kalshi_candlesticks.first_candle_logged_this_run = True

                candlesticks_from_api = api_response_data.get("candlesticks", [])

                if candlesticks_from_api:
                    for candle_data in candlesticks_from_api:
                        ts = candle_data.get("end_period_ts")
                        if ts is None or ts < current_start_ts : continue # Skip if no timestamp or before our query window start

                        trade_price_info = candle_data.get("price", {})
                        yes_bid_info = candle_data.get("yes_bid", {})
                        yes_ask_info = candle_data.get("yes_ask", {})

                        all_candlesticks_processed.append({
                            "timestamp_s": ts,
                            "trade_open_cents": trade_price_info.get("open"), "trade_high_cents": trade_price_info.get("high"),
                            "trade_low_cents": trade_price_info.get("low"), "trade_close_cents": trade_price_info.get("close"),
                            "yes_bid_open_cents": yes_bid_info.get("open"), "yes_bid_high_cents": yes_bid_info.get("high"),
                            "yes_bid_low_cents": yes_bid_info.get("low"), "yes_bid_close_cents": yes_bid_info.get("close"),
                            "yes_ask_open_cents": yes_ask_info.get("open"), "yes_ask_high_cents": yes_ask_info.get("high"),
                            "yes_ask_low_cents": yes_ask_info.get("low"), "yes_ask_close_cents": yes_ask_info.get("close"),
                            "volume": candle_data.get("volume"),
                            "open_interest": candle_data.get("open_interest")
                        })
                    
                    # Advance based on the timestamp of the last candle received in this chunk
                    last_candle_ts_in_chunk = candlesticks_from_api[-1]["end_period_ts"]
                    current_start_ts = last_candle_ts_in_chunk + (period_minutes * 60) # Next period to query
                    pbar.update(len(candlesticks_from_api))
                else: # No candlesticks in this specific chunk, advance current_start_ts past chunk_end_ts
                    logger.debug(f"No candlesticks returned for chunk ending {chunk_end_ts}. Advancing query window.")
                    current_start_ts = chunk_end_ts + (period_minutes * 60)
                    # If no data, still advance pbar by expected number in this theoretical chunk if it's small
                    if (chunk_end_ts - params["start_ts"]) // (period_minutes * 60) < 10: # Avoid large jumps on empty initial calls
                         pbar.update((chunk_end_ts - params["start_ts"]) // (period_minutes * 60) +1)


                time.sleep(0.6)

            except requests.exceptions.HTTPError as e:
                logger.error(f"HTTP Error for {market_ticker}: {e.response.status_code} - {e.response.text}")
                break 
            except Exception as e:
                logger.exception(f"Unexpected error fetching data for {market_ticker}: {e}")
                break
    
    if pbar.n < pbar.total and total_expected_intervals > 0:
        pbar.update(pbar.total - pbar.n)
    pbar.close()

    if not all_candlesticks_processed:
        logger.warning(f"No candlestick data successfully processed for {market_ticker}.")
        return None

    df = pd.DataFrame(all_candlesticks_processed)
    # Important: Kalshi's end_period_ts is the *end* of the candle. 
    # For analysis, it's often better to associate data with the *start* of the candle period.
    # However, for consistency with how Kalshi returns it, we'll use end_period_ts for now.
    df['timestamp_utc_end_of_period'] = pd.to_datetime(df['timestamp_s'], unit='s', utc=True)
    df.set_index('timestamp_utc_end_of_period', inplace=True)
    df.drop(columns=['timestamp_s'], inplace=True)
    df = df[~df.index.duplicated(keep='first')] # Remove duplicate timestamps if any (should not happen with correct paging)
    df.sort_index(inplace=True)
    
    numeric_cols = [col for col in df.columns if 'cents' in col or 'volume' in col or 'interest' in col]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

if __name__ == "__main__":
    if not KALSHI_API_KEY_ID or not KALSHI_PRIVATE_KEY_PATH:
        logger.critical("CRITICAL: KALSHI_API_KEY_ID or KALSHI_PRIVATE_KEY_PATH not found in .env file.")
        exit()
    
    private_key = load_private_key(KALSHI_PRIVATE_KEY_PATH)
    if private_key is None:
        logger.critical(f"CRITICAL: Could not load private key from {KALSHI_PRIVATE_KEY_PATH}.")
        exit()

    logger.info(f"--- Fetching Historical Data for Market: {TARGET_MARKET_TICKER} ---")
    
    market_history_df = fetch_and_parse_kalshi_candlesticks(
        series_ticker=TARGET_SERIES_TICKER,
        market_ticker=TARGET_MARKET_TICKER,
        start_dt_utc=MARKET_OPEN_DT_UTC,
        end_dt_utc=MARKET_CLOSE_DT_UTC,
        period_minutes=PERIOD_INTERVAL_MINUTES,
        private_key=private_key,
        key_id=KALSHI_API_KEY_ID
    )

    if market_history_df is not None and not market_history_df.empty:
        base_output_filename = f"{TARGET_MARKET_TICKER}_historical_detailed_{PERIOD_INTERVAL_MINUTES}min"
        output_filepath_parquet = OUTPUT_DIR / f"{base_output_filename}.parquet"
        output_filepath_csv = OUTPUT_DIR / f"{base_output_filename}.csv"

        try:
            market_history_df.to_parquet(output_filepath_parquet)
            logger.info(f"Successfully saved detailed market history to Parquet: {output_filepath_parquet} (Shape: {market_history_df.shape})")
        except Exception as e: logger.error(f"Error saving data to Parquet: {e}")

        try:
            market_history_df.to_csv(output_filepath_csv)
            logger.info(f"Successfully saved detailed market history to CSV: {output_filepath_csv} (Shape: {market_history_df.shape})")
        except Exception as e: logger.error(f"Error saving data to CSV: {e}")

        logger.info("\nSample of fetched data:")
        print(market_history_df.head())
        logger.info("\nData types:")
        print(market_history_df.dtypes)
        logger.info("\nDescription of numeric data (check for NaNs, typical values):")
        # Only describe columns that are likely to be fully numeric
        desc_cols = [c for c in ['yes_bid_close_cents', 'yes_ask_close_cents', 'trade_close_cents', 'volume'] if c in market_history_df.columns]
        if desc_cols:
            print(market_history_df[desc_cols].describe())
        else:
            print("No standard numeric columns found to describe.")

    else:
        logger.warning(f"Failed to fetch or no data available for {TARGET_MARKET_TICKER} in the specified range.")

    logger.info("--- Historical Data Fetching Complete ---")