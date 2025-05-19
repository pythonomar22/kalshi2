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

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
load_dotenv()
KALSHI_API_KEY_ID = os.getenv("KALSHI_API_KEY_ID")
KALSHI_PRIVATE_KEY_PATH = os.getenv("KALSHI_PRIVATE_KEY_PATH")

IS_DEMO_MODE = os.getenv("KALSHI_DEMO_MODE", "false").lower() == "true"
if IS_DEMO_MODE: # ... (same demo mode handling as before) ...
    logger.info("KALSHI: Running in DEMO mode.")
    KALSHI_BASE_URL = "https://demo-api.kalshi.co"
    KALSHI_DEMO_API_KEY_ID = os.getenv("KALSHI_DEMO_API_KEY_ID", KALSHI_API_KEY_ID)
    KALSHI_DEMO_PRIVATE_KEY_PATH = os.getenv("KALSHI_DEMO_PRIVATE_KEY_PATH", KALSHI_PRIVATE_KEY_PATH)
    if KALSHI_DEMO_API_KEY_ID != KALSHI_API_KEY_ID : KALSHI_API_KEY_ID = KALSHI_DEMO_API_KEY_ID
    if KALSHI_DEMO_PRIVATE_KEY_PATH != KALSHI_PRIVATE_KEY_PATH : KALSHI_PRIVATE_KEY_PATH = KALSHI_DEMO_PRIVATE_KEY_PATH
else:
    logger.info("KALSHI: Running in PRODUCTION mode.")
    KALSHI_BASE_URL = "https://api.elections.kalshi.com"

# --- Target Series and Date ---
TARGET_SERIES_TICKER = "KXBTCD" # Bitcoin Series
TARGET_DATE_STR = "2025-05-15" # The date for which we want hourly markets (YYYY-MM-DD)

# Define common Bitcoin strike thresholds you are interested in (these are the values after 'T')
# Example: For a $102,500 strike, the threshold is 102499.99
# You might need to adjust these based on what Kalshi typically offers or get them from an API endpoint.
COMMON_BTC_STRIKE_THRESHOLDS_STR = [
    "99499.99",   # For $99,500
    "99999.99",   # For $100,000
    "100499.99",  # For $100,500
    "100999.99",  # For $101,000
    "101499.99",  # For $101,500
    "101999.99",  # For $102,000
    "102499.99",  # For $102,500
    "102999.99",  # For $103,000
    "103499.99",  # For $103,500
    "103999.99",  # For $104,000
    "104499.99",  # For $104,500
    "104999.99",  # For $105,000
    "105499.99"   # For $105,500
]

# --- Data Fetching Parameters ---
PERIOD_INTERVAL_MINUTES = 1
MAX_PERIODS_PER_REQUEST = 4900

# --- Output Configuration ---
OUTPUT_DIR = Path(f"./kalshi_historical_data/{TARGET_SERIES_TICKER}_{TARGET_DATE_STR.replace('-', '')}")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Auth and Fetching Functions (remain the same as your last historical.py) ---
def load_private_key(file_path: str) -> rsa.RSAPrivateKey | None:
    # ... (same as before) ...
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
    # ... (same as before) ...
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
    # ... (same as before) ...
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

def fetch_and_parse_kalshi_candlesticks(
    series_ticker: str, market_ticker: str, start_dt_utc: dt.datetime, end_dt_utc: dt.datetime,
    period_minutes: int, private_key: rsa.RSAPrivateKey, key_id: str
) -> pd.DataFrame | None:
    # ... (This function remains largely the same as your last version) ...
    # ... (Ensure the logging of the first candlestick structure is still there) ...
    all_candlesticks_processed = []
    start_ts_s = int(start_dt_utc.timestamp())
    end_ts_s = int(end_dt_utc.timestamp())
    current_start_ts = start_ts_s

    logger.info(f"Fetching candlesticks for {market_ticker}")
    logger.info(f"Target Period: {start_dt_utc.strftime('%Y-%m-%d %H:%M:%S %Z')} to {end_dt_utc.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    
    total_duration_seconds = max(0, end_ts_s - start_ts_s)
    total_expected_intervals = (total_duration_seconds // (period_minutes * 60)) +1 if total_duration_seconds >=0 else 0

    if 'first_candle_logged_this_run' not in fetch_and_parse_kalshi_candlesticks.__dict__:
        fetch_and_parse_kalshi_candlesticks.first_candle_logged_this_run = False

    with tqdm(total=total_expected_intervals, desc=f"Fetching {market_ticker}", leave=False) as pbar:
        while current_start_ts <= end_ts_s:
            chunk_end_ts = min(end_ts_s, current_start_ts + (MAX_PERIODS_PER_REQUEST -1) * period_minutes * 60)
            api_path = f"/trade-api/v2/series/{series_ticker}/markets/{market_ticker}/candlesticks"
            params = {"start_ts": current_start_ts, "end_ts": chunk_end_ts, "period_interval": period_minutes}
            # logger.debug(f"Requesting chunk: start_ts={current_start_ts}, end_ts={chunk_end_ts}")
            headers = get_kalshi_auth_headers("GET", api_path, private_key, key_id)
            if headers is None: return None
            try:
                response = requests.get(f"{KALSHI_BASE_URL}{api_path}", headers=headers, params=params)
                response.raise_for_status()
                api_response_data = response.json()
                
                if not fetch_and_parse_kalshi_candlesticks.first_candle_logged_this_run and api_response_data.get("candlesticks"):
                    logger.info("Structure of the first candlestick received from API (this market):")
                    logger.info(json.dumps(api_response_data["candlesticks"][0], indent=2))
                    fetch_and_parse_kalshi_candlesticks.first_candle_logged_this_run = True

                candlesticks_from_api = api_response_data.get("candlesticks", [])
                if candlesticks_from_api:
                    for candle_data in candlesticks_from_api:
                        ts = candle_data.get("end_period_ts")
                        if ts is None or ts < current_start_ts : continue
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
                            "volume": candle_data.get("volume"), "open_interest": candle_data.get("open_interest")
                        })
                    last_candle_ts_in_chunk = candlesticks_from_api[-1]["end_period_ts"]
                    current_start_ts = last_candle_ts_in_chunk + (period_minutes * 60)
                    pbar.update(len(candlesticks_from_api))
                else:
                    # logger.debug(f"No candlesticks in chunk ending {chunk_end_ts}. Advancing.")
                    current_start_ts = chunk_end_ts + (period_minutes * 60)
                    pbar.update(max(1, (chunk_end_ts - params["start_ts"]) // (period_minutes * 60) +1 ) if chunk_end_ts >= params["start_ts"] else 1)


                time.sleep(0.6)
            except requests.exceptions.HTTPError as e:
                logger.error(f"HTTP Error for {market_ticker}: {e.response.status_code} - {e.response.text if e.response else str(e)}")
                if e.response and e.response.status_code == 404:
                    logger.warning(f"Market {market_ticker} likely not found or no data for this period.")
                break 
            except Exception as e:
                logger.exception(f"Unexpected error for {market_ticker}: {e}")
                break
    if pbar.n < pbar.total and total_expected_intervals > 0: pbar.update(pbar.total - pbar.n)
    pbar.close()
    if not all_candlesticks_processed: return None
    df = pd.DataFrame(all_candlesticks_processed)
    df['timestamp_utc_end_of_period'] = pd.to_datetime(df['timestamp_s'], unit='s', utc=True)
    df.set_index('timestamp_utc_end_of_period', inplace=True)
    df.drop(columns=['timestamp_s'], inplace=True)
    df = df[~df.index.duplicated(keep='first')].sort_index()
    numeric_cols = [col for col in df.columns if 'cents' in col or 'volume' in col or 'interest' in col]
    for col in numeric_cols: df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

# --- Main Loop to Iterate Through Hours and Strikes ---
if __name__ == "__main__":
    if not KALSHI_API_KEY_ID or not KALSHI_PRIVATE_KEY_PATH: # ... (same credential check) ...
        logger.critical("CRITICAL: KALSHI_API_KEY_ID or KALSHI_PRIVATE_KEY_PATH not found in .env file.")
        exit()
    private_key = load_private_key(KALSHI_PRIVATE_KEY_PATH) # ... (same credential check) ...
    if private_key is None:
        logger.critical(f"CRITICAL: Could not load private key from {KALSHI_PRIVATE_KEY_PATH}.")
        exit()

    # Parse the target date
    target_date_obj = dt.datetime.strptime(TARGET_DATE_STR, "%Y-%m-%d").replace(tzinfo=timezone.utc) # Treat as UTC day start

    # Iterate through each hour of the target day (00:00 EDT to 23:00 EDT)
    # EDT is UTC-4. So 00:00 EDT is 04:00 UTC. 23:00 EDT is 03:00 UTC the *next* day.
    for hour_edt in range(24): # 0 to 23 for EDT hours
        # Calculate close time for this hour's market
        # Market closing at `hour_edt` on `target_date_obj` (EDT)
        market_close_dt_naive_edt = target_date_obj.replace(hour=hour_edt, minute=0, second=0, microsecond=0)
        
        # Convert EDT naive time to UTC
        # We need to handle the date change for early morning UTC hours
        # Example: May 15, 00:00 EDT = May 15, 04:00 UTC
        # Example: May 15, 23:00 EDT = May 16, 03:00 UTC
        offset_edt = dt.timedelta(hours=-4)
        market_close_dt_utc = (market_close_dt_naive_edt - offset_edt).replace(tzinfo=timezone.utc)
        
        # Ensure we are constructing the correct UTC day for the close time
        # If target_date_str is "2025-05-15"
        # For hour_edt = 0 (12 AM EDT), close_dt_utc should be 2025-05-15 04:00:00 UTC
        # For hour_edt = 23 (11 PM EDT), close_dt_utc should be 2025-05-16 03:00:00 UTC

        # Kalshi event ticker part needs YYMMDDHH format for the *closing time in EDT*
        # Date part (YYMMDD) for event ticker is based on the EDT closing day.
        event_date_str_yy = market_close_dt_naive_edt.strftime("%y") # e.g., 25
        event_date_str_mon = market_close_dt_naive_edt.strftime("%b").upper() # e.g., MAY
        event_date_str_dd = market_close_dt_naive_edt.strftime("%d") # e.g., 15
        event_hour_str_hh = market_close_dt_naive_edt.strftime("%H") # e.g., 00, 01, ..., 23

        event_ticker_time_part = f"{event_date_str_yy}{event_date_str_mon}{event_date_str_dd}{event_hour_str_hh}"
        
        market_open_dt_utc = market_close_dt_utc - dt.timedelta(hours=1)

        logger.info(f"\nProcessing Event for {TARGET_SERIES_TICKER} closing {hour_edt:02d}:00 EDT on {target_date_obj.strftime('%Y-%m-%d')}")
        logger.info(f"Event Ticker Time Part: {event_ticker_time_part}")
        logger.info(f"Market Window (UTC): {market_open_dt_utc} to {market_close_dt_utc}")

        fetch_and_parse_kalshi_candlesticks.first_candle_logged_this_run = False # Reset for each event group

        for strike_threshold_str in COMMON_BTC_STRIKE_THRESHOLDS_STR:
            market_ticker = f"{TARGET_SERIES_TICKER}-{event_ticker_time_part}-T{strike_threshold_str}"
            
            logger.info(f"--- Attempting to fetch data for Market: {market_ticker} ---")

            market_history_df = fetch_and_parse_kalshi_candlesticks(
                series_ticker=TARGET_SERIES_TICKER,
                market_ticker=market_ticker,
                start_dt_utc=market_open_dt_utc,
                end_dt_utc=market_close_dt_utc,
                period_minutes=PERIOD_INTERVAL_MINUTES,
                private_key=private_key,
                key_id=KALSHI_API_KEY_ID
            )

            if market_history_df is not None and not market_history_df.empty:
                base_output_filename = f"{market_ticker}_historical_detailed_{PERIOD_INTERVAL_MINUTES}min"
                output_filepath_csv = OUTPUT_DIR / f"{base_output_filename}.csv"
                # output_filepath_parquet = OUTPUT_DIR / f"{base_output_filename}.parquet" # Optional

                try:
                    market_history_df.to_csv(output_filepath_csv)
                    logger.info(f"Saved: {output_filepath_csv} (Shape: {market_history_df.shape})")
                except Exception as e: logger.error(f"Error saving CSV for {market_ticker}: {e}")
                
                # logger.info(f"Sample for {market_ticker}:\n{market_history_df.head()}")
                # if 'volume' in market_history_df.columns:
                #     total_vol = market_history_df['volume'].sum()
                #     logger.info(f"Total volume for {market_ticker}: {total_vol}")
            else:
                logger.warning(f"No data fetched for {market_ticker}.")
            
            time.sleep(1) # Pause between fetching different strikes within the same hour

        logger.info(f"--- Finished processing hour {hour_edt:02d}:00 EDT ---")
        time.sleep(5) # Pause between fetching different hours

    logger.info("--- All Historical Data Fetching Complete ---")