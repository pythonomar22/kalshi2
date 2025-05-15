# download_kalshi_market_history_multiple.py
import os
import requests
import time
import datetime
import base64
import json
from pathlib import Path
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.exceptions import InvalidSignature
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv

# --- Configuration ---
load_dotenv() # Load environment variables from .env file

# Get credentials from environment variables
KALSHI_API_KEY_ID = os.getenv("KALSHI_API_KEY_ID")
KALSHI_PRIVATE_KEY_PATH = os.getenv("KALSHI_PRIVATE_KEY_PATH")

# Kalshi API details
KALSHI_BASE_URL = "https://api.elections.kalshi.com" # For production
# KALSHI_BASE_URL = "https://demo-api.kalshi.co" # For demo environment

# Common series ticker for these markets
TARGET_SERIES_TICKER = "KXETHD"
COMMON_DATE_STR = "2025-05-13" # All markets are for this date
COMMON_THRESHOLD_PART = "T2649.99" # Common threshold for these markets

# --- Define the 5 target markets ---
# Each market trades for 1 hour before its closing time.
# The hour in the ticker (e.g., '18' in KXETHD-25MAY1318) is the *closing hour in EDT (24h format)*.
MARKETS_TO_FETCH = [
    {
        "closing_hour_edt": 17, # 6 PM EDT
        "description": "ETH price at 5 PM EDT, $2650 or above"
    },
    {
        "closing_hour_edt": 18, # 6 PM EDT
        "description": "ETH price at 6 PM EDT, $2650 or above"
    },
    {
        "closing_hour_edt": 19, # 7 PM EDT
        "description": "ETH price at 7 PM EDT, $2650 or above"
    },
    {
        "closing_hour_edt": 20, # 8 PM EDT
        "description": "ETH price at 8 PM EDT, $2650 or above"
    },
    {
        "closing_hour_edt": 21, # 9 PM EDT
        "description": "ETH price at 9 PM EDT, $2650 or above"
    },
    {
        "closing_hour_edt": 22, # 10 PM EDT
        "description": "ETH price at 10 PM EDT, $2650 or above"
    },
    {
        "closing_hour_edt": 23, # 11 PM EDT
        "description": "ETH price at 11 PM EDT, $2650 or above"
    }
]

# --- Data Fetching Parameters ---
PERIOD_INTERVAL_MINUTES = 1 # 1-minute candlesticks
MAX_PERIODS_PER_REQUEST = 4900

# --- Output Configuration ---
OUTPUT_DIR = Path("./kalshi_market_data_hourly_eth")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Helper Functions for Kalshi API Auth (same as before) ---
def load_private_key(file_path: str) -> rsa.RSAPrivateKey | None:
    try:
        with open(file_path, "rb") as key_file:
            private_key = serialization.load_pem_private_key(
                key_file.read(),
                password=None,
            )
        return private_key
    except Exception as e:
        print(f"Error loading private key from {file_path}: {e}")
        return None

def sign_pss_text(private_key: rsa.RSAPrivateKey, text: str) -> str | None:
    message = text.encode('utf-8')
    try:
        signature = private_key.sign(
            message,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.DIGEST_LENGTH
            ),
            hashes.SHA256()
        )
        return base64.b64encode(signature).decode('utf-8')
    except InvalidSignature as e:
        print(f"RSA sign PSS failed: {e}")
        return None
    except Exception as e:
        print(f"Error during signing: {e}")
        return None

def get_kalshi_auth_headers(method: str, path: str, private_key: rsa.RSAPrivateKey, key_id: str) -> dict | None:
    timestamp_ms_str = str(int(time.time() * 1000))
    message_to_sign = timestamp_ms_str + method.upper() + path

    signature = sign_pss_text(private_key, message_to_sign)
    if signature is None:
        return None

    return {
        'accept': 'application/json',
        'KALSHI-ACCESS-KEY': key_id,
        'KALSHI-ACCESS-SIGNATURE': signature,
        'KALSHI-ACCESS-TIMESTAMP': timestamp_ms_str
    }

# --- Main Data Fetching Logic (same as before) ---
def fetch_kalshi_candlesticks(
    series_ticker: str,
    market_ticker: str,
    start_ts_s: int,
    end_ts_s: int,
    period_minutes: int,
    private_key: rsa.RSAPrivateKey,
    key_id: str
) -> pd.DataFrame | None:

    all_candlesticks_processed = []
    current_start_ts = start_ts_s

    print(f"Fetching candlesticks for {market_ticker}")
    print(f"Period: {datetime.datetime.fromtimestamp(start_ts_s, tz=datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S %Z')} to {datetime.datetime.fromtimestamp(end_ts_s, tz=datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S %Z')}")

    total_duration_seconds = max(0, end_ts_s - start_ts_s)
    total_expected_intervals = total_duration_seconds // (period_minutes * 60) if period_minutes > 0 else 0

    with tqdm(total=total_expected_intervals, desc=f"Fetching {market_ticker}") as pbar:
        while current_start_ts < end_ts_s:
            chunk_end_ts = min(end_ts_s, current_start_ts + (MAX_PERIODS_PER_REQUEST - 1) * period_minutes * 60)
            if chunk_end_ts <= current_start_ts :
                break

            path = f"/trade-api/v2/series/{series_ticker}/markets/{market_ticker}/candlesticks"
            params = {
                "start_ts": current_start_ts,
                "end_ts": chunk_end_ts,
                "period_interval": period_minutes
            }

            headers = get_kalshi_auth_headers("GET", path, private_key, key_id)
            if headers is None:
                print("Failed to generate authentication headers.")
                return None

            try:
                response = requests.get(f"{KALSHI_BASE_URL}{path}", headers=headers, params=params)
                response.raise_for_status()
                api_response_data = response.json()

                candlesticks_from_api = api_response_data.get("candlesticks")

                if candlesticks_from_api and isinstance(candlesticks_from_api, list):
                    processed_chunk = []
                    for candle_data in candlesticks_from_api:
                        ts = candle_data.get("end_period_ts")
                        volume = candle_data.get("volume", 0)
                        price_info = candle_data.get("price", {})
                        open_price = price_info.get("open")
                        high_price = price_info.get("high")
                        low_price = price_info.get("low")
                        close_price = price_info.get("close")

                        if open_price is None and "yes_bid" in candle_data:
                            open_price = candle_data["yes_bid"].get("open")
                        if high_price is None and "yes_bid" in candle_data:
                            high_price = candle_data["yes_bid"].get("high")
                        if low_price is None and "yes_bid" in candle_data:
                            low_price = candle_data["yes_bid"].get("low")
                        if close_price is None and "yes_bid" in candle_data:
                            close_price = candle_data["yes_bid"].get("close")

                        if ts is None or open_price is None or high_price is None or low_price is None or close_price is None:
                            continue

                        processed_chunk.append({
                            "timestamp_s": ts,
                            "open": open_price,
                            "high": high_price,
                            "low": low_price,
                            "close": close_price,
                            "volume": volume
                        })

                    if processed_chunk:
                        all_candlesticks_processed.extend(processed_chunk)

                    actual_chunk_duration_seconds = chunk_end_ts - current_start_ts
                    intervals_in_this_chunk_param = actual_chunk_duration_seconds // (period_minutes * 60) if period_minutes > 0 else 0
                    pbar.update(max(0, intervals_in_this_chunk_param))

                    if processed_chunk:
                        last_processed_ts = processed_chunk[-1]["timestamp_s"]
                        current_start_ts = last_processed_ts + 1
                    else:
                        current_start_ts = chunk_end_ts + 1
                else:
                    if current_start_ts < end_ts_s:
                        pbar.update(max(0, (chunk_end_ts - current_start_ts) // (period_minutes * 60) if period_minutes > 0 else 0))
                    current_start_ts = chunk_end_ts + 1

                if current_start_ts >= end_ts_s:
                    break

                time.sleep(0.7) # Be respectful to the API

            except requests.exceptions.HTTPError as e:
                print(f"HTTP Error: {e.response.status_code} - {e.response.text}")
                if e.response.status_code == 401: print("Auth failed.")
                break
            except Exception as e:
                print(f"An unexpected error occurred in loop: {e}")
                import traceback
                traceback.print_exc()
                break

    if pbar.n < pbar.total:
        pbar.update(pbar.total - pbar.n)
    pbar.close()

    if not all_candlesticks_processed:
        print(f"No candlestick data successfully collected and processed for {market_ticker}.")
        return None

    df = pd.DataFrame(all_candlesticks_processed)
    df['timestamp'] = pd.to_datetime(df['timestamp_s'], unit='s', utc=True)
    df.set_index('timestamp', inplace=True)
    df_final = df[['open', 'high', 'low', 'close', 'volume']].copy()
    df_final.rename(columns={
        'open': 'kalshi_open_cents',
        'high': 'kalshi_high_cents',
        'low': 'kalshi_low_cents',
        'close': 'kalshi_close_cents',
        'volume': 'kalshi_volume'
    }, inplace=True)

    return df_final

if __name__ == "__main__":
    if not KALSHI_API_KEY_ID or not KALSHI_PRIVATE_KEY_PATH:
        print("Error: KALSHI_API_KEY_ID or KALSHI_PRIVATE_KEY_PATH not found in .env file.")
        exit()

    private_key = load_private_key(KALSHI_PRIVATE_KEY_PATH)
    if private_key is None:
        exit()

    edt_tz = datetime.timezone(datetime.timedelta(hours=-4)) # EDT is UTC-4

    for market_config in MARKETS_TO_FETCH:
        closing_hour_edt = market_config["closing_hour_edt"]
        market_description = market_config["description"]

        # Construct the target market ticker
        # Example: KXETHD-25MAY1318-T2649.99
        # Date part for ticker: YYMMDD -> 25MAY13 (derived from COMMON_DATE_STR "2025-05-13")
        date_obj = datetime.datetime.strptime(COMMON_DATE_STR, "%Y-%m-%d")
        ticker_date_part = f"{date_obj.strftime('%y%b%d').upper()}" # e.g., 25MAY13

        target_market_ticker = f"{TARGET_SERIES_TICKER}-{ticker_date_part}{closing_hour_edt:02d}-{COMMON_THRESHOLD_PART}"

        # Determine start and end times for fetching
        # Market opens 1 hour before closing_hour_edt and closes at closing_hour_edt
        end_dt_edt = datetime.datetime.strptime(f"{COMMON_DATE_STR} {closing_hour_edt:02d}:00:00", "%Y-%m-%d %H:%M:%S").replace(tzinfo=edt_tz)
        start_dt_edt = end_dt_edt - datetime.timedelta(hours=1)

        try:
            start_timestamp_s = int(start_dt_edt.timestamp())
            end_timestamp_s = int(end_dt_edt.timestamp())
        except ValueError as e:
            print(f"Error parsing dates for market {target_market_ticker}. Ensure they are correct and EDT offset is appropriate: {e}")
            continue # Skip to the next market

        print(f"\n--- Processing Market: {target_market_ticker} ({market_description}) ---")
        print(f"Market Open (EDT): {start_dt_edt.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        print(f"Market Close (EDT): {end_dt_edt.strftime('%Y-%m-%d %H:%M:%S %Z')}")


        # Define output filenames for this specific market
        base_output_filename = f"{target_market_ticker}_candlesticks_{PERIOD_INTERVAL_MINUTES}min"
        output_filepath_parquet = OUTPUT_DIR / f"{base_output_filename}.parquet"
        output_filepath_csv = OUTPUT_DIR / f"{base_output_filename}.csv"

        market_history_df = fetch_kalshi_candlesticks(
            series_ticker=TARGET_SERIES_TICKER,
            market_ticker=target_market_ticker,
            start_ts_s=start_timestamp_s,
            end_ts_s=end_timestamp_s,
            period_minutes=PERIOD_INTERVAL_MINUTES,
            private_key=private_key,
            key_id=KALSHI_API_KEY_ID
        )

        if market_history_df is not None and not market_history_df.empty:
            # Save to Parquet
            try:
                market_history_df.to_parquet(output_filepath_parquet)
                print(f"Successfully downloaded and saved market history to Parquet: {output_filepath_parquet}")
                print(f"Shape of saved Parquet data: {market_history_df.shape}")
            except Exception as e:
                print(f"Error saving data to Parquet: {e}")

            # Save to CSV
            try:
                market_history_df.to_csv(output_filepath_csv)
                print(f"Successfully downloaded and saved market history to CSV: {output_filepath_csv}")
                print(f"Shape of saved CSV data: {market_history_df.shape}")
            except Exception as e:
                print(f"Error saving data to CSV: {e}")

            print("\nSample data (from DataFrame):")
            print(market_history_df.head())
            # Check for volume
            total_volume = market_history_df['kalshi_volume'].sum()
            print(f"Total volume for {target_market_ticker} in this period: {total_volume}")
            if total_volume > 0:
                print(f"SUCCESS: Found volume for {target_market_ticker}!")
            else:
                print(f"NOTE: No volume found for {target_market_ticker} in this period (as expected for future markets).")

        else:
            print(f"Failed to fetch or no data available for {target_market_ticker} in the specified range.")

        print("--------------------------------------------------")
        time.sleep(1) # Small delay before fetching the next market