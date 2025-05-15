import asyncio
import websockets # Use version 10.1 as it worked: pip install websockets==10.1
import json
import time
import base64
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import padding, rsa
import os
from dotenv import load_dotenv
from datetime import datetime, timezone
import logging
from pathlib import Path

# --- Logging Setup ---
# General logger for script activity
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')
script_logger = logging.getLogger("gather_data_script")

# Specific loggers for raw data from each source
kalshi_data_logger = logging.getLogger("kalshi_raw_data")
binance_data_logger = logging.getLogger("binance_raw_data")

# --- Configuration ---
load_dotenv()
KALSHI_API_KEY_ID = os.getenv("KALSHI_API_KEY_ID")
KALSHI_PRIVATE_KEY_PATH = os.getenv("KALSHI_PRIVATE_KEY_PATH")

IS_DEMO_MODE = os.getenv("KALSHI_DEMO_MODE", "false").lower() == "true"
if IS_DEMO_MODE:
    script_logger.info("KALSHI: Running in DEMO mode for data gathering.")
    KALSHI_WS_BASE_URL = "wss://demo-api.kalshi.co/trade-api/ws/v2"
    KALSHI_DEMO_API_KEY_ID = os.getenv("KALSHI_DEMO_API_KEY_ID", KALSHI_API_KEY_ID)
    KALSHI_DEMO_PRIVATE_KEY_PATH = os.getenv("KALSHI_DEMO_PRIVATE_KEY_PATH", KALSHI_PRIVATE_KEY_PATH)
    if KALSHI_DEMO_API_KEY_ID != KALSHI_API_KEY_ID : KALSHI_API_KEY_ID = KALSHI_DEMO_API_KEY_ID
    if KALSHI_DEMO_PRIVATE_KEY_PATH != KALSHI_PRIVATE_KEY_PATH : KALSHI_PRIVATE_KEY_PATH = KALSHI_DEMO_PRIVATE_KEY_PATH
else:
    script_logger.info("KALSHI: Running in PRODUCTION mode for data gathering.")
    KALSHI_WS_BASE_URL = "wss://api.elections.kalshi.com/trade-api/ws/v2"

BINANCE_WS_ENDPOINT = "wss://data-stream.binance.vision/stream"

# Target Kalshi market
TARGET_KALSHI_MARKET_TICKER = "KXETHD-25MAY1422-T2629.99"
# Target Binance symbol and streams
BINANCE_SYMBOL_ETH = "ETHUSDT"
BINANCE_STREAMS_TO_SUBSCRIBE = [
    f"{BINANCE_SYMBOL_ETH.lower()}@kline_1m", # 1-minute klines
    # f"{BINANCE_SYMBOL_ETH.lower()}@trade",    # Individual trades (can be very verbose)
    # f"{BINANCE_SYMBOL_ETH.lower()}@depth5@100ms" # Top 5 levels of order book, 100ms updates
]

# --- Data Storage ---
DATA_OUTPUT_DIR = Path("./collected_market_data")
DATA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Kalshi Auth Functions (same as before) ---
def load_private_key(file_path: str) -> rsa.RSAPrivateKey | None:
    try:
        with open(file_path, "rb") as key_file:
            private_key = serialization.load_pem_private_key(key_file.read(), password=None)
        return private_key
    except FileNotFoundError:
        script_logger.error(f"Private key file not found: {file_path}")
        return None
    except Exception as e:
        script_logger.error(f"Error loading private key from {file_path}: {e}")
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
        script_logger.error(f"Error during signing: {e}")
        return None

def get_kalshi_ws_auth_headers(private_key: rsa.RSAPrivateKey, key_id: str) -> dict | None:
    ws_path_for_signing = "/trade-api/ws/v2"
    method = "GET"
    timestamp_ms_str = str(int(time.time() * 1000))
    message_to_sign = timestamp_ms_str + method.upper() + ws_path_for_signing
    signature = sign_pss_text(private_key, message_to_sign)
    if signature is None: return None
    return {'KALSHI-ACCESS-KEY': key_id, 'KALSHI-ACCESS-SIGNATURE': signature, 'KALSHI-ACCESS-TIMESTAMP': timestamp_ms_str}

# --- File Handlers for Data Logging ---
def setup_data_logger(logger_name, filename_prefix):
    log = logging.getLogger(logger_name)
    log.setLevel(logging.INFO) # Log every message
    log.propagate = False # Don't pass to root logger (which might have different formatting/level)
    
    # Construct filename with date to avoid overwriting and keep runs separate
    date_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    filepath = DATA_OUTPUT_DIR / f"{filename_prefix}_{date_str}.jsonl"
    
    # Use a simple formatter for raw data - just the message
    # We'll prepend our own timestamp when writing.
    # handler = logging.FileHandler(filepath, mode='a') # Append mode
    # log.addHandler(handler)
    script_logger.info(f"Data for {logger_name} will be saved to: {filepath}")
    return filepath # Return the path to be used for direct writing

# --- Kalshi WebSocket Data Gatherer ---
async def kalshi_data_gatherer_task(market_ticker: str, output_filepath: Path):
    if not KALSHI_API_KEY_ID or not KALSHI_PRIVATE_KEY_PATH:
        script_logger.critical("Kalshi API credentials not set. Kalshi data gatherer cannot start.")
        return
    private_key = load_private_key(KALSHI_PRIVATE_KEY_PATH)
    if not private_key:
        script_logger.critical(f"Failed to load Kalshi private key. Kalshi data gatherer cannot start.")
        return

    command_id_counter = 1
    while True:
        auth_headers = get_kalshi_ws_auth_headers(private_key, KALSHI_API_KEY_ID)
        if not auth_headers:
            script_logger.error("Kalshi: Failed to generate auth headers. Retrying in 30s.")
            await asyncio.sleep(30); continue

        uri = KALSHI_WS_BASE_URL
        script_logger.info(f"Kalshi: Attempting to connect to {uri} for market {market_ticker}")
        try:
            async with websockets.connect(
                uri, extra_headers=auth_headers, ping_interval=10, ping_timeout=20, open_timeout=15
            ) as websocket:
                script_logger.info(f"Kalshi: Successfully connected for {market_ticker}.")
                
                subscribe_cmd = {
                    "id": command_id_counter, "cmd": "subscribe",
                    "params": {"channels": ["orderbook_delta"], "market_tickers": [market_ticker]}
                }
                command_id_counter += 1
                await websocket.send(json.dumps(subscribe_cmd))
                script_logger.info(f"Kalshi: Sent subscription for {market_ticker}: {json.dumps(subscribe_cmd)}")

                with open(output_filepath, 'a') as f_out: # Open file for appending
                    async for message_str in websocket:
                        try:
                            # Prepend a precise timestamp to the raw message string
                            log_line = f'{datetime.now(timezone.utc).isoformat()}\t{message_str}\n'
                            f_out.write(log_line)
                            f_out.flush() # Ensure it's written immediately

                            # Optional: also log parsed message type for monitoring
                            message_data = json.loads(message_str)
                            msg_type = message_data.get("type")
                            if msg_type == "subscribed": script_logger.info(f"Kalshi: Confirmed subscription for {market_ticker}")
                            elif msg_type == "error": script_logger.error(f"Kalshi WS Error: {message_data}")
                            # Add more specific logging for other message types if needed for monitoring

                        except Exception as e:
                            script_logger.exception(f"Kalshi: Error processing/writing message for {market_ticker}: {e}")
        
        except Exception as e:
            script_logger.error(f"Kalshi: WS connection issue for {market_ticker}: {type(e).__name__} - {e}. Retrying...")
        
        script_logger.info(f"Kalshi: Retrying connection for {market_ticker} in 15 seconds.")
        await asyncio.sleep(15)


# --- Binance WebSocket Data Gatherer ---
async def binance_data_gatherer_task(streams: list, output_filepath: Path):
    uri = BINANCE_WS_ENDPOINT
    subscription_id_counter = 1
    while True:
        script_logger.info(f"Binance: Attempting to connect to {uri} for streams: {streams}")
        try:
            async with websockets.connect(uri, ping_interval=20, ping_timeout=60, open_timeout=10) as websocket:
                script_logger.info("Binance: Successfully connected.")
                
                subscribe_payload = {"method": "SUBSCRIBE", "params": streams, "id": subscription_id_counter}
                subscription_id_counter += 1
                await websocket.send(json.dumps(subscribe_payload))
                script_logger.info(f"Binance: Sent subscription: {json.dumps(subscribe_payload)}")

                with open(output_filepath, 'a') as f_out:
                    async for message_str in websocket:
                        try:
                            log_line = f'{datetime.now(timezone.utc).isoformat()}\t{message_str}\n'
                            f_out.write(log_line)
                            f_out.flush()

                            message_data = json.loads(message_str)
                            if "result" in message_data and "id" in message_data:
                                script_logger.info(f"Binance: Subscription response: {json.dumps(message_data)}")
                            # Add more specific logging if needed
                        except Exception as e:
                             script_logger.exception(f"Binance: Error processing/writing message: {e}")
        except Exception as e:
            script_logger.error(f"Binance: WS connection error: {e}. Retrying in 10s.")
        await asyncio.sleep(10)


# --- Main Application ---
async def main_data_gathering_session():
    script_logger.info("Starting Data Gathering Session...")

    if not (KALSHI_API_KEY_ID and KALSHI_PRIVATE_KEY_PATH and os.path.exists(KALSHI_PRIVATE_KEY_PATH)):
        script_logger.critical("CRITICAL: Kalshi API Key ID or Private Key Path is missing or invalid. Exiting.")
        return

    # Setup file paths for this session's data
    # Using market ticker in filename for Kalshi, and symbol for Binance
    kalshi_output_file = DATA_OUTPUT_DIR / f"kalshi_raw_{TARGET_KALSHI_MARKET_TICKER.replace('-', '_')}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.jsonl"
    binance_output_file_eth = DATA_OUTPUT_DIR / f"binance_raw_{BINANCE_SYMBOL_ETH.lower()}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.jsonl"
    
    script_logger.info(f"Kalshi data will be logged to: {kalshi_output_file}")
    script_logger.info(f"Binance ETH data will be logged to: {binance_output_file_eth}")

    kalshi_gather_task = asyncio.create_task(kalshi_data_gatherer_task(TARGET_KALSHI_MARKET_TICKER, kalshi_output_file))
    binance_gather_task = asyncio.create_task(binance_data_gatherer_task(BINANCE_STREAMS_TO_SUBSCRIBE, binance_output_file_eth))

    try:
        # Keep the script running indefinitely (or until Ctrl+C)
        # You could add a timer here if you only want to collect for a specific duration
        # For example, to run for 1 hour:
        # await asyncio.sleep(3600) 
        # logger.info("Data gathering duration reached. Shutting down.")
        await asyncio.gather(kalshi_gather_task, binance_gather_task)
    except KeyboardInterrupt:
        script_logger.info("Data gathering interrupted by user.")
    except Exception as e:
        script_logger.exception("Critical error in main gathering session: %s", e)
    finally:
        script_logger.info("Cancelling data gathering tasks...")
        if 'kalshi_gather_task' in locals() and not kalshi_gather_task.done(): kalshi_gather_task.cancel()
        if 'binance_gather_task' in locals() and not binance_gather_task.done(): binance_gather_task.cancel()
        await asyncio.sleep(1) # Allow tasks to attempt cleanup
        script_logger.info("Data gathering session ended.")

if __name__ == "__main__":
    # Ensure .env is set up
    if not (os.getenv("KALSHI_API_KEY_ID") and os.getenv("KALSHI_PRIVATE_KEY_PATH")):
        print("CRITICAL: KALSHI_API_KEY_ID or KALSHI_PRIVATE_KEY_PATH .env variables not set.")
    elif not os.path.exists(os.getenv("KALSHI_PRIVATE_KEY_PATH")):
         print(f"CRITICAL: Kalshi private key file not found at path: {os.getenv('KALSHI_PRIVATE_KEY_PATH')}")
    else:
        asyncio.run(main_data_gathering_session())