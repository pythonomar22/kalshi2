import asyncio
import websockets
import json
import time
from datetime import datetime
import csv
import pathlib # For CSV path and directory management
import logging # Using logging for better output control

# --- Basic Logging Setup ---
# Configure logging to be similar to your Kalshi script for consistency
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(funcName)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("binance_streamer")

# --- Configuration ---
STREAMS_TO_SUBSCRIBE = [
    # "ethusdt@miniTicker",
    # "btcusdt@miniTicker",
    # "ethusdt@kline_1m",
    "btcusdt@kline_1m"
    # "ethusdt@trade",
    # "btcusdt@trade",
]

BINANCE_WS_ENDPOINT = "wss://data-stream.binance.vision/stream"
CSV_LOG_DIR = pathlib.Path("binance_market_data_logs")
SUBSCRIPTION_ID_COUNTER = 1 # Global counter for subscription IDs

# --- CSV Logging Function ---
def ensure_log_dir():
    """Ensures the CSV log directory exists."""
    CSV_LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Binance CSV logs will be saved to: {CSV_LOG_DIR.resolve()}")

def log_kline_to_csv(kline_event_data: dict):
    """Logs kline data to a CSV file."""
    stream_name = kline_event_data.get('stream', 'unknown_stream')
    kline_data = kline_event_data.get('data', {}).get('k', {})

    if not kline_data:
        logger.warning(f"Missing kline data in event: {kline_event_data}")
        return

    symbol = kline_data.get('s')
    # Create a unique CSV filename per symbol and interval to keep data organized
    # e.g., btcusdt_kline_1m.csv
    csv_filename = f"{symbol.lower()}_2kline_{kline_data.get('i', 'unknown_interval')}.csv"
    csv_file_path = CSV_LOG_DIR / csv_filename
    file_exists = csv_file_path.exists()

    # Define the data to be written
    row_data = {
        'reception_timestamp_utc': datetime.utcnow().isoformat() + "Z", # Timestamp when data was received by script
        'kline_start_time_ms': kline_data.get('t'),
        'kline_close_time_ms': kline_data.get('T'),
        'symbol': symbol,
        'interval': kline_data.get('i'),
        'open_price': kline_data.get('o'),
        'close_price': kline_data.get('c'),
        'high_price': kline_data.get('h'),
        'low_price': kline_data.get('l'),
        'base_asset_volume': kline_data.get('v'),
        'number_of_trades': kline_data.get('n'),
        'is_kline_closed': kline_data.get('x'),
        'quote_asset_volume': kline_data.get('q'),
        'taker_buy_base_asset_volume': kline_data.get('V'),
        'taker_buy_quote_asset_volume': kline_data.get('Q'),
        'stream_name': stream_name
    }

    try:
        with open(csv_file_path, 'a', newline='', encoding='utf-8') as csvfile:
            fieldnames = list(row_data.keys())
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            if not file_exists:
                writer.writeheader()
                logger.info(f"Created new CSV log file: {csv_file_path}")
            
            writer.writerow(row_data)
    except IOError as e:
        logger.error(f"Error writing Binance kline data to CSV {csv_file_path}: {e}")
    except Exception as e:
        logger.error(f"Unexpected error during CSV writing for Binance: {e}")


# --- Binance WebSocket Client ---
async def binance_websocket_client_task():
    global SUBSCRIPTION_ID_COUNTER # Use the global counter
    uri = BINANCE_WS_ENDPOINT
    ensure_log_dir() # Make sure log directory exists

    while True: # Outer loop for reconnection
        current_subscription_id = SUBSCRIPTION_ID_COUNTER # Use and then increment for this connection attempt
        SUBSCRIPTION_ID_COUNTER +=1
        
        logger.info(f"Attempting to connect to Binance WebSocket at {uri}")
        try:
            async with websockets.connect(uri, ping_interval=20, ping_timeout=60, open_timeout=10) as websocket:
                logger.info(f"Successfully connected to Binance WebSocket.")

                if not STREAMS_TO_SUBSCRIBE:
                    logger.warning("No streams configured in STREAMS_TO_SUBSCRIBE. Listener will be idle.")
                    await asyncio.sleep(3600) # Sleep for a long time if no streams
                    continue

                subscribe_payload = {
                    "method": "SUBSCRIBE",
                    "params": STREAMS_TO_SUBSCRIBE,
                    "id": current_subscription_id
                }
                await websocket.send(json.dumps(subscribe_payload))
                logger.info(f"Sent subscription request (ID: {current_subscription_id}): {json.dumps(subscribe_payload)}")

                logger.info("--- Listening for live Binance data ---")
                async for message_str in websocket:
                    try:
                        message_data = json.loads(message_str)

                        if 'stream' in message_data and 'data' in message_data:
                            data_payload = message_data['data']
                            event_type = data_payload.get('e')
                            stream_name = message_data['stream'] # For logging context

                            if event_type == 'kline':
                                kline_data_full = data_payload['k']
                                symbol = kline_data_full['s']
                                interval = kline_data_full['i']
                                is_closed = kline_data_full['x']
                                close_price = kline_data_full['c']
                                
                                # Log to console (optional, can be made more concise or removed)
                                logger.debug(f"KLINE_{interval} | {symbol}: C={close_price}, Closed={is_closed} (Stream: {stream_name})")
                                
                                # Log to CSV
                                log_kline_to_csv(message_data) # Pass the whole message_data for context

                            elif event_type == '24hrMiniTicker':
                                symbol = data_payload['s']
                                last_price = data_payload['c']
                                logger.debug(f"MINI_TICKER | {symbol}: Last Price = {last_price} (Stream: {stream_name})")
                                # CSV logging for miniTicker could be added here if needed, similar to klines

                            # Add handlers for other event_types like 'trade' if you subscribe to them

                        elif "result" in message_data and "id" in message_data:
                            logger.info(f"Subscription response (ID: {message_data['id']}): {json.dumps(message_data)}")
                        else:
                            logger.debug(f"Other Binance message: {json.dumps(message_data)}")

                    except json.JSONDecodeError:
                        logger.error(f"Binance: JSON Decode Error - {message_str}")
                    except Exception as e:
                        logger.exception(f"Binance: Error processing message - {e} - Message: {message_str}")
        
        except (websockets.exceptions.ConnectionClosed, websockets.exceptions.ConnectionClosedError,
                websockets.exceptions.ConnectionClosedOK, websockets.exceptions.InvalidStatusCode,
                asyncio.TimeoutError, ConnectionRefusedError) as e:
            logger.error(f"Binance WebSocket connection issue: {type(e).__name__} - {e}.")
        except Exception as e:
            logger.exception(f"Unexpected error in Binance WebSocket client task: {e}")
        
        logger.info("Attempting to reconnect to Binance in 10 seconds...")
        await asyncio.sleep(10)


if __name__ == "__main__":
    logger.info("Starting Binance WebSocket client...")
    if STREAMS_TO_SUBSCRIBE:
        logger.info(f"Attempting to subscribe to: {', '.join(STREAMS_TO_SUBSCRIBE)}")
    else:
        logger.warning("STREAMS_TO_SUBSCRIBE is empty. The client will connect but not receive market data.")

    try:
        asyncio.run(binance_websocket_client_task())
    except KeyboardInterrupt:
        logger.info("\nManually interrupted by user. Exiting Binance streamer.")
    except Exception as e:
        logger.exception(f"Main execution error for Binance streamer: {e}")