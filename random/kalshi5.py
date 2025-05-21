# kalshi.py
import asyncio
import websockets
import json
import time
import base64
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import padding, rsa
import os
from dotenv import load_dotenv
import logging
from datetime import datetime
import csv
import pathlib # For path manipulation

# --- Basic Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(funcName)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("kalshi_monitor")

# --- Configuration (loaded once) ---
KALSHI_API_KEY_ID = None
KALSHI_PRIVATE_KEY_PATH = None
KALSHI_WS_BASE_URL = None
CSV_LOG_DIR = pathlib.Path("market_data_logs")

# --- Kalshi Auth Functions ---
def load_private_key(file_path: str) -> rsa.RSAPrivateKey | None:
    try:
        with open(file_path, "rb") as key_file:
            private_key = serialization.load_pem_private_key(key_file.read(), password=None)
        return private_key
    except FileNotFoundError:
        logger.error(f"Kalshi private key file not found: {file_path}")
        return None
    except Exception as e:
        logger.error(f"Error loading Kalshi private key from {file_path}: {e}")
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
        logger.error(f"Error during Kalshi signing: {e}")
        return None

def get_kalshi_ws_auth_headers(private_key: rsa.RSAPrivateKey, key_id: str) -> dict | None:
    ws_path_for_signing = "/trade-api/ws/v2"
    method = "GET"
    timestamp_ms_str = str(int(time.time() * 1000))
    message_to_sign = timestamp_ms_str + method.upper() + ws_path_for_signing
    signature = sign_pss_text(private_key, message_to_sign)
    if signature is None: return None
    return {'KALSHI-ACCESS-KEY': key_id, 'KALSHI-ACCESS-SIGNATURE': signature, 'KALSHI-ACCESS-TIMESTAMP': timestamp_ms_str}

def init_config():
    global KALSHI_API_KEY_ID, KALSHI_PRIVATE_KEY_PATH, KALSHI_WS_BASE_URL
    load_dotenv()
    KALSHI_API_KEY_ID = os.getenv("KALSHI_API_KEY_ID")
    KALSHI_PRIVATE_KEY_PATH = os.getenv("KALSHI_PRIVATE_KEY_PATH")

    is_demo = os.getenv("KALSHI_DEMO_MODE", "false").lower() == "true"
    if is_demo:
        logger.info("Kalshi: Running in DEMO mode.")
        KALSHI_WS_BASE_URL = "wss://demo-api.kalshi.co/trade-api/ws/v2"
        demo_key_id = os.getenv("KALSHI_DEMO_API_KEY_ID")
        demo_key_path = os.getenv("KALSHI_DEMO_PRIVATE_KEY_PATH")
        if demo_key_id: KALSHI_API_KEY_ID = demo_key_id
        if demo_key_path: KALSHI_PRIVATE_KEY_PATH = demo_key_path
    else:
        logger.info("Kalshi: Running in PRODUCTION mode.")
        KALSHI_WS_BASE_URL = os.getenv("KALSHI_PROD_WS_BASE_URL", "wss://api.elections.kalshi.com/trade-api/ws/v2")

    if not KALSHI_API_KEY_ID or not KALSHI_PRIVATE_KEY_PATH:
        logger.critical("Kalshi API credentials (API Key ID or Private Key Path) not found in .env. Please set them.")
        exit(1)
    if not os.path.exists(KALSHI_PRIVATE_KEY_PATH):
        logger.critical(f"Kalshi private key file not found at path: {KALSHI_PRIVATE_KEY_PATH}. Please check .env.")
        exit(1)
    
    CSV_LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"CSV logs will be saved to: {CSV_LOG_DIR.resolve()}")


def _extract_ui_bid_ask(yes_book_data: dict, no_book_data: dict):
    yes_bids_prices = sorted([p for p, q in yes_book_data.items() if q > 0], reverse=True)
    no_bids_prices = sorted([p for p, q in no_book_data.items() if q > 0], reverse=True)

    ui_yes_bid = yes_bids_prices[0] if yes_bids_prices else None
    ui_yes_bid_qty = yes_book_data.get(ui_yes_bid, 0) if ui_yes_bid is not None else 0

    ui_yes_ask = (100 - no_bids_prices[0]) if no_bids_prices else None
    ui_yes_ask_qty_on_no_side = 0
    if ui_yes_ask is not None:
        original_no_bid_price = 100 - ui_yes_ask
        ui_yes_ask_qty_on_no_side = no_book_data.get(original_no_bid_price, 0)

    return {
        "ui_yes_bid": ui_yes_bid,
        "ui_yes_bid_qty": ui_yes_bid_qty,
        "ui_yes_ask": ui_yes_ask,
        "ui_yes_ask_qty_on_no_side": ui_yes_ask_qty_on_no_side
    }

def _log_market_data_to_csv(market_ticker: str, data_type: str, seq: int, ui_data: dict, delta_details: dict = None):
    csv_file_path = CSV_LOG_DIR / f"{market_ticker.replace('/', '_')}.csv"
    file_exists = csv_file_path.exists()

    row_data = {
        'timestamp': datetime.now().isoformat(),
        'market_ticker': market_ticker,
        'type': data_type, 
        'yes_bid_price_cents': ui_data.get('ui_yes_bid'),
        'yes_bid_qty': ui_data.get('ui_yes_bid_qty', 0),
        'yes_ask_price_cents': ui_data.get('ui_yes_ask'),
        'yes_ask_qty_on_no_side': ui_data.get('ui_yes_ask_qty_on_no_side', 0),
        'sequence_num': seq,
        'delta_side': None,
        'delta_price_cents': None,
        'delta_quantity': None
    }

    if data_type == 'DELTA' and delta_details:
        row_data.update({
            'delta_side': delta_details.get('side'),
            'delta_price_cents': delta_details.get('price'),
            'delta_quantity': delta_details.get('delta_val')
        })

    try:
        with open(csv_file_path, 'a', newline='', encoding='utf-8') as csvfile:
            fieldnames = list(row_data.keys())
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row_data)
    except IOError as e:
        logger.error(f"Error writing to CSV {csv_file_path}: {e}")


async def kalshi_orderbook_listener(market_ticker_to_subscribe: str):
    init_config() 

    private_key = load_private_key(KALSHI_PRIVATE_KEY_PATH)
    if not private_key:
        logger.critical(f"Failed to load Kalshi private key from {KALSHI_PRIVATE_KEY_PATH}. Exiting.")
        return

    current_market_book = {"active": False, "seq": 0, "yes": {}, "no": {}, "sid": None}
    kalshi_command_id_counter = 1

    while True:
        auth_headers = get_kalshi_ws_auth_headers(private_key, KALSHI_API_KEY_ID)
        if not auth_headers:
            logger.error("Kalshi: Failed to generate auth headers. Retrying in 30s.")
            await asyncio.sleep(30)
            continue

        uri = KALSHI_WS_BASE_URL
        logger.info(f"Kalshi: Attempting to connect to {uri} for market: {market_ticker_to_subscribe}")
        connection_established_once = False
        try:
            async with websockets.connect(
                uri, extra_headers=auth_headers, ping_interval=10, ping_timeout=20, open_timeout=15
            ) as websocket:
                connection_established_once = True
                logger.info(f"Kalshi: Successfully connected to WebSocket for {market_ticker_to_subscribe}.")
                
                current_market_book = {"active": False, "seq": 0, "yes": {}, "no": {}, "sid": None}

                subscribe_cmd = {
                    "id": kalshi_command_id_counter,
                    "cmd": "subscribe",
                    "params": {
                        "channels": ["orderbook_delta"],
                        "market_tickers": [market_ticker_to_subscribe] # API expects a list
                    }
                }
                kalshi_command_id_counter += 1
                await websocket.send(json.dumps(subscribe_cmd))
                logger.info(f"Kalshi: Sent subscription request: {json.dumps(subscribe_cmd)}")

                async for message_str in websocket:
                    try:
                        message = json.loads(message_str)
                        msg_type = message.get("type")
                        server_sid = message.get("sid") 

                        if msg_type == "subscribed":
                            channel = message.get("msg", {}).get("channel")
                            actual_subscription_sid = message.get("msg", {}).get("sid") 
                            logger.info(f"Kalshi: Subscribed to '{channel}' for {market_ticker_to_subscribe} (Server SID: {actual_subscription_sid}).")
                            current_market_book.update({"sid": actual_subscription_sid, "active": True, "seq": 0})

                        elif msg_type == "orderbook_snapshot":
                            seq = message.get("seq")
                            data = message.get("msg", {})
                            market_ticker_resp = data.get("market_ticker")

                            if market_ticker_resp == market_ticker_to_subscribe:
                                # logger.info(f"Kalshi: ORDERBOOK_SNAPSHOT for {market_ticker_resp} (SID: {server_sid}, Seq: {seq})")
                                yes_data = {int(level[0]): int(level[1]) for level in data.get("yes", [])}
                                no_data = {int(level[0]): int(level[1]) for level in data.get("no", [])}
                                current_market_book.update({"yes": yes_data, "no": no_data, "seq": seq, "active": True})
                                
                                ui_data = _extract_ui_bid_ask(yes_data, no_data)
                                _log_market_data_to_csv(market_ticker_resp, 'SNAPSHOT', seq, ui_data)

                                print(f"[{datetime.now().strftime('%H:%M:%S')}] SNAPSHOT {market_ticker_resp}: "
                                      f"Yes Bid: {ui_data['ui_yes_bid']}¢ (Qty: {ui_data['ui_yes_bid_qty']}), "
                                      f"Yes Ask: {ui_data['ui_yes_ask']}¢ (Qty: {ui_data['ui_yes_ask_qty_on_no_side']}) "
                                      f"| Seq: {seq}")
                            else:
                                logger.warning(f"Received snapshot for unexpected ticker: {market_ticker_resp} (subscribed to {market_ticker_to_subscribe})")


                        elif msg_type == "orderbook_delta":
                            seq = message.get("seq")
                            data = message.get("msg", {})
                            market_ticker_resp = data.get("market_ticker")

                            if market_ticker_resp == market_ticker_to_subscribe:
                                if not current_market_book.get("active"):
                                    continue 

                                price = data.get("price")
                                delta_val = data.get("delta")
                                side_of_book = data.get("side") 
                                delta_details = {"side": side_of_book, "price": price, "delta_val": delta_val}

                                if seq != current_market_book.get("seq", -1) + 1:
                                    logger.warning(f"Kalshi: Sequence gap for {market_ticker_resp}! Expected {current_market_book.get('seq', -1) + 1}, got {seq}. Closing to resync.")
                                    current_market_book["active"] = False
                                    await websocket.close(code=1000, reason="Resync due to sequence gap")
                                    break 

                                current_market_book["seq"] = seq
                                side_book_data = current_market_book.get(side_of_book, {})
                                new_quantity = side_book_data.get(price, 0) + delta_val
                                if new_quantity > 0:
                                    side_book_data[price] = new_quantity
                                else:
                                    side_book_data.pop(price, None)
                                current_market_book[side_of_book] = side_book_data
                                
                                ui_data = _extract_ui_bid_ask(current_market_book["yes"], current_market_book["no"])
                                _log_market_data_to_csv(market_ticker_resp, 'DELTA', seq, ui_data, delta_details)
                                
                                print(f"[{datetime.now().strftime('%H:%M:%S')}] DELTA    {market_ticker_resp}: "
                                      f"Yes Bid: {ui_data['ui_yes_bid']}¢ (Qty: {ui_data['ui_yes_bid_qty']}), "
                                      f"Yes Ask: {ui_data['ui_yes_ask']}¢ (Qty: {ui_data['ui_yes_ask_qty_on_no_side']}) "
                                      f"| Seq: {seq}, Δ {side_of_book.upper()}@{price}c by {delta_val}")
                            else:
                                logger.warning(f"Received delta for unexpected ticker: {market_ticker_resp} (subscribed to {market_ticker_to_subscribe})")

                        elif msg_type == "error":
                            error_msg_details = message.get("msg", {})
                            cmd_id_resp = message.get("id")
                            logger.error(f"Kalshi WS Error (CmdID: {cmd_id_resp}, SID: {server_sid}): Code {error_msg_details.get('code')}, Msg: {error_msg_details.get('msg')}")
                            if error_msg_details.get('code') == 6: 
                                logger.info(f"Kalshi: 'Already subscribed' for {market_ticker_to_subscribe}. Server SID: {server_sid}. Assuming active.")
                                current_market_book.update({"sid": server_sid, "active": True})
                    
                    except json.JSONDecodeError:
                        logger.error(f"Kalshi: JSON Decode Error - {message_str}")
                    except Exception as e:
                        logger.exception(f"Kalshi: Error processing message - {message_str} - {e}")
        
        except (websockets.exceptions.InvalidStatusCode, websockets.exceptions.ConnectionClosed,
                ConnectionRefusedError, websockets.exceptions.WebSocketException, asyncio.TimeoutError) as e:
            logger.error(f"Kalshi: WS connection issue for {market_ticker_to_subscribe}: {type(e).__name__} - {e}. Retrying...")
        except Exception as e: 
            logger.exception(f"Kalshi: Unexpected WS connection error for {market_ticker_to_subscribe}: {e}. Retrying...")

        current_market_book["active"] = False 
        
        retry_delay = 10 if connection_established_once else 30 
        logger.info(f"Kalshi: Retrying connection for {market_ticker_to_subscribe} in {retry_delay} seconds.")
        await asyncio.sleep(retry_delay)

async def main():
    print("Kalshi Live Market Monitor")
    print("--------------------------")
    
    target_market_ticker = "KXBTCD-25MAY2116-T108749.99" 

    logger.info(f"Monitoring market: {target_market_ticker}")
    logger.info(f"Data will be logged to CSV and printed to console.")

    try:
        await kalshi_orderbook_listener(target_market_ticker)
    except KeyboardInterrupt:
        logger.info("Kalshi monitor stopped by user.")
    except Exception as e:
        logger.exception(f"An unhandled error occurred in main: {e}")
    finally:
        logger.info("Kalshi monitor finished.")

if __name__ == "__main__":
    asyncio.run(main())