import asyncio
import websockets # Ensure this is up-to-date: pip install --upgrade websockets
import json
import time
import base64
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import padding, rsa
import os
from dotenv import load_dotenv
from datetime import datetime
import logging

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
load_dotenv()
KALSHI_API_KEY_ID = os.getenv("KALSHI_API_KEY_ID")
KALSHI_PRIVATE_KEY_PATH = os.getenv("KALSHI_PRIVATE_KEY_PATH")

# Check if running in DEMO mode via an environment variable
IS_DEMO_MODE = os.getenv("KALSHI_DEMO_MODE", "false").lower() == "true"
if IS_DEMO_MODE:
    logger.info("KALSHI: Running in DEMO mode.")
    KALSHI_WS_BASE_URL = "wss://demo-api.kalshi.co/trade-api/ws/v2"
    # For demo, you might need different API keys if they are separate
    KALSHI_DEMO_API_KEY_ID = os.getenv("KALSHI_DEMO_API_KEY_ID", KALSHI_API_KEY_ID)
    KALSHI_DEMO_PRIVATE_KEY_PATH = os.getenv("KALSHI_DEMO_PRIVATE_KEY_PATH", KALSHI_PRIVATE_KEY_PATH)
    if KALSHI_DEMO_API_KEY_ID != KALSHI_API_KEY_ID : KALSHI_API_KEY_ID = KALSHI_DEMO_API_KEY_ID
    if KALSHI_DEMO_PRIVATE_KEY_PATH != KALSHI_PRIVATE_KEY_PATH : KALSHI_PRIVATE_KEY_PATH = KALSHI_DEMO_PRIVATE_KEY_PATH
else:
    logger.info("KALSHI: Running in PRODUCTION mode.")
    KALSHI_WS_BASE_URL = "wss://api.elections.kalshi.com/trade-api/ws/v2"


# Target market from your latest screenshot
TARGET_KALSHI_MARKET_TICKER = "KXETHD-25MAY1421-T2629.99" # ETH > $2630 @ 9PM EDT May 14

# --- Global State ---
kalshi_order_books = {}
kalshi_command_id_counter = 1

# --- Kalshi Auth Functions ---
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
            message,
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH),
            hashes.SHA256()
        )
        return base64.b64encode(signature).decode('utf-8')
    except Exception as e:
        logger.error(f"Error during signing: {e}")
        return None

def get_kalshi_ws_auth_headers(private_key: rsa.RSAPrivateKey, key_id: str) -> dict | None:
    ws_path_for_signing = "/trade-api/ws/v2"
    method = "GET"
    timestamp_ms_str = str(int(time.time() * 1000))
    message_to_sign = timestamp_ms_str + method.upper() + ws_path_for_signing
    signature = sign_pss_text(private_key, message_to_sign)
    if signature is None:
        logger.error("Failed to sign Kalshi auth message.")
        return None
    headers = {
        'KALSHI-ACCESS-KEY': key_id,
        'KALSHI-ACCESS-SIGNATURE': signature,
        'KALSHI-ACCESS-TIMESTAMP': timestamp_ms_str,
    }
    logger.debug(f"Generated Kalshi WS Auth Headers: {headers}")
    return headers

# --- Kalshi WebSocket Client Task ---
async def kalshi_websocket_client_task(market_ticker_to_subscribe: str):
    global kalshi_command_id_counter, kalshi_order_books

    if not KALSHI_API_KEY_ID or not KALSHI_PRIVATE_KEY_PATH:
        logger.error("Kalshi API credentials not found in .env file. KALSHI_API_KEY_ID and KALSHI_PRIVATE_KEY_PATH must be set.")
        return
    private_key = load_private_key(KALSHI_PRIVATE_KEY_PATH)
    if not private_key:
        logger.error(f"Failed to load Kalshi private key from: {KALSHI_PRIVATE_KEY_PATH}")
        return

    while True:
        auth_headers = get_kalshi_ws_auth_headers(private_key, KALSHI_API_KEY_ID)
        if not auth_headers:
            logger.error("Failed to generate Kalshi WS auth headers. Retrying in 30s.")
            await asyncio.sleep(30)
            continue

        uri = KALSHI_WS_BASE_URL
        logger.info(f"Attempting to connect to Kalshi WebSocket: {uri} for market {market_ticker_to_subscribe}")
        
        connection_established_once = False
        try:
            # Key change: Set a shorter open_timeout to fail faster if connection cannot be established
            async with websockets.connect(
                uri,
                extra_headers=auth_headers,
                ping_interval=10,
                ping_timeout=20,
                open_timeout=10 # Timeout for WebSocket opening handshake
            ) as websocket:
                connection_established_once = True
                logger.info(f"Kalshi: Successfully connected to WebSocket for {market_ticker_to_subscribe}.")
                kalshi_command_id_counter = 1 

                kalshi_order_books[market_ticker_to_subscribe] = {"active": False, "seq": 0, "yes": {}, "no": {}}

                subscribe_cmd = {
                    "id": kalshi_command_id_counter,
                    "cmd": "subscribe",
                    "params": {
                        "channels": ["orderbook_delta"],
                        "market_tickers": [market_ticker_to_subscribe]
                    }
                }
                kalshi_command_id_counter += 1
                await websocket.send(json.dumps(subscribe_cmd))
                logger.info(f"Kalshi: Sent subscription request for {market_ticker_to_subscribe}: {json.dumps(subscribe_cmd)}")

                async for message_str in websocket:
                    try:
                        message = json.loads(message_str)
                        logger.debug(f"Kalshi Raw for {market_ticker_to_subscribe}: {message}")
                        # ... (rest of message handling logic from previous script) ...
                        msg_type = message.get("type")
                        sid = message.get("sid")
                        cmd_id_resp = message.get("id")

                        if msg_type == "subscribed":
                            channel = message.get("msg", {}).get("channel")
                            server_assigned_sid = message.get("msg", {}).get("sid")
                            logger.info(f"Kalshi: Subscribed to '{channel}' for {market_ticker_to_subscribe} (CmdID: {cmd_id_resp}, Server SID: {server_assigned_sid}).")
                            kalshi_order_books[market_ticker_to_subscribe].update({
                                "sid": server_assigned_sid, "active": True, "seq": 0
                            })

                        elif msg_type == "orderbook_snapshot":
                            seq = message.get("seq")
                            data = message.get("msg", {})
                            msg_market_ticker = data.get("market_ticker")
                            if msg_market_ticker == market_ticker_to_subscribe:
                                logger.info(f"Kalshi: ORDERBOOK_SNAPSHOT for {msg_market_ticker} (SID: {sid}, Seq: {seq})")
                                kalshi_order_books[msg_market_ticker].update({
                                    "yes": {int(level[0]): int(level[1]) for level in data.get("yes", [])},
                                    "no": {int(level[0]): int(level[1]) for level in data.get("no", [])},
                                    "seq": seq, "active": True
                                })
                                print_kalshi_book_summary(msg_market_ticker)

                        elif msg_type == "orderbook_delta":
                            seq = message.get("seq")
                            data = message.get("msg", {})
                            msg_market_ticker = data.get("market_ticker")
                            if msg_market_ticker == market_ticker_to_subscribe and \
                               kalshi_order_books.get(market_ticker_to_subscribe, {}).get("active"):
                                price = data.get("price")
                                delta_val = data.get("delta")
                                side = data.get("side")
                                current_book_info = kalshi_order_books[msg_market_ticker]
                                if seq != current_book_info.get("seq", -1) + 1:
                                    logger.warning(f"Kalshi: Sequence gap for {msg_market_ticker}! Expected {current_book_info.get('seq', -1) + 1}, got {seq}. Closing to resync.")
                                    current_book_info["active"] = False
                                    await websocket.close(code=1000, reason="Resync due to sequence gap")
                                    break 
                                current_book_info["seq"] = seq
                                side_book = current_book_info.get(side, {})
                                new_quantity = side_book.get(price, 0) + delta_val
                                if new_quantity > 0: side_book[price] = new_quantity
                                else: side_book.pop(price, None)
                                current_book_info[side] = side_book
                            
                        elif msg_type == "error":
                            error_msg = message.get("msg", {})
                            logger.error(f"Kalshi WS Error for {market_ticker_to_subscribe} (CmdID: {cmd_id_resp}): Code {error_msg.get('code')}, Msg: {error_msg.get('msg')}")
                            # If error is "Already subscribed", we might need to update our state if SIDs differ or is missing
                            if error_msg.get('code') == 6 and market_ticker_to_subscribe in kalshi_order_books: # "Already subscribed"
                                 logger.info(f"Kalshi: 'Already subscribed' for {market_ticker_to_subscribe}. SID in error: {sid}. Our SID: {kalshi_order_books[market_ticker_to_subscribe].get('sid')}")
                                 # This usually means a previous subscription is still seen as active by server.
                                 # Could try to use the SID from the error message if logic is in place.
                                 # Or just let the reconnection logic handle it by restarting.
                                 # Forcing a reconnect might be safest here.
                                 kalshi_order_books[market_ticker_to_subscribe]['active'] = False
                                 await websocket.close(code=1000, reason="Handling 'Already Subscribed' error")
                                 break


                    except json.JSONDecodeError:
                        logger.error(f"Kalshi: JSON Decode Error for {market_ticker_to_subscribe} - {message_str}")
                    except Exception as e:
                        logger.exception(f"Kalshi: Error processing message for {market_ticker_to_subscribe} - {e}")
        
        except TypeError as e:
            if "unexpected keyword argument 'extra_headers'" in str(e):
                logger.error(f"Kalshi: CRITICAL TypeError related to 'extra_headers'. Error: {e}")
                logger.error("This indicates a fundamental issue with the 'websockets' library version or its compatibility with your Python environment (e.g., Python 3.13).")
                logger.error("Please try: 1. `pip install --upgrade websockets`. 2. If problem persists, check websockets library issues for Python 3.13. 3. Consider testing with Python 3.10/3.11.")
                logger.error("Stopping further Kalshi connection attempts due to this TypeError.")
                return # Exit this task permanently
            else:
                logger.exception(f"Kalshi: Unexpected TypeError for {market_ticker_to_subscribe}: {e}. Retrying in 30s.")
        except websockets.exceptions.InvalidStatusCode as e:
            logger.error(f"Kalshi: WS connection failed for {market_ticker_to_subscribe} with status {e.status_code}. Headers: {e.response_headers if hasattr(e, 'response_headers') else 'N/A'}. Retrying in 30s.")
            if e.status_code == 401: logger.error("Kalshi: Authentication failed (401). Check API key, signature, and timestamp logic.")
        except (websockets.exceptions.ConnectionClosed, ConnectionRefusedError, websockets.exceptions.WebSocketException, asyncio.TimeoutError) as e: # Added TimeoutError
            logger.error(f"Kalshi: WS connection issue for {market_ticker_to_subscribe}: {type(e).__name__} - {e}. Retrying...")
        except Exception as e:
            logger.exception(f"Kalshi: Unexpected WS error for {market_ticker_to_subscribe}: {e}. Retrying in 30s.")
        
        if market_ticker_to_subscribe in kalshi_order_books: # Mark as inactive on any disconnect/error before retry
            kalshi_order_books[market_ticker_to_subscribe]["active"] = False
        
        retry_delay = 10 if connection_established_once else 30 # Shorter retry if we were connected once
        logger.info(f"Kalshi: Will retry connection for {market_ticker_to_subscribe} in {retry_delay} seconds.")
        await asyncio.sleep(retry_delay)

# --- Helper to Print Kalshi Book Summary ---
def print_kalshi_book_summary(market_ticker):
    # ... (same as previous version) ...
    if market_ticker in kalshi_order_books and kalshi_order_books[market_ticker].get('active'):
        book_info = kalshi_order_books[market_ticker]
        yes_book = book_info.get("yes", {}) 
        no_book = book_info.get("no", {})

        yes_bids_prices = sorted([p for p, q in yes_book.items() if q > 0], reverse=True)
        no_bids_prices = sorted([p for p, q in no_book.items() if q > 0], reverse=True)

        ui_yes_bid = yes_bids_prices[0] if yes_bids_prices else None
        ui_yes_ask = (100 - no_bids_prices[0]) if no_bids_prices else None

        logger.info(f"--- Kalshi Summary: {market_ticker} (Seq: {book_info.get('seq', 'N/A')}, SID: {book_info.get('sid', 'N/A')}) ---")
        
        yb_qty = yes_book.get(ui_yes_bid, 0) if ui_yes_bid is not None else 0
        logger.info(f"  Best Yes Bid: {ui_yes_bid if ui_yes_bid is not None else 'N/A'}¢ (Qty: {yb_qty})")
        
        ya_qty_origin_no_bid = 0
        if ui_yes_ask is not None:
            original_no_bid_price = 100 - ui_yes_ask
            ya_qty_origin_no_bid = no_book.get(original_no_bid_price, 0)
        logger.info(f"  Best Yes Ask: {ui_yes_ask if ui_yes_ask is not None else 'N/A'}¢ (Qty from No Side: {ya_qty_origin_no_bid})")
        
        # logger.debug(f"  Raw Yes Book: {yes_book}")
        # logger.debug(f"  Raw No Book (Bids for NO): {no_book}")
    else:
        logger.info(f"  No active order book data for Kalshi market: {market_ticker}")


# --- Main Application ---
async def main():
    logger.info("Starting Kalshi Focused Data Streamer...")
    
    if not (KALSHI_API_KEY_ID and KALSHI_PRIVATE_KEY_PATH and os.path.exists(KALSHI_PRIVATE_KEY_PATH)):
        logger.critical("CRITICAL: Kalshi API Key ID or Private Key Path is missing or invalid in .env. Exiting.")
        return

    kalshi_task = asyncio.create_task(kalshi_websocket_client_task(TARGET_KALSHI_MARKET_TICKER))

    display_interval = 10
    while True:
        await asyncio.sleep(display_interval)
        logger.info(f"===== Main Loop Display Tick ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) =====")
        print_kalshi_book_summary(TARGET_KALSHI_MARKET_TICKER)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Application interrupted by user.")
    except Exception as e:
        logger.exception("Unhandled exception in main application.")