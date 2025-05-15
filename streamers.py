# streamers.py
import asyncio
import websockets
import json
import time
import base64
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import padding, rsa
import os
from dotenv import load_dotenv
from datetime import datetime
import logging

logger = logging.getLogger(__name__) # Will be configured by the main script

# --- Global Data Storage ---
# These will be updated by the streamers and read by the trader_core
latest_binance_data = {} # { "SYMBOL": {"c": "price", "h": "high", ... kline_data} }
latest_kalshi_books = {} # { "MARKET_TICKER": {"ui_yes_bid": cents, "ui_yes_ask": cents, "yes_bid_qty": qty, ... full_book_details } }

# --- Configuration (loaded once) ---
KALSHI_API_KEY_ID = None
KALSHI_PRIVATE_KEY_PATH = None
KALSHI_WS_BASE_URL = None
BINANCE_WS_ENDPOINT = "wss://data-stream.binance.vision/stream"

kalshi_command_id_counter = 1
binance_subscription_id_counter = 1

def init_config():
    global KALSHI_API_KEY_ID, KALSHI_PRIVATE_KEY_PATH, KALSHI_WS_BASE_URL
    load_dotenv()
    KALSHI_API_KEY_ID = os.getenv("KALSHI_API_KEY_ID")
    KALSHI_PRIVATE_KEY_PATH = os.getenv("KALSHI_PRIVATE_KEY_PATH")
    
    is_demo = os.getenv("KALSHI_DEMO_MODE", "false").lower() == "true"
    if is_demo:
        logger.info("KALSHI Streamer: Running in DEMO mode.")
        KALSHI_WS_BASE_URL = "wss://demo-api.kalshi.co/trade-api/ws/v2"
        # Potentially override API keys for demo if different
        demo_key_id = os.getenv("KALSHI_DEMO_API_KEY_ID")
        demo_key_path = os.getenv("KALSHI_DEMO_PRIVATE_KEY_PATH")
        if demo_key_id: KALSHI_API_KEY_ID = demo_key_id
        if demo_key_path: KALSHI_PRIVATE_KEY_PATH = demo_key_path
    else:
        logger.info("KALSHI Streamer: Running in PRODUCTION mode.")
        KALSHI_WS_BASE_URL = "wss://api.elections.kalshi.com/trade-api/ws/v2"

# --- Kalshi Auth Functions (copied from your working trader.py) ---
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

# --- Kalshi WebSocket Client Task ---
async def kalshi_websocket_client_task(market_tickers_to_subscribe: list):
    global kalshi_command_id_counter, latest_kalshi_books
    init_config() # Ensure config is loaded when task starts

    if not KALSHI_API_KEY_ID or not KALSHI_PRIVATE_KEY_PATH:
        logger.critical("Kalshi API credentials not found. KALSHI streamer cannot start.")
        return
    private_key = load_private_key(KALSHI_PRIVATE_KEY_PATH)
    if not private_key:
        logger.critical(f"Failed to load Kalshi private key. KALSHI streamer cannot start.")
        return

    while True:
        auth_headers = get_kalshi_ws_auth_headers(private_key, KALSHI_API_KEY_ID)
        if not auth_headers:
            logger.error("Kalshi: Failed to generate auth headers. Retrying in 30s.")
            await asyncio.sleep(30)
            continue

        uri = KALSHI_WS_BASE_URL
        logger.info(f"Kalshi: Attempting to connect to {uri} for markets: {market_tickers_to_subscribe}")
        connection_established_once = False
        try:
            async with websockets.connect(
                uri, extra_headers=auth_headers, ping_interval=10, ping_timeout=20, open_timeout=15
            ) as websocket:
                connection_established_once = True
                logger.info(f"Kalshi: Successfully connected to WebSocket.")
                current_kalshi_command_id = 1 # Reset for this connection

                for ticker in market_tickers_to_subscribe: # Initialize book structure
                     latest_kalshi_books[ticker] = {"active": False, "seq": 0, "yes": {}, "no": {}, "sid": None}
                
                subscribe_cmd = {
                    "id": current_kalshi_command_id, "cmd": "subscribe",
                    "params": {"channels": ["orderbook_delta"], "market_tickers": market_tickers_to_subscribe}
                }
                current_kalshi_command_id += 1
                await websocket.send(json.dumps(subscribe_cmd))
                logger.info(f"Kalshi: Sent subscription request: {json.dumps(subscribe_cmd)}")

                async for message_str in websocket:
                    try:
                        message = json.loads(message_str)
                        logger.debug(f"Kalshi Raw: {message}")
                        msg_type = message.get("type")
                        sid = message.get("sid")

                        if msg_type == "subscribed":
                            channel = message.get("msg", {}).get("channel")
                            server_assigned_sid = message.get("msg", {}).get("sid")
                            logger.info(f"Kalshi: Subscribed to '{channel}' (Server SID: {server_assigned_sid}).")
                            # Associate SID with all subscribed markets for this connection
                            # This simple model assumes one subscribe cmd covers all tickers for this SID
                            for ticker in market_tickers_to_subscribe:
                                latest_kalshi_books[ticker].update({"sid": server_assigned_sid, "active": True, "seq": 0})
                        
                        elif msg_type == "orderbook_snapshot":
                            seq = message.get("seq")
                            data = message.get("msg", {})
                            market_ticker = data.get("market_ticker")
                            if market_ticker in latest_kalshi_books:
                                logger.info(f"Kalshi: ORDERBOOK_SNAPSHOT for {market_ticker} (SID: {sid}, Seq: {seq})")
                                yes_data = {int(level[0]): int(level[1]) for level in data.get("yes", [])}
                                no_data = {int(level[0]): int(level[1]) for level in data.get("no", [])}
                                latest_kalshi_books[market_ticker].update({
                                    "yes": yes_data, "no": no_data, "seq": seq, "active": True,
                                    **_extract_ui_bid_ask(yes_data, no_data) # Add UI bids/asks directly
                                })
                        
                        elif msg_type == "orderbook_delta":
                            seq = message.get("seq")
                            data = message.get("msg", {})
                            market_ticker = data.get("market_ticker")
                            if market_ticker in latest_kalshi_books and latest_kalshi_books[market_ticker].get("active"):
                                price = data.get("price")
                                delta_val = data.get("delta")
                                side = data.get("side") # "yes" or "no"
                                current_book_info = latest_kalshi_books[market_ticker]

                                if seq != current_book_info.get("seq", -1) + 1:
                                    logger.warning(f"Kalshi: Sequence gap for {market_ticker}! Expected {current_book_info.get('seq', -1) + 1}, got {seq}. Closing to resync.")
                                    current_book_info["active"] = False
                                    await websocket.close(code=1000, reason="Resync due to sequence gap")
                                    break
                                current_book_info["seq"] = seq
                                side_book = current_book_info.get(side, {})
                                new_quantity = side_book.get(price, 0) + delta_val
                                if new_quantity > 0: side_book[price] = new_quantity
                                else: side_book.pop(price, None)
                                current_book_info[side] = side_book
                                # Update UI bid/ask after delta
                                current_book_info.update(_extract_ui_bid_ask(current_book_info["yes"], current_book_info["no"]))
                        
                        elif msg_type == "error":
                            error_msg = message.get("msg", {})
                            cmd_id_resp = message.get("id")
                            logger.error(f"Kalshi WS Error (CmdID: {cmd_id_resp}): Code {error_msg.get('code')}, Msg: {error_msg.get('msg')}")
                            if error_msg.get('code') == 6: # Already subscribed
                                logger.info(f"Kalshi: 'Already subscribed'. Server SID: {sid}. Assuming active.")
                                for ticker in market_tickers_to_subscribe: # This is a guess on how to handle sid here
                                    if ticker in latest_kalshi_books:
                                        latest_kalshi_books[ticker].update({"sid": sid, "active": True})


                    except json.JSONDecodeError: logger.error(f"Kalshi: JSON Decode Error - {message_str}")
                    except Exception as e: logger.exception(f"Kalshi: Error processing message - {e}")
        
        except (websockets.exceptions.InvalidStatusCode, websockets.exceptions.ConnectionClosed,
                ConnectionRefusedError, websockets.exceptions.WebSocketException, asyncio.TimeoutError) as e:
            logger.error(f"Kalshi: WS connection issue: {type(e).__name__} - {e}. Retrying...")
        except Exception as e: # Catch-all for other unexpected errors during connect
            logger.exception(f"Kalshi: Unexpected WS connection error: {e}. Retrying...")

        for ticker in market_tickers_to_subscribe: # Mark as inactive on any disconnect
            if ticker in latest_kalshi_books: latest_kalshi_books[ticker]["active"] = False
        
        retry_delay = 10 if connection_established_once else 30
        logger.info(f"Kalshi: Retrying connection in {retry_delay} seconds.")
        await asyncio.sleep(retry_delay)

def _extract_ui_bid_ask(yes_book_data: dict, no_book_data: dict):
    """Helper to derive UI-style bid/ask from raw book data."""
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


# --- Binance WebSocket Client Task ---
async def binance_websocket_client_task(streams_to_subscribe: list):
    global binance_subscription_id_counter, latest_binance_data
    init_config() # Ensure config (like BINANCE_WS_ENDPOINT) is available
    uri = BINANCE_WS_ENDPOINT

    while True:
        logger.info(f"Binance: Attempting to connect to {uri}")
        try:
            async with websockets.connect(uri, ping_interval=20, ping_timeout=60, open_timeout=10) as websocket:
                logger.info("Binance: Successfully connected to WebSocket.")
                current_binance_subscription_id = 1 # Reset for this connection

                subscribe_payload = {
                    "method": "SUBSCRIBE", "params": streams_to_subscribe, "id": current_binance_subscription_id
                }
                current_binance_subscription_id += 1
                await websocket.send(json.dumps(subscribe_payload))
                logger.info(f"Binance: Sent subscription request: {json.dumps(subscribe_payload)}")

                async for message_str in websocket:
                    try:
                        message_data = json.loads(message_str)
                        # logger.debug(f"Binance Raw: {message_data}")
                        if 'stream' in message_data and 'data' in message_data:
                            data_payload = message_data['data']
                            event_type = data_payload.get('e')
                            if event_type == 'kline':
                                kline = data_payload['k']
                                symbol = kline['s']
                                latest_binance_data[symbol] = kline # Store latest kline
                                logger.debug(f"Binance: KLINE_{kline['i']} | {symbol}: C={kline['c']}")
                        elif "result" in message_data and "id" in message_data: # Subscription confirmation
                            logger.info(f"Binance: Subscription response: {json.dumps(message_data)}")
                    except json.JSONDecodeError: logger.error(f"Binance: JSON Decode Error - {message_str}")
                    except Exception as e: logger.exception(f"Binance: Error processing message - {e}")
        except Exception as e:
            logger.error(f"Binance: WS connection error: {e}. Retrying in 10s.")
        await asyncio.sleep(10)

# --- Test function if this file is run directly ---
async def _test_streamers():
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s')
    logger.info("Testing streamers...")
    
    # Kalshi target for testing
    # You would get this from your main trading logic based on what you want to trade
    test_kalshi_ticker = "KXETHD-25MAY1421-T2629.99" # ETH > $2630 @ 9PM EDT May 14
    
    # Binance targets
    test_binance_streams = ["ethusdt@kline_1m", "btcusdt@kline_1m"]

    kalshi_task = asyncio.create_task(kalshi_websocket_client_task([test_kalshi_ticker]))
    binance_task = asyncio.create_task(binance_websocket_client_task(test_binance_streams))

    try:
        while True:
            await asyncio.sleep(10)
            logger.info("--- Test Loop: Current Data ---")
            if test_kalshi_ticker in latest_kalshi_books and latest_kalshi_books[test_kalshi_ticker].get('active'):
                kb = latest_kalshi_books[test_kalshi_ticker]
                logger.info(f"Kalshi {test_kalshi_ticker}: YesBid={kb.get('ui_yes_bid')}¢, YesAsk={kb.get('ui_yes_ask')}¢ (Seq: {kb.get('seq')})")
            else:
                logger.info(f"Kalshi {test_kalshi_ticker}: No active data.")
            
            for stream_name_part in ["ETHUSDT", "BTCUSDT"]:
                if stream_name_part in latest_binance_data:
                    bk = latest_binance_data[stream_name_part]
                    logger.info(f"Binance {stream_name_part} (1m kline): Close={bk.get('c')}")
                else:
                    logger.info(f"Binance {stream_name_part}: No data.")
    except KeyboardInterrupt:
        logger.info("Test streamers interrupted.")
    finally:
        kalshi_task.cancel()
        binance_task.cancel()
        await asyncio.gather(kalshi_task, binance_task, return_exceptions=True)
        logger.info("Test streamers finished.")

if __name__ == "__main__":
    asyncio.run(_test_streamers())