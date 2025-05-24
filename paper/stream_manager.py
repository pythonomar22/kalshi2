# /paper/stream_manager.py

import asyncio
import json
import logging
from collections import deque
from datetime import datetime, timezone, timedelta
from typing import Deque, Dict, Optional, Callable, Awaitable, Set, List 

import websockets
import pytz 

import paper.config as cfg
from paper.utils import get_kalshi_ws_auth_headers, load_private_key
from paper.data_models import BinanceKline, KalshiMarketState, KalshiMarketInfo

logger = logging.getLogger("paper_stream_manager")
# logger.setLevel(logging.DEBUG) # Uncomment for very verbose WebSocket message logging

# --- In-Memory Data Stores ---
MAX_KLINE_HISTORY = max(cfg.ROLLING_WINDOWS_MINUTES, default=30) + max(cfg.LAG_WINDOWS_MINUTES, default=30) + 60 
g_binance_kline_history: Deque[BinanceKline] = deque(maxlen=MAX_KLINE_HISTORY)
g_kalshi_market_states: Dict[str, KalshiMarketState] = {}
g_active_kalshi_ws_tasks: Dict[str, asyncio.Task] = {} 
g_kalshi_private_key = load_private_key(cfg.KALSHI_PRIVATE_KEY_PATH) 

# *** GLOBAL COUNTER FOR KALSHI COMMAND IDS ***
g_kalshi_global_command_id_counter = int(datetime.now(timezone.utc).timestamp())


# --- Binance WebSocket Handler ---
async def binance_stream_listener(
    shared_kline_history: Deque[BinanceKline],
    on_connect_callback: Optional[Callable[[], Awaitable[None]]] = None # Kept for potential future use
):
    uri = cfg.BINANCE_WS_ENDPOINT
    subscription_id_counter = 1 

    while True:
        logger.info("Binance: Attempting to connect...")
        try:
            async with websockets.connect(uri, ping_interval=20, ping_timeout=60, open_timeout=15) as websocket:
                logger.info("Binance: Successfully connected to WebSocket.")
                if on_connect_callback: # If a callback is provided by the bot
                    await on_connect_callback()

                subscribe_payload = {
                    "method": "SUBSCRIBE",
                    "params": cfg.BINANCE_STREAM_SUBSCRIPTIONS,
                    "id": subscription_id_counter
                }
                subscription_id_counter += 1
                await websocket.send(json.dumps(subscribe_payload))
                logger.info(f"Binance: Sent subscription request: {json.dumps(subscribe_payload)}")

                async for message_str in websocket:
                    logger.debug(f"Binance RAW: {message_str}") 
                    try:
                        message_data = json.loads(message_str)
                        if 'stream' in message_data and 'data' in message_data:
                            data_payload = message_data['data'] 
                            
                            if data_payload.get('e') == 'kline':
                                kline_obj = BinanceKline.model_validate(data_payload)
                                shared_kline_history.append(kline_obj)
                                logger.debug(f"Binance Parsed Kline: {kline_obj.symbol} {kline_obj.interval} C: {kline_obj.close_price} EndT: {kline_obj.kline_end_time} Closed: {kline_obj.is_closed}")
                        elif "result" in message_data and "id" in message_data:
                            logger.info(f"Binance: Subscription response (ID: {message_data['id']}): {json.dumps(message_data)}")
                    except json.JSONDecodeError:
                        logger.error(f"Binance: JSON Decode Error - {message_str}")
                    except Exception as e: 
                        logger.exception(f"Binance: Error processing message - {e} - Message: {message_str}")
        except (websockets.exceptions.ConnectionClosed, ConnectionRefusedError, websockets.exceptions.WebSocketException, asyncio.TimeoutError) as e:
            logger.error(f"Binance: WebSocket connection issue: {type(e).__name__} - {e}.")
        except Exception as e:
            logger.exception(f"Binance: Unexpected error in WebSocket client task: {e}")
        
        logger.info("Binance: Attempting to reconnect in 10 seconds...")
        await asyncio.sleep(10)

# --- Kalshi WebSocket Handler ---
async def kalshi_market_stream_listener(
    market_info: KalshiMarketInfo,
    shared_market_states: Dict[str, KalshiMarketState],
    on_disconnect_callback: Optional[Callable[[str], Awaitable[None]]] = None
):
    global g_kalshi_global_command_id_counter # Use the global counter

    market_ticker = market_info.ticker
    if not g_kalshi_private_key:
        logger.error(f"Kalshi ({market_ticker}): Private key not loaded. Cannot connect.")
        if on_disconnect_callback: await on_disconnect_callback(market_ticker)
        return

    while True: 
        auth_headers = get_kalshi_ws_auth_headers(g_kalshi_private_key, cfg.KALSHI_API_KEY_ID)
        if not auth_headers:
            logger.error(f"Kalshi ({market_ticker}): Failed to generate auth headers. Retrying in 30s.")
            await asyncio.sleep(30)
            continue

        uri = cfg.KALSHI_WS_BASE_URL
        logger.info(f"Kalshi ({market_ticker}): Attempting to connect to {uri} ...")
        connection_established_once = False
        
        try:
            async with websockets.connect(
                uri, extra_headers=auth_headers, ping_interval=10, ping_timeout=20, open_timeout=15
            ) as websocket:
                connection_established_once = True
                logger.info(f"Kalshi ({market_ticker}): Successfully connected.")
                
                current_market_state = KalshiMarketState(market_ticker=market_ticker)
                shared_market_states[market_ticker] = current_market_state

                g_kalshi_global_command_id_counter += 1
                current_command_id = g_kalshi_global_command_id_counter
                
                subscribe_cmd = {
                    "id": current_command_id, # Use simple integer ID
                    "cmd": "subscribe",
                    "params": {"channels": ["orderbook_delta"], "market_tickers": [market_ticker]}
                }
                await websocket.send(json.dumps(subscribe_cmd))
                logger.info(f"Kalshi ({market_ticker}): Sent subscription (ID: {current_command_id}): {json.dumps(subscribe_cmd)}")

                async for message_str in websocket:
                    logger.debug(f"Kalshi ({market_ticker}) RAW: {message_str}") 
                    try:
                        message = json.loads(message_str)
                        msg_type = message.get("type")
                        
                        if msg_type == "subscribed":
                            logger.info(f"Kalshi ({market_ticker}): Subscribed successfully. SID: {message.get('sid')}, CmdID: {message.get('id')}")
                            current_market_state.sequence_num = 0 
                        
                        elif msg_type == "orderbook_snapshot":
                            data = message.get("msg", {})
                            seq = message.get("seq", 0)
                            logger.debug(f"Kalshi ({market_ticker}): Snapshot received (Seq: {seq}).")
                            
                            current_market_state.yes_book = {int(lvl[0]): int(lvl[1]) for lvl in data.get("yes", [])}
                            current_market_state.no_book = {int(lvl[0]): int(lvl[1]) for lvl in data.get("no", [])}
                            current_market_state.sequence_num = seq
                            current_market_state.timestamp_utc = datetime.now(timezone.utc)
                            current_market_state.update_ui_bid_ask()
                            logger.debug(f"Kalshi ({market_ticker}) SNAPSHOT: Yes Bid: {current_market_state.ui_yes_bid_cents}c ({current_market_state.ui_yes_bid_qty}), Yes Ask: {current_market_state.ui_yes_ask_cents}c ({current_market_state.ui_yes_ask_qty_on_no_side}) | Seq: {seq}")

                        elif msg_type == "orderbook_delta":
                            data = message.get("msg", {})
                            seq = message.get("seq", 0)
                            
                            if seq != current_market_state.sequence_num + 1:
                                logger.warning(f"Kalshi ({market_ticker}): Sequence gap! Expected {current_market_state.sequence_num + 1}, got {seq}. Closing to resync.")
                                await websocket.close(code=1000, reason="Resync sequence gap")
                                break 

                            current_market_state.sequence_num = seq
                            price = data.get("price")
                            delta_val = data.get("delta")
                            side_of_book = data.get("side") 

                            if side_of_book == "yes": book_to_update = current_market_state.yes_book
                            elif side_of_book == "no": book_to_update = current_market_state.no_book
                            else:
                                logger.warning(f"Kalshi ({market_ticker}): Unknown delta side: {side_of_book}")
                                continue
                            
                            new_quantity = book_to_update.get(price, 0) + delta_val
                            if new_quantity > 0: book_to_update[price] = new_quantity
                            else: book_to_update.pop(price, None)
                            
                            current_market_state.timestamp_utc = datetime.now(timezone.utc)
                            current_market_state.update_ui_bid_ask()
                            logger.debug(f"Kalshi ({market_ticker}) DELTA: Yes Bid: {current_market_state.ui_yes_bid_cents}c ({current_market_state.ui_yes_bid_qty}), Yes Ask: {current_market_state.ui_yes_ask_cents}c ({current_market_state.ui_yes_ask_qty_on_no_side}) | Seq: {seq}, Δ {side_of_book.upper()}@{price}c by {delta_val}")

                        elif msg_type == "error":
                            err_msg = message.get("msg", {})
                            cmd_id_resp = message.get("id")
                            logger.error(f"Kalshi ({market_ticker}): WS Error (CmdID Ref: {cmd_id_resp}): Code {err_msg.get('code')} - {err_msg.get('msg')}")
                            if err_msg.get('code') == 6: 
                                logger.info(f"Kalshi ({market_ticker}): 'Already subscribed'. Assuming active.")
                    
                    except json.JSONDecodeError:
                        logger.error(f"Kalshi ({market_ticker}): JSON Decode Error - {message_str}")
                    except Exception as e:
                        logger.exception(f"Kalshi ({market_ticker}): Error processing message - {e} - Message: {message_str}")
        
        except (websockets.exceptions.ConnectionClosed, ConnectionRefusedError, websockets.exceptions.WebSocketException, asyncio.TimeoutError) as e:
            logger.error(f"Kalshi ({market_ticker}): WS connection issue: {type(e).__name__} - {e}. Retrying...")
        except Exception as e:
            logger.exception(f"Kalshi ({market_ticker}): Unexpected WS connection error: {e}. Retrying...")
        
        logger.info(f"Kalshi ({market_ticker}): Disconnected.")
        if market_ticker in shared_market_states: 
            del shared_market_states[market_ticker]
        
        if on_disconnect_callback: 
            task_still_active = market_ticker in g_active_kalshi_ws_tasks and \
                                g_active_kalshi_ws_tasks.get(market_ticker) is asyncio.current_task() and \
                                not g_active_kalshi_ws_tasks[market_ticker].done() # type: ignore
            if task_still_active:
                 await on_disconnect_callback(market_ticker) 

        now_utc = datetime.now(timezone.utc)
        if now_utc >= market_info.close_time_utc + timedelta(minutes=1): # Add buffer for market truly being done
            logger.info(f"Kalshi ({market_ticker}): Market has closed (or passed close_time). Stopping listener task for it.")
            break 

        retry_delay = 10 if connection_established_once else 30
        logger.info(f"Kalshi ({market_ticker}): Retrying connection in {retry_delay} seconds.")
        await asyncio.sleep(retry_delay)
    
    # Clean up task reference from global dict if this task instance is exiting
    current_task_obj = asyncio.current_task()
    if market_ticker in g_active_kalshi_ws_tasks and g_active_kalshi_ws_tasks.get(market_ticker) is current_task_obj:
        del g_active_kalshi_ws_tasks[market_ticker]
    logger.info(f"Kalshi ({market_ticker}): Listener task terminated.")


async def start_kalshi_market_stream(
    market_info: KalshiMarketInfo,
    shared_market_states: Dict[str, KalshiMarketState],
    on_disconnect_callback: Callable[[str], Awaitable[None]]
):
    market_ticker = market_info.ticker
    existing_task = g_active_kalshi_ws_tasks.get(market_ticker)
    if not existing_task or existing_task.done():
        if existing_task and existing_task.done():
             logger.info(f"StreamManager: Previous task for {market_ticker} was done. Cleaning up before restart.")
             try: await existing_task # Await a done task to retrieve exceptions if any
             except asyncio.CancelledError: pass # Expected
             except Exception as e: logger.error(f"StreamManager: Exception from prev done task {market_ticker}: {e}")
        
        logger.info(f"StreamManager: Starting stream for Kalshi market {market_ticker}")
        task = asyncio.create_task(
            kalshi_market_stream_listener(market_info, shared_market_states, on_disconnect_callback),
            name=f"KalshiListener-{market_ticker}" 
        )
        g_active_kalshi_ws_tasks[market_ticker] = task
    else:
        logger.debug(f"StreamManager: Stream for Kalshi market {market_ticker} already active.")


async def stop_kalshi_market_stream(market_ticker: str):
    if market_ticker in g_active_kalshi_ws_tasks:
        task = g_active_kalshi_ws_tasks.pop(market_ticker) 
        logger.info(f"StreamManager: Attempting to stop stream for Kalshi market {market_ticker}.")
        if task and not task.done():
            task.cancel()
            try:
                await asyncio.wait_for(task, timeout=5.0) # Give it some time to cancel
            except asyncio.CancelledError:
                logger.info(f"StreamManager: Successfully cancelled stream for Kalshi market {market_ticker}.")
            except asyncio.TimeoutError:
                logger.warning(f"StreamManager: Timeout waiting for stream task {market_ticker} to cancel.")
            except Exception as e:
                logger.error(f"StreamManager: Error during cancellation of stream for {market_ticker}: {e}")
        else:
             logger.info(f"StreamManager: Stream for Kalshi market {market_ticker} was already done or not found.")
        if market_ticker in g_kalshi_market_states: # Clean up shared state
            del g_kalshi_market_states[market_ticker]


async def update_active_kalshi_streams(
    discovered_markets: List[KalshiMarketInfo],
    shared_market_states: Dict[str, KalshiMarketState],
    on_kalshi_market_disconnect: Callable[[str], Awaitable[None]]
):
    discovered_tickers = {m.ticker for m in discovered_markets}
    current_streaming_tickers = {
        ticker for ticker, task in g_active_kalshi_ws_tasks.items() if task and not task.done()
    }

    for market_info in discovered_markets:
        if market_info.ticker not in current_streaming_tickers:
            active_tasks_count = sum(1 for task in g_active_kalshi_ws_tasks.values() if task and not task.done())
            if active_tasks_count < cfg.MAX_MARKETS_TO_MONITOR:
                await start_kalshi_market_stream(market_info, shared_market_states, on_kalshi_market_disconnect)
                await asyncio.sleep(0.5) # Stagger new connection attempts slightly (0.2 to 1.0s)
            else:
                logger.warning(f"Reached max Kalshi market monitor limit ({cfg.MAX_MARKETS_TO_MONITOR}). Not starting stream for {market_info.ticker}")
                break # Stop trying to add more if limit reached

    tickers_to_stop = current_streaming_tickers - discovered_tickers
    for ticker in tickers_to_stop:
        logger.info(f"Market {ticker} no longer in NTM/active discovered list. Stopping its stream.")
        await stop_kalshi_market_stream(ticker)