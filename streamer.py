import asyncio
import websockets
import json
import time
from datetime import datetime

# Configuration
STREAMS_TO_SUBSCRIBE = [
    # "ethusdt@miniTicker",
    # "btcusdt@miniTicker",
    "ethusdt@kline_1m",  # 1-minute candlesticks for ETH/USDT
    "btcusdt@kline_1m"   # 1-minute candlesticks for BTC/USDT
    # For even more granularity (consider data volume):
    # "ethusdt@kline_1s",
    # "btcusdt@kline_1s",
    # "ethusdt@trade",
    # "btcusdt@trade",
]

BINANCE_WS_ENDPOINT = "wss://data-stream.binance.vision/stream"
SUBSCRIPTION_ID_COUNTER = 1

# Store latest kline data (optional, for more complex logic)
latest_klines = {}

async def binance_websocket_client():
    global SUBSCRIPTION_ID_COUNTER, latest_klines
    uri = BINANCE_WS_ENDPOINT

    async with websockets.connect(uri) as websocket:
        print(f"Connected to Binance WebSocket at {uri}")

        subscribe_payload = {
            "method": "SUBSCRIBE",
            "params": STREAMS_TO_SUBSCRIBE,
            "id": SUBSCRIPTION_ID_COUNTER
        }
        SUBSCRIPTION_ID_COUNTER += 1
        await websocket.send(json.dumps(subscribe_payload))
        print(f"Sent subscription request: {json.dumps(subscribe_payload)}")

        print("\n--- Listening for live data ---")
        while True:
            try:
                message_str = await websocket.recv()
                message_data = json.loads(message_str)

                if 'stream' in message_data and 'data' in message_data:
                    stream_name = message_data['stream']
                    data_payload = message_data['data']
                    event_type = data_payload.get('e')

                    current_time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

                    if event_type == '24hrMiniTicker':
                        symbol = data_payload['s']
                        last_price = data_payload['c']
                        print(f"{current_time_str} | MINI_TICKER | {symbol}: Last Price = {last_price}")

                    elif event_type == 'kline':
                        kline_data = data_payload['k']
                        symbol = kline_data['s']
                        interval = kline_data['i']
                        is_closed = kline_data['x']
                        open_price = kline_data['o']
                        close_price = kline_data['c'] # This updates with current price for unclosed kline
                        high_price = kline_data['h']
                        low_price = kline_data['l']
                        volume = kline_data['v']

                        # Store the latest kline data if you want to perform calculations on it
                        latest_klines[f"{symbol}_{interval}"] = kline_data

                        print(f"{current_time_str} | KLINE_{interval} | {symbol}: O={open_price}, H={high_price}, L={low_price}, C={close_price}, V={volume}, Closed={is_closed}")

                        # --- Your prediction logic would go here ---
                        # Example: Check momentum on the 1m kline
                        # if symbol == "ETHUSDT" and interval == "1m":
                        #     if float(close_price) > float(open_price):
                        #         print(f"    ETHUSDT 1m kline: Bullish momentum (C > O)")
                        #     elif float(close_price) < float(open_price):
                        #         print(f"    ETHUSDT 1m kline: Bearish momentum (C < O)")


                elif "result" in message_data and "id" in message_data:
                    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]} | Subscription response: {json.dumps(message_data)}")
                else:
                    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]} | Other message: {json.dumps(message_data)}")

            except websockets.exceptions.ConnectionClosedOK:
                print("Connection closed normally.")
                break
            except websockets.exceptions.ConnectionClosedError as e:
                print(f"Connection closed with error: {e} - Attempting to reconnect...")
                await asyncio.sleep(5)
                asyncio.run(binance_websocket_client())
                break
            except websockets.exceptions.ConnectionClosed as e:
                print(f"Connection closed unexpectedly: {e} - Attempting to reconnect...")
                await asyncio.sleep(5)
                asyncio.run(binance_websocket_client())
                break
            except json.JSONDecodeError as e:
                print(f"JSON Decode Error: {e} - Received: {message_str}")
            except Exception as e:
                print(f"An error occurred: {e} - Message: {message_str}")
                await asyncio.sleep(1)

if __name__ == "__main__":
    print("Starting Binance WebSocket client for ETH and BTC prices + Klines...")
    print(f"Attempting to subscribe to: {', '.join(STREAMS_TO_SUBSCRIBE)}")
    try:
        asyncio.run(binance_websocket_client())
    except KeyboardInterrupt:
        print("\nManually interrupted by user. Exiting.")
    except Exception as e:
        print(f"Main execution error: {e}")