# trader_core.py
import asyncio
import logging
import os
from datetime import datetime, timezone, timedelta

# Import from our other modules
import streamers # This will make latest_binance_data and latest_kalshi_books available
import model     # This provides calculate_p_model

# --- Trader Configuration ---
TARGET_KALSHI_ETH_TICKER = "KXETHD-25MAY1422-T2629.99" # ETH > $2630 @ 9PM EDT May 14
KALSHI_ETH_STRIKE_PRICE = 2629.99 # The value it needs to be strictly ">"
# Define expiry time for this contract in UTC
# 9 PM EDT on May 14, 2025. EDT is UTC-4.
# So, 9 PM EDT = 01:00 UTC on May 15, 2025
KALSHI_ETH_EXPIRY_DT_UTC = datetime(2025, 5, 15, 2, 0, 0, tzinfo=timezone.utc)
# If the contract is "May 14, 2025 · 9PM EDT", and it's an IN market (settles at 9PM)
# then 2025-05-14 21:00:00 EDT -> 2025-05-15 01:00:00 UTC


BINANCE_ETH_SYMBOL = "ETHUSDT"

MIN_EDGE_THRESHOLD = 0.05 # Minimum 5% edge needed to consider a paper trade
PAPER_TRADE_LOG_FILE = "paper_trades.log"

# --- Logging Setup ---
# Configure logging level and format for all modules
logging.basicConfig(
    level=logging.INFO, # Change to DEBUG for more verbosity from streamers/model
    format='%(asctime)s - %(levelname)s - %(module)s.%(funcName)s - %(message)s',
    handlers=[
        logging.StreamHandler(), # To console
        logging.FileHandler("trader_activity.log") # To a file
    ]
)
logger = logging.getLogger(__name__)


def log_paper_trade(market_ticker, side, price, contracts, p_model, p_market, reason):
    """Logs a paper trade decision."""
    log_entry = (
        f"{datetime.now(timezone.utc).isoformat()} | PAPER_TRADE | {market_ticker} | "
        f"Side: {side} | Price: {price}c | Contracts: {contracts} | "
        f"P_model: {p_model:.4f} | P_market: {p_market:.4f} | Reason: {reason}\n"
    )
    logger.info(f"PAPER TRADE: {market_ticker} - {side} at {price}c. P_model={p_model:.4f}, P_market={p_market:.4f}. Reason: {reason}")
    with open(PAPER_TRADE_LOG_FILE, "a") as f:
        f.write(log_entry)

async def trading_logic_loop():
    logger.info("Starting trading logic loop...")
    while True:
        await asyncio.sleep(10) # Decision frequency

        # --- Get ETH Data ---
        current_eth_kline = streamers.latest_binance_data.get(BINANCE_ETH_SYMBOL)
        kalshi_eth_book = streamers.latest_kalshi_books.get(TARGET_KALSHI_ETH_TICKER)

        if not current_eth_kline:
            logger.warning(f"No Binance data for {BINANCE_ETH_SYMBOL} yet.")
            continue
        if not kalshi_eth_book or not kalshi_eth_book.get("active"):
            logger.warning(f"No active Kalshi book for {TARGET_KALSHI_ETH_TICKER} yet.")
            continue

        try:
            current_eth_price = float(current_eth_kline.get('c')) # Close price of 1m kline
            # Crude volatility from 1m kline (High - Low) / Close
            # This is just a placeholder for a real volatility measure
            h = float(current_eth_kline.get('h'))
            l = float(current_eth_kline.get('l'))
            volatility_metric_placeholder = (h - l) / current_eth_price if current_eth_price > 0 else 0.01

        except (TypeError, ValueError) as e:
            logger.error(f"Could not parse Binance ETH price/volatility: {e}. Kline data: {current_eth_kline}")
            continue

        kalshi_yes_bid = kalshi_eth_book.get("ui_yes_bid")
        kalshi_yes_ask = kalshi_eth_book.get("ui_yes_ask")

        if kalshi_yes_bid is None or kalshi_yes_ask is None:
            logger.warning(f"Kalshi ETH book ({TARGET_KALSHI_ETH_TICKER}) missing bid/ask. YesBid: {kalshi_yes_bid}, YesAsk: {kalshi_yes_ask}")
            continue
        
        # --- Calculate P_model ---
        # Pass the Binance kline data as 'recent_volatility_data' for the model to potentially use
        p_model = model.calculate_p_model(
            underlying_asset_symbol=BINANCE_ETH_SYMBOL,
            current_asset_price=current_eth_price,
            kalshi_contract_strike_price=KALSHI_ETH_STRIKE_PRICE,
            kalshi_contract_expiry_dt_utc=KALSHI_ETH_EXPIRY_DT_UTC,
            recent_volatility_data=current_eth_kline # Pass whole kline dict
        )

        if p_model is None:
            logger.warning("P_model calculation failed.")
            continue

        # --- Decision Logic ---
        # Buy Yes if P_model > P_market_ask + edge
        cost_to_buy_yes = kalshi_yes_ask
        p_market_ask_yes = cost_to_buy_yes / 100.0
        
        if p_model > (p_market_ask_yes + MIN_EDGE_THRESHOLD):
            reason_buy_yes = f"P_model ({p_model:.4f}) > P_market_ask_yes ({p_market_ask_yes:.4f}) + Edge ({MIN_EDGE_THRESHOLD})"
            log_paper_trade(TARGET_KALSHI_ETH_TICKER, "BUY_YES", cost_to_buy_yes, 1, p_model, p_market_ask_yes, reason_buy_yes)
            # In real trading: await place_kalshi_order(...)
        
        # Buy No (Sell Yes) if (1 - P_model) > P_market_ask_no + edge
        # P_market_ask_no is the cost to buy a NO contract.
        # If we interpret selling YES at kalshi_yes_bid as buying NO:
        # Cost to "buy NO" (by selling YES) is (100 - kalshi_yes_bid)
        # So the market implies P(No) = (100 - kalshi_yes_bid) / 100.0
        # Or, P(Yes) from selling Yes perspective is kalshi_yes_bid / 100.0
        
        p_model_no = 1.0 - p_model
        
        # Cost to buy NO = Price of NO contract you pay.
        # Kalshi UI shows: "No" side with its own bid/ask.
        # "No Bid" Xc means you can sell No for Xc (same as buying Yes for 100-Xc)
        # "No Ask" Yc means you can buy No for Yc (same as selling Yes for 100-Yc)
        # From our derived ui_yes_bid/ask:
        #   ui_yes_bid is what you get if you sell Yes.
        #   ui_yes_ask is what you pay if you buy Yes.
        # So, if you want to BUY NO: you are effectively SELLING YES. The "price" for "NO" you are paying
        # is conceptually 100 - (price you get for selling Yes).
        # Or, more directly, if Kalshi shows No Ask X, you buy No at X.
        # Our current `latest_kalshi_books` has `ui_yes_bid` and `ui_yes_ask`.
        # `ui_yes_bid` is the highest price someone will pay for a YES contract.
        # `ui_yes_ask` is the lowest price someone will sell a YES contract.
        # If you want to BUY a NO contract: This is equivalent to someone SELLING you a NO contract.
        # This means they are BUYING a YES contract from you.
        # The price for this NO contract would be 100 - ui_yes_bid (what you'd get if you sold YES)
        
        cost_to_buy_no = (100 - kalshi_yes_bid) # This is the price you'd pay for a NO contract if you transact at the current best YES bid
        p_market_ask_no = cost_to_buy_no / 100.0
        
        if p_model_no > (p_market_ask_no + MIN_EDGE_THRESHOLD):
            reason_buy_no = f"P_model_No ({p_model_no:.4f}) > P_market_ask_no ({p_market_ask_no:.4f}) + Edge ({MIN_EDGE_THRESHOLD})"
            log_paper_trade(TARGET_KALSHI_ETH_TICKER, "BUY_NO", cost_to_buy_no, 1, p_model_no, p_market_ask_no, reason_buy_no)
            # In real trading: await place_kalshi_order(...)

async def main():
    logger.info("Initializing Trader Core...")
    streamers.init_config() # Load .env for streamers

    # --- Define Kalshi and Binance contracts/symbols of interest ---
    kalshi_eth_market_list = [TARGET_KALSHI_ETH_TICKER] 
    # Add BTC or other Kalshi markets here if needed:
    # TARGET_KALSHI_BTC_TICKER = "KXBTCD-25MAY1421-T103999.99" # Example
    # kalshi_market_list.append(TARGET_KALSHI_BTC_TICKER)

    binance_streams = [f"{BINANCE_ETH_SYMBOL.lower()}@kline_1m"]
    # Add BTC or other Binance streams:
    # BINANCE_BTC_SYMBOL = "BTCUSDT"
    # binance_streams.append(f"{BINANCE_BTC_SYMBOL.lower()}@kline_1m")

    # --- Start Data Streamers ---
    logger.info("Starting data streamers...")
    kalshi_streamer_task = asyncio.create_task(streamers.kalshi_websocket_client_task(kalshi_eth_market_list))
    binance_streamer_task = asyncio.create_task(streamers.binance_websocket_client_task(binance_streams))
    
    # --- Start Trading Logic ---
    trading_task = asyncio.create_task(trading_logic_loop())

    try:
        await asyncio.gather(kalshi_streamer_task, binance_streamer_task, trading_task)
    except KeyboardInterrupt:
        logger.info("Trader Core shutting down by user interrupt...")
    except Exception as e:
        logger.exception("Critical error in main gather: %s", e)
    finally:
        logger.info("Cancelling tasks...")
        if 'kalshi_streamer_task' in locals() and not kalshi_streamer_task.done(): kalshi_streamer_task.cancel()
        if 'binance_streamer_task' in locals() and not binance_streamer_task.done(): binance_streamer_task.cancel()
        if 'trading_task' in locals() and not trading_task.done(): trading_task.cancel()
        # Allow tasks to clean up
        await asyncio.sleep(1) 
        logger.info("Trader Core shutdown complete.")


if __name__ == "__main__":
    # Ensure .env file is correctly set up with KALSHI_API_KEY_ID, KALSHI_PRIVATE_KEY_PATH
    # and optionally KALSHI_DEMO_MODE, KALSHI_DEMO_API_KEY_ID, KALSHI_DEMO_PRIVATE_KEY_PATH
    if not (os.getenv("KALSHI_API_KEY_ID") and os.getenv("KALSHI_PRIVATE_KEY_PATH")):
        print("CRITICAL: KALSHI_API_KEY_ID or KALSHI_PRIVATE_KEY_PATH environment variables are not set.")
        print("Please create a .env file with these values.")
    elif not os.path.exists(os.getenv("KALSHI_PRIVATE_KEY_PATH")):
         print(f"CRITICAL: Kalshi private key file not found at path: {os.getenv('KALSHI_PRIVATE_KEY_PATH')}")
    else:
        asyncio.run(main())