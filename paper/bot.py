# /paper/bot.py

import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Deque
# import pandas as pd # Not strictly needed here anymore for prefill

import httpx 

import paper.config as cfg
from paper.utils import load_model_and_dependencies 
# Removed download_file_async, extract_zip_file, cleanup_temp_dir as they are not used with REST prefill
from paper.data_models import BinanceKline, KalshiMarketState, KalshiMarketInfo, PaperTrade 
from paper.stream_manager import (
    binance_stream_listener, 
    update_active_kalshi_streams,
    stop_kalshi_market_stream, 
    g_binance_kline_history, 
    g_kalshi_market_states,
    g_active_kalshi_ws_tasks 
)
from paper.market_discovery import fetch_active_markets, get_resolved_market_outcome
from paper.feature_engine import generate_live_features
from paper.trading_logic import get_trade_decision
from paper.portfolio_manager import PortfolioManager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(module)s.%(funcName)s:%(lineno)d - %(message)s')
logger = logging.getLogger("paper_bot_main")

MIN_KLINES_FOR_FEATURES = max(cfg.LAG_WINDOWS_MINUTES + cfg.ROLLING_WINDOWS_MINUTES, default=30) + 2

class PaperTradingBot:
    def __init__(self):
        self.model, self.scaler, self.model_feature_names = load_model_and_dependencies()
        self.portfolio_manager = PortfolioManager(cfg.INITIAL_PAPER_CAPITAL_CENTS)
        self.http_client = httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0)) 
        
        self.binance_kline_history: Deque[BinanceKline] = g_binance_kline_history
        self.kalshi_market_states: Dict[str, KalshiMarketState] = g_kalshi_market_states
        
        self.active_kalshi_market_info: Dict[str, KalshiMarketInfo] = {} 
        self.tasks = [] 
        self._stop_event = asyncio.Event()
        self._binance_data_sufficient_event = asyncio.Event()

    async def prefill_binance_kline_history_with_hashkey(self):
        """Fetches recent historical klines from HashKey REST API to pre-fill history."""
        num_klines_to_fetch = MIN_KLINES_FOR_FEATURES + 10 
        logger.info(f"Attempting to pre-fill Binance kline history with up to {num_klines_to_fetch} klines using HashKey API...")
        
        params = {
            "symbol": cfg.HASHKEY_PREFILL_SYMBOL, # e.g., "BTCUSDT" or "BTCUSDT-PERPETUAL"
            "interval": cfg.HASHKEY_PREFILL_INTERVAL, # e.g., "1min"
            "limit": num_klines_to_fetch 
        }
        # HashKey API might require endTime to get the *most recent* klines.
        # If endTime is not provided, it might return the oldest available.
        # Let's try without endTime first, as their example doesn't use it with limit.
        # If it returns old data, we'll need to add endTime = current_time_ms.
        
        try:
            response = await self.http_client.get(cfg.HASHKEY_API_KLINES_URL, params=params, timeout=20.0)
            logger.info(f"Prefill request to HashKey: {response.url}")
            response.raise_for_status()
            
            # HashKey API returns a list of lists directly if successful
            # If it's nested under a "data" key like their example, adjust here:
            # klines_data_from_api = response.json().get("data", []) 
            klines_data_from_api = response.json() # Assuming direct list of lists as per sample response

            if not isinstance(klines_data_from_api, list) or \
               (klines_data_from_api and not isinstance(klines_data_from_api[0], list)):
                logger.error(f"Unexpected response format from HashKey API. Expected list of lists. Got: {str(klines_data_from_api)[:200]}")
                return


            self.binance_kline_history.clear() 
            parsed_klines_for_deque: List[BinanceKline] = []

            for kline_item_list in klines_data_from_api:
                try:
                    kline_obj = BinanceKline.from_hashkey_api_list( # Use the new parser
                        kline_data_list=kline_item_list,
                        symbol=params["symbol"], # Pass symbol and interval for context
                        interval_str=params["interval"]
                    )
                    parsed_klines_for_deque.append(kline_obj)
                except Exception as e:
                    logger.error(f"Error parsing individual historical kline from HashKey API: {kline_item_list}, Error: {e}")
            
            # HashKey API typically returns newest first if `endTime` isn't used, or oldest first if `startTime` is used.
            # If using `limit` only, it's often newest first. Let's sort to be sure (oldest first for deque).
            parsed_klines_for_deque.sort(key=lambda k: k.kline_start_time) 
            
            for kline_obj in parsed_klines_for_deque: 
                 self.binance_kline_history.append(kline_obj)

            logger.info(f"Successfully pre-filled {len(self.binance_kline_history)} Binance klines from HashKey API. "
                        f"Oldest: {datetime.fromtimestamp(self.binance_kline_history[0].kline_start_time/1000, tz=timezone.utc).isoformat() if self.binance_kline_history else 'N/A'}, "
                        f"Newest: {datetime.fromtimestamp(self.binance_kline_history[-1].kline_start_time/1000, tz=timezone.utc).isoformat() if self.binance_kline_history else 'N/A'}")
            
            if len(self.binance_kline_history) >= MIN_KLINES_FOR_FEATURES:
                logger.info("Prefill successful via HashKey API: Sufficient kline data now available.")
                self._binance_data_sufficient_event.set()
            else:
                logger.warning(f"Prefill via HashKey API resulted in {len(self.binance_kline_history)} klines, "
                               f"less than required {MIN_KLINES_FOR_FEATURES}. WebSocket will need to fill the rest.")

        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error pre-filling Binance klines via HashKey API: Status {e.response.status_code} - {e.response.text}")
            logger.error(f"URL attempted: {e.request.url}")
            logger.warning("Prefill failed. History will rely solely on WebSocket.")
        except Exception as e:
            logger.error(f"Error pre-filling Binance kline history via HashKey API: {e}", exc_info=True)
            logger.warning("Prefill failed. History will rely solely on WebSocket.")

    async def _wait_for_sufficient_binance_data(self, timeout_seconds=60 * 5):
        # ... (This function remains the same)
        if self._binance_data_sufficient_event.is_set():
            logger.info("Sufficient Binance data already available (from pre-fill or previous check).")
            return

        logger.info(f"Waiting for WebSocket to ensure at least {MIN_KLINES_FOR_FEATURES} total closed Binance klines...")
        start_time = asyncio.get_event_loop().time()
        while not self._stop_event.is_set():
            closed_klines_in_history = [k for k in self.binance_kline_history if k.is_closed]
            if len(closed_klines_in_history) >= MIN_KLINES_FOR_FEATURES:
                logger.info(f"Sufficient Binance kline data confirmed ({len(closed_klines_in_history)} distinct closed klines via WebSocket/prefill).")
                self._binance_data_sufficient_event.set()
                return
            if asyncio.get_event_loop().time() - start_time > timeout_seconds:
                logger.error(f"Timeout waiting for sufficient Binance kline data via WebSocket after {timeout_seconds}s. Have {len(closed_klines_in_history)} closed klines.")
                self._binance_data_sufficient_event.set() 
                return 
            logger.debug(f"Waiting for WebSocket Binance data: Have {len(closed_klines_in_history)}/{MIN_KLINES_FOR_FEATURES} closed klines.")
            await asyncio.sleep(10) 
        logger.info("Stop event received while waiting for sufficient Binance data.")

    async def _kalshi_market_disconnect_handler(self, market_ticker: str):
        # ... (same)
        logger.warning(f"Bot noticed Kalshi market {market_ticker} disconnected (via callback).")
        if market_ticker in self.active_kalshi_market_info:
            del self.active_kalshi_market_info[market_ticker]

    async def _run_market_discovery_loop(self):
        # ... (same)
        logger.info("Starting market discovery loop...")
        await self._binance_data_sufficient_event.wait() 
        logger.info("Sufficient Binance data detected, proceeding with market discovery cycles.")
        while not self._stop_event.is_set():
            current_btc_price: Optional[float] = None
            if self.binance_kline_history:
                for kline in reversed(self.binance_kline_history): 
                    if kline.is_closed: 
                        current_btc_price = kline.close 
                        logger.debug(f"Market Discovery: Using BTC price {current_btc_price} from kline ending {datetime.fromtimestamp(kline.kline_end_time/1000, tz=timezone.utc)}")
                        break 
            if current_btc_price is None and self._binance_data_sufficient_event.is_set():
                logger.warning("Market Discovery: BTC price still unavailable for NTM selection even after wait period. Skipping NTM-specific selection this cycle.")
            elif current_btc_price is not None:
                 logger.info(f"Market Discovery: Using current BTC price {current_btc_price:.2f} for NTM selection.")
            try:
                discovered_raw = await fetch_active_markets(
                    self.http_client, cfg.TARGET_EVENT_SERIES_TICKER, current_btc_price 
                )
                new_active_market_info = {m.ticker: m for m in discovered_raw}
                current_active_tickers = set(self.active_kalshi_market_info.keys())
                discovered_tickers_set = set(new_active_market_info.keys())
                for ticker_to_prune in current_active_tickers - discovered_tickers_set:
                    if ticker_to_prune in self.active_kalshi_market_info:
                        del self.active_kalshi_market_info[ticker_to_prune]
                        logger.info(f"Pruned {ticker_to_prune} from internal NTM cache. Stopping its stream.")
                        await stop_kalshi_market_stream(ticker_to_prune)
                self.active_kalshi_market_info.update(new_active_market_info)
                await update_active_kalshi_streams(
                    list(self.active_kalshi_market_info.values()), self.kalshi_market_states, self._kalshi_market_disconnect_handler
                )
                active_ws_count = sum(1 for task in g_active_kalshi_ws_tasks.values() if task and not task.done())
                logger.info(f"Market discovery cycle complete. Now tracking {len(self.active_kalshi_market_info)} Kalshi NTM markets. Active WS: {active_ws_count}.")
            except Exception as e: logger.error(f"Error in market discovery loop: {e}", exc_info=True)
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=cfg.MARKET_DISCOVERY_INTERVAL_SECONDS)
                if self._stop_event.is_set(): break
            except asyncio.TimeoutError: pass 
    
    async def _run_trade_resolution_loop(self):
        # ... (same)
        logger.info("Starting trade resolution loop...")
        await self._binance_data_sufficient_event.wait() 
        while not self._stop_event.is_set():
            try:
                current_time_utc = datetime.now(timezone.utc)
                self.portfolio_manager.check_and_close_resolved_trades(
                    current_time_utc, lambda ticker: get_resolved_market_outcome(self.http_client, ticker)
                )
            except Exception as e: logger.error(f"Error in trade resolution loop: {e}", exc_info=True)
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=max(60, cfg.MARKET_DISCOVERY_INTERVAL_SECONDS // 2)) 
                if self._stop_event.is_set(): break
            except asyncio.TimeoutError: pass

    async def _run_trading_logic_loop(self):
        # ... (same)
        logger.info("Starting main trading logic loop...")
        await self._binance_data_sufficient_event.wait()
        logger.info("Trading logic: Sufficient Binance data available. Starting cycles.")
        while not self._stop_event.is_set():
            now = datetime.now(timezone.utc)
            wait_offset = 0.2 
            seconds_until_next_minute_aligned = (59 - now.second) + (1 - now.microsecond / 1_000_000.0) + wait_offset
            if seconds_until_next_minute_aligned < wait_offset: seconds_until_next_minute_aligned += 60
            logger.debug(f"Trading logic: Sleeping for {seconds_until_next_minute_aligned:.2f}s until aligned decision time.")
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=seconds_until_next_minute_aligned)
                if self._stop_event.is_set(): break
            except asyncio.TimeoutError: pass 
            decision_time_utc = datetime.now(timezone.utc).replace(second=0, microsecond=0)
            logger.info(f"--- Trading cycle for decision time: {decision_time_utc.isoformat()} ---")
            closed_klines_count = sum(1 for kline in self.binance_kline_history if kline.is_closed)
            if closed_klines_count < MIN_KLINES_FOR_FEATURES:
                logger.warning(f"Trading logic: Insufficient Binance kline history ({closed_klines_count}/{MIN_KLINES_FOR_FEATURES} closed klines). Skipping decisions this cycle.")
                continue
            active_market_tickers_to_consider = list(self.active_kalshi_market_info.keys())
            if not active_market_tickers_to_consider: logger.info("No active Kalshi NTM markets currently being tracked for trading decisions.")
            for market_ticker in active_market_tickers_to_consider:
                if self._stop_event.is_set(): break 
                market_info = self.active_kalshi_market_info.get(market_ticker)
                latest_kalshi_state = self.kalshi_market_states.get(market_ticker)
                if not market_info: 
                    logger.warning(f"Trading logic: Stale market info for {market_ticker}. Skipping.")
                    continue
                if not latest_kalshi_state:
                    logger.debug(f"Trading logic: No live Kalshi state for {market_ticker} yet. Skipping.")
                    continue
                time_to_res_seconds = (market_info.close_time_utc - decision_time_utc).total_seconds()
                if (time_to_res_seconds / 60.0) < cfg.MIN_MINUTES_BEFORE_RESOLUTION_FOR_DECISION:
                    logger.debug(f"Trading logic: Market {market_ticker} too close ({time_to_res_seconds/60.0:.1f}m). Skipping.")
                    continue
                if cfg.ONE_BET_PER_KALSHI_MARKET and self.portfolio_manager.get_open_trades_for_market(market_ticker):
                    logger.debug(f"Trading logic: Already open trade for {market_ticker}. Skipping.")
                    continue
                try:
                    features_dict = generate_live_features(
                        market_info=market_info, decision_ts_utc=decision_time_utc,
                        latest_kalshi_state=latest_kalshi_state, binance_kline_history=self.binance_kline_history,
                        model_feature_names=self.model_feature_names
                    )
                    if features_dict:
                        trade_decision_params = get_trade_decision(
                            features=features_dict, model=self.model, scaler=self.scaler,
                            model_feature_names=self.model_feature_names, portfolio_manager=self.portfolio_manager
                        )
                        if trade_decision_params:
                            action, entry_price_c, contracts, pred_prob_yes, kelly_f = trade_decision_params
                            logger.info(f"Decision for {market_ticker}: {action} {contracts} @ {entry_price_c}c (P(Yes)={pred_prob_yes:.4f}, KellyF*={kelly_f or 0:.4f})")
                            self.portfolio_manager.open_paper_trade(
                                market_ticker=market_ticker, decision_timestamp_utc=decision_time_utc, action=action,
                                predicted_prob_yes=pred_prob_yes, entry_price_cents=entry_price_c, contracts_traded=contracts,
                                resolution_time_utc=market_info.close_time_utc, kelly_f_star=kelly_f
                            )
                except Exception as e: logger.error(f"Error in trading logic for {market_ticker}: {e}", exc_info=True)
            if decision_time_utc.minute % 15 == 0: 
                 summary = self.portfolio_manager.get_summary()
                 logger.info(f"Portfolio Summary: {summary}")

    async def start(self):
        logger.info("Starting Paper Trading Bot...")
        self._stop_event.clear()
        self._binance_data_sufficient_event.clear()

        # *** MODIFIED: Call the new HashKey prefill method ***
        await self.prefill_binance_kline_history_with_hashkey() 

        initial_data_wait_task = asyncio.create_task(self._wait_for_sufficient_binance_data(), name="SufficientDataWait")
        self.tasks.append(initial_data_wait_task)
        binance_task = asyncio.create_task(
            binance_stream_listener(self.binance_kline_history, on_connect_callback=None), name="BinanceListener"
        )
        self.tasks.append(binance_task)
        discovery_task = asyncio.create_task(self._run_market_discovery_loop(), name="MarketDiscovery")
        self.tasks.append(discovery_task)
        resolution_task = asyncio.create_task(self._run_trade_resolution_loop(), name="TradeResolution")
        self.tasks.append(resolution_task)
        trading_task = asyncio.create_task(self._run_trading_logic_loop(), name="TradingLogic")
        self.tasks.append(trading_task)
        try:
            await asyncio.gather(*self.tasks)
        except Exception as e: 
            logger.critical(f"A critical task failed unexpectedly at gather level: {e}", exc_info=True)
            await self.stop() 

    async def stop(self):
        # ... (stop logic remains the same)
        logger.info("Stopping Paper Trading Bot...")
        self._stop_event.set() 
        active_kalshi_tickers = list(g_active_kalshi_ws_tasks.keys()) 
        for ticker in active_kalshi_tickers: await stop_kalshi_market_stream(ticker)
        await asyncio.sleep(2) 
        for task in self.tasks:
            if not task.done(): task.cancel()
        results = await asyncio.gather(*self.tasks, return_exceptions=True)
        for i, result in enumerate(results):
            if isinstance(result, Exception) and not isinstance(result, asyncio.CancelledError):
                task_name = f"Task-{i}"
                try: task_name = self.tasks[i].get_name()
                except: pass 
                logger.error(f"Error during task shutdown ({task_name}): {result}")
        if self.http_client: await self.http_client.aclose()
        logger.info("Paper Trading Bot stopped.")
        summary = self.portfolio_manager.get_summary()
        logger.info(f"Final Portfolio Summary: {summary}")

async def main():
    bot = PaperTradingBot()
    try:
        await bot.start()
    except KeyboardInterrupt: logger.info("KeyboardInterrupt received. Shutting down...")
    except Exception as e: logger.critical(f"Unhandled exception in main: {e}", exc_info=True)
    finally:
        logger.info("Attempting graceful shutdown...")
        await bot.stop()

if __name__ == "__main__":
    asyncio.run(main())