# /random/backtest/live_backtest_main.py

import pandas as pd
import logging
from pathlib import Path
import time

# Changed from relative to direct import
import live_backtest_config as live_cfg
import live_backtest_data_utils as live_data_utils
import live_backtest_feature_eng as live_feature_eng
import live_backtest_engine as live_engine

main_logger = logging.getLogger("live_backtest_main")
main_logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
ch = logging.StreamHandler()
ch.setFormatter(formatter)
if not any(isinstance(h, logging.StreamHandler) for h in main_logger.handlers):
    main_logger.addHandler(ch)

def main():
    main_logger.info("--- Starting Live Data Backtest Simulation (with Kelly Sizing if enabled) ---")
    try:
        model, scaler, model_feature_names = live_data_utils.load_model_and_dependencies()
        main_logger.info(f"Loaded model expecting {len(model_feature_names)} features.")
    except Exception as e:
        main_logger.critical(f"Failed to load model components: {e}. Aborting.", exc_info=True)
        return
    live_outcomes_df = live_data_utils.load_live_market_outcomes()
    if live_outcomes_df is None or live_outcomes_df.empty:
        main_logger.critical("No live market outcomes loaded. Aborting.")
        return
    main_logger.info(f"Loaded {len(live_outcomes_df)} markets from live outcomes manifest.")
    all_markets_features_list = []
    live_data_utils.clear_live_data_caches() 
    total_markets = len(live_outcomes_df)
    processed_count = 0
    for index, market_row in live_outcomes_df.iterrows():
        processed_count += 1; market_ticker = market_row['market_ticker']
        main_logger.info(f"[{processed_count}/{total_markets}] Generating features for market: {market_ticker}")
        try:
            market_features_df = live_feature_eng.generate_features_for_market_live(market_row,model_feature_names)
            if not market_features_df.empty: all_markets_features_list.append(market_features_df)
        except Exception as e: main_logger.error(f"Error generating features for {market_ticker}: {e}", exc_info=False)
    if not all_markets_features_list:
        main_logger.critical("No features were generated for any market. Aborting backtest.")
        return
    all_features_to_backtest_df = pd.concat(all_markets_features_list, ignore_index=True)
    main_logger.info(f"Total decision points across all live markets initially: {len(all_features_to_backtest_df)}")
    cols_to_check = ['market_ticker', 'decision_timestamp_s', 'resolution_time_ts', 'target', 'strike_price', 'time_to_resolution_minutes']
    for col in model_feature_names: 
        if col not in cols_to_check: cols_to_check.append(col)
    nan_check = all_features_to_backtest_df[cols_to_check].isnull().sum()
    if nan_check.sum() > 0:
        main_logger.warning(f"NaNs found in combined feature DataFrame before engine run:\n{nan_check[nan_check > 0]}")
        main_logger.info("Attempting to drop rows with NaNs in model features before backtesting...")
        all_features_to_backtest_df.dropna(subset=model_feature_names, inplace=True)
        main_logger.info(f"Shape after dropping NaN rows from model features: {all_features_to_backtest_df.shape}")
        if all_features_to_backtest_df.empty:
            main_logger.critical("All rows dropped due to NaNs in model features. Cannot proceed.")
            return
    else: main_logger.info("No NaNs found in essential/model columns of combined feature DataFrame.")
    if not all_features_to_backtest_df.empty:
        main_logger.info("Proceeding to run the live backtest engine...")
        try:
            simple_sum_pnl_cents, total_contracts_traded, trades_log_df, final_capital_cents = live_engine.run_live_data_backtest(all_features_to_backtest_df,model,scaler,model_feature_names)
            main_logger.info("--- Live Data Backtest Execution Finished ---")
            if live_cfg.USE_KELLY_CRITERION:
                main_logger.info(f"Final Capital (Kelly): {final_capital_cents / 100.0 :.2f} USD (Started with: {live_cfg.INITIAL_CAPITAL_CENTS / 100.0:.2f} USD)")
                kelly_return_pct = ((final_capital_cents - live_cfg.INITIAL_CAPITAL_CENTS) / live_cfg.INITIAL_CAPITAL_CENTS) * 100 if live_cfg.INITIAL_CAPITAL_CENTS > 0 else 0
                main_logger.info(f"Overall Return on Initial Capital (Kelly): {kelly_return_pct:.2f}%")
            main_logger.info(f"Overall P&L (Simple Sum of Trades): {simple_sum_pnl_cents / 100.0 :.2f} USD")
            main_logger.info(f"Total contracts traded: {total_contracts_traded}")
            main_logger.info(f"Daily trade logs are in subdirectories of: {live_cfg.LOG_DIR}") 
            if not trades_log_df.empty:
                consolidated_log_path = live_cfg.LOG_DIR / f"all_live_trades_summary_kelly_{time.strftime('%Y%m%d_%H%M%S')}.csv"
                trades_log_df.to_csv(consolidated_log_path, index=False)
                main_logger.info(f"Consolidated trade log saved to: {consolidated_log_path}")
            else: main_logger.info("No trades were made, so no consolidated log to save.")
        except Exception as e: main_logger.error(f"An error occurred during run_live_data_backtest: {e}", exc_info=True)
    else: main_logger.error("Feature DataFrame is empty. Cannot run live backtest engine.")
    main_logger.info("--- Live Data Backtest Simulation Complete ---")

if __name__ == "__main__":
    main()