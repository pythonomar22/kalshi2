# /random/backtest/live_backtest_engine.py

import pandas as pd
import numpy as np
import datetime as dt # <<<<<<<<<<< ENSURE THIS IS PRESENT if dt.datetime is used implicitly by other modules
from datetime import timezone # dt.datetime uses this timezone
import logging
import math

# Changed from relative to direct import
import live_backtest_config as live_cfg
import live_backtest_data_utils as live_data_utils # This now imports dt

def calculate_kelly_fraction(prob_win: float, bet_price_cents: int) -> float:
    if not (0 < bet_price_cents < 100): return 0.0 
    if not (0 < prob_win < 1): return 0.0
    p = prob_win; q = 1.0 - p
    b = (100.0 - bet_price_cents) / bet_price_cents
    if b <= 0: return 0.0
    f_star = (b * p - q) / b
    return max(0.0, f_star) 

def run_live_data_backtest(all_features_df: pd.DataFrame, model, scaler, model_feature_names: list):
    engine_logger = logging.getLogger("live_backtest_engine")
    engine_logger.info("Starting LIVE DATA backtest engine (with Kelly Sizing if enabled)...")

    if all_features_df.empty:
        engine_logger.warning("No features provided. Exiting.")
        return 0, 0, pd.DataFrame(), 0 

    all_features_df.sort_values(by='decision_timestamp_s', inplace=True)
    engine_logger.info(f"Processing {len(all_features_df)} decision points from live data.")

    current_trade_log_file = None
    current_log_date_hour_key = None 

    current_capital_cents = live_cfg.INITIAL_CAPITAL_CENTS if live_cfg.USE_KELLY_CRITERION else 0
    starting_capital_cents = current_capital_cents
    overall_pnl_cents = 0
    trade_stats = {'yes_won': 0, 'yes_lost': 0, 'no_won': 0, 'no_lost': 0}
    active_market_bets = {}
    all_trades_log_list = []

    for index, decision_point_row in all_features_df.iterrows():
        market_ticker = decision_point_row['market_ticker']
        decision_timestamp_s = decision_point_row['decision_timestamp_s']
        resolution_time_ts = decision_point_row['resolution_time_ts']
        strike_price = decision_point_row['strike_price'] 
        actual_market_outcome = decision_point_row['target']
        time_to_resolution_min = decision_point_row['time_to_resolution_minutes']

        decision_date_obj_utc = dt.datetime.fromtimestamp(decision_timestamp_s, tz=timezone.utc) # Uses dt
        
        log_key_for_this_decision = decision_date_obj_utc.strftime("%Y-%m-%d_%H")

        if log_key_for_this_decision != current_log_date_hour_key:
            current_trade_log_file = live_data_utils.setup_hourly_trade_logger(decision_date_obj_utc, live_cfg.LOG_DIR)
            current_log_date_hour_key = log_key_for_this_decision
            engine_logger.info(f"Logging live trades for decisions made on {decision_date_obj_utc.strftime('%Y-%m-%d %H')} to {current_trade_log_file}")

        if live_cfg.ONE_BET_PER_KALSHI_MARKET and market_ticker in active_market_bets: continue

        try:
            current_features_for_model = decision_point_row[model_feature_names].copy()
        except KeyError as e:
            engine_logger.error(f"KeyError for features of {market_ticker} at {decision_timestamp_s}: {e}.")
            continue
        
        if current_features_for_model.isnull().any():
            nan_cols = current_features_for_model[current_features_for_model.isnull()].index.tolist()
            engine_logger.warning(f"NaNs in features for {market_ticker} at {decision_timestamp_s}: {nan_cols}. Skipping.")
            continue
        
        single_decision_point_features_df = current_features_for_model.to_frame().T
        try:
            scaled_features = scaler.transform(single_decision_point_features_df)
            predicted_proba_yes = model.predict_proba(scaled_features)[0, 1]
        except Exception as e:
            engine_logger.error(f"Error scaling/predicting for {market_ticker} at {decision_timestamp_s}: {e}")
            continue
            
        action = "HOLD"; bet_cost_per_contract_cents=0; pnl_for_this_trade_cents=0; contracts_traded=0
        prob_of_winning_bet=0.0; kelly_f_star=0.0; capital_to_risk_cents=0; trade_value_cents=0
        capital_before_trade_cents = current_capital_cents

        if predicted_proba_yes > live_cfg.PROBABILITY_THRESHOLD_YES:
            action = "BUY_YES"; prob_of_winning_bet = predicted_proba_yes
            bet_cost_per_contract_cents = int(round(predicted_proba_yes * 100))
        elif (1 - predicted_proba_yes) > live_cfg.PROBABILITY_THRESHOLD_NO:
            action = "BUY_NO"; prob_of_winning_bet = 1 - predicted_proba_yes
            bet_cost_per_contract_cents = int(round((1 - predicted_proba_yes) * 100))
        
        if not (1 <= bet_cost_per_contract_cents <= 99): action = "HOLD"

        if action != "HOLD":
            if live_cfg.USE_KELLY_CRITERION:
                if current_capital_cents <= 0: action = "HOLD"
                else:
                    kelly_f_star = calculate_kelly_fraction(prob_of_winning_bet, bet_cost_per_contract_cents)
                    target_risk_fraction = min(kelly_f_star * live_cfg.KELLY_FRACTION, live_cfg.MAX_PCT_CAPITAL_PER_TRADE)
                    capital_to_risk_cents = int(round(current_capital_cents * target_risk_fraction))
                    contracts_traded = math.floor(capital_to_risk_cents / bet_cost_per_contract_cents) if bet_cost_per_contract_cents > 0 else 0
                    contracts_traded = max(live_cfg.MIN_CONTRACTS_TO_TRADE if capital_to_risk_cents > bet_cost_per_contract_cents else 0, contracts_traded)
                    contracts_traded = min(contracts_traded, live_cfg.MAX_CONTRACTS_TO_TRADE)
                    if (contracts_traded * bet_cost_per_contract_cents) > current_capital_cents : contracts_traded = math.floor(current_capital_cents / bet_cost_per_contract_cents)
                    if contracts_traded < live_cfg.MIN_CONTRACTS_TO_TRADE: action = "HOLD"; contracts_traded = 0
                    else: trade_value_cents = contracts_traded * bet_cost_per_contract_cents
            else: contracts_traded = 1; trade_value_cents = contracts_traded * bet_cost_per_contract_cents

        if action != "HOLD" and contracts_traded > 0:
            pnl_per_contract_cents = 0
            if action == "BUY_YES":
                if actual_market_outcome == 1: pnl_per_contract_cents = 100 - bet_cost_per_contract_cents; trade_stats['yes_won'] += contracts_traded
                else: pnl_per_contract_cents = -bet_cost_per_contract_cents; trade_stats['yes_lost'] += contracts_traded
            elif action == "BUY_NO":
                if actual_market_outcome == 0: pnl_per_contract_cents = 100 - bet_cost_per_contract_cents; trade_stats['no_won'] += contracts_traded
                else: pnl_per_contract_cents = -bet_cost_per_contract_cents; trade_stats['no_lost'] += contracts_traded
            pnl_for_this_trade_cents = pnl_per_contract_cents * contracts_traded
            overall_pnl_cents += pnl_for_this_trade_cents
            if live_cfg.USE_KELLY_CRITERION: current_capital_cents += pnl_for_this_trade_cents
        
        if action != "HOLD":
            if live_cfg.ONE_BET_PER_KALSHI_MARKET: active_market_bets[market_ticker] = action
            trade_log_entry_dict = {"trade_execution_time_utc":decision_date_obj_utc.isoformat(),"market_ticker":market_ticker,"strike_price":strike_price,"resolution_time_ts":resolution_time_ts,"decision_timestamp_s":decision_timestamp_s,"time_to_resolution_minutes":f"{time_to_resolution_min:.2f}","action":action,"predicted_prob_yes":f"{predicted_proba_yes:.4f}","bet_cost_cents_per_contract":bet_cost_per_contract_cents,"contracts_traded":contracts_traded,"actual_outcome_target":actual_market_outcome,"pnl_cents":pnl_for_this_trade_cents,"current_capital_before_trade_cents":capital_before_trade_cents if live_cfg.USE_KELLY_CRITERION else 0,"kelly_fraction_f_star":f"{kelly_f_star:.4f}" if live_cfg.USE_KELLY_CRITERION else 0,"capital_to_risk_cents":capital_to_risk_cents if live_cfg.USE_KELLY_CRITERION else 0,"trade_value_cents":trade_value_cents}
            all_trades_log_list.append(trade_log_entry_dict)
            trade_log_entry_csv_str = (f"{decision_date_obj_utc.isoformat()},{market_ticker},{strike_price},{resolution_time_ts},{decision_timestamp_s},{time_to_resolution_min:.2f},{action},{predicted_proba_yes:.4f},{bet_cost_per_contract_cents},{contracts_traded},{actual_market_outcome},{pnl_for_this_trade_cents},{capital_before_trade_cents if live_cfg.USE_KELLY_CRITERION else ''},{kelly_f_star:.4f} if live_cfg.USE_KELLY_CRITERION else '',{capital_to_risk_cents if live_cfg.USE_KELLY_CRITERION else ''},{trade_value_cents}\n")
            try:
                if current_trade_log_file:
                    with open(current_trade_log_file, 'a') as f: f.write(trade_log_entry_csv_str)
            except Exception as e: engine_logger.error(f"Error writing to trade log {current_trade_log_file}: {e}")

    engine_logger.info("Live data backtest engine finished processing.")
    engine_logger.info(f"--- Live Data Backtest Summary ---")
    if live_cfg.USE_KELLY_CRITERION:
        engine_logger.info(f"Starting Capital (Kelly): {starting_capital_cents / 100:.2f} USD")
        engine_logger.info(f"Ending Capital (Kelly): {current_capital_cents / 100:.2f} USD")
        kelly_pnl = (current_capital_cents - starting_capital_cents) / 100.0
        engine_logger.info(f"Total P&L (from Kelly Sized Capital Change): {kelly_pnl:.2f} USD")
    engine_logger.info(f"Total P&L (Simple Sum of Trade P&Ls): {overall_pnl_cents / 100:.2f} USD")
    total_yes_bets = trade_stats['yes_won'] + trade_stats['yes_lost']; total_no_bets = trade_stats['no_won'] + trade_stats['no_lost']
    engine_logger.info(f"YES Bets: {trade_stats['yes_won']} Won Contracts, {trade_stats['yes_lost']} Lost Contracts (Total: {total_yes_bets} contracts)")
    engine_logger.info(f"NO Bets: {trade_stats['no_won']} Won Contracts, {trade_stats['no_lost']} Lost Contracts (Total: {total_no_bets} contracts)")
    total_contracts_traded = sum(trade_stats.values())
    engine_logger.info(f"Total Contracts Traded: {total_contracts_traded}")
    trade_log_df = pd.DataFrame(all_trades_log_list)
    final_capital = current_capital_cents if live_cfg.USE_KELLY_CRITERION else (starting_capital_cents + overall_pnl_cents)
    return overall_pnl_cents, total_contracts_traded, trade_log_df, final_capital