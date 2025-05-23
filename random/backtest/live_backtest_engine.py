# /random/backtest/live_backtest_engine.py

import pandas as pd
import numpy as np
import datetime as dt
from datetime import timezone
import logging
import math

# Import from live_backtest_config and live_backtest_data_utils
try:
    from . import live_backtest_config as live_cfg
    from . import live_backtest_data_utils as live_data_utils
except ImportError: # For direct script execution if needed
    import live_backtest_config as live_cfg
    import live_backtest_data_utils as live_data_utils


def calculate_kelly_fraction(prob_win: float, bet_price_cents: int) -> float:
    """
    Calculates the Kelly fraction.
    prob_win: Model's probability of winning this specific bet (e.g., P(Yes) for a YES bet).
    bet_price_cents: The cost of one contract in cents.
    """
    if not (0 < bet_price_cents < 100): # Price must be between 1 and 99 cents
        return 0.0 
    if not (0 < prob_win < 1): # Probability must be valid
        return 0.0

    p = prob_win
    q = 1.0 - p
    
    # Odds b = (potential profit) / (amount risked)
    # Amount risked = bet_price_cents
    # Potential profit if win = 100 - bet_price_cents
    b = (100.0 - bet_price_cents) / bet_price_cents
    
    if b <= 0: # Should not happen if bet_price_cents < 100
        return 0.0

    f_star = (b * p - q) / b
    return max(0.0, f_star) # Kelly fraction cannot be negative


def run_live_data_backtest(all_features_df: pd.DataFrame, model, scaler, model_feature_names: list):
    engine_logger = logging.getLogger("live_backtest_engine")
    engine_logger.info("Starting LIVE DATA backtest engine (with Kelly Sizing if enabled)...")

    if all_features_df.empty:
        engine_logger.warning("No features provided to the live backtest engine. Exiting.")
        return 0, 0, pd.DataFrame(), 0 # pnl, trades, trade_log_df, final_capital

    all_features_df.sort_values(by='decision_timestamp_s', inplace=True)

    num_total_decision_points_to_simulate = len(all_features_df)
    engine_logger.info(f"Processing {num_total_decision_points_to_simulate} decision points from live data.")

    current_trade_log_file = None
    current_log_date_str = None

    # Initialize capital for Kelly sizing
    current_capital_cents = live_cfg.INITIAL_CAPITAL_CENTS if live_cfg.USE_KELLY_CRITERION else 0
    starting_capital_cents = current_capital_cents

    overall_pnl_cents = 0 # This will track P&L irrespective of Kelly's capital changes for comparison
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

        decision_date_obj = dt.datetime.fromtimestamp(decision_timestamp_s, tz=timezone.utc)
        decision_date_str = decision_date_obj.strftime("%Y-%m-%d")

        if decision_date_str != current_log_date_str:
            current_trade_log_file = live_data_utils.setup_daily_trade_logger(decision_date_str, live_cfg.LOG_DIR)
            current_log_date_str = decision_date_str
            engine_logger.info(f"Logging live trades for decisions made on {current_log_date_str} to {current_trade_log_file}")

        if live_cfg.ONE_BET_PER_KALSHI_MARKET and market_ticker in active_market_bets:
            continue

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
            
        action = "HOLD"
        bet_cost_per_contract_cents = 0
        pnl_for_this_trade_cents = 0
        contracts_traded = 0
        prob_of_winning_bet = 0.0
        kelly_f_star = 0.0
        capital_to_risk_cents = 0
        trade_value_cents = 0 # Total value of contracts bought

        capital_before_trade_cents = current_capital_cents

        # --- Determine Action and Bet Price (per contract) ---
        if predicted_proba_yes > live_cfg.PROBABILITY_THRESHOLD_YES:
            action = "BUY_YES"
            prob_of_winning_bet = predicted_proba_yes
            # Assume fill at model's perceived fair price
            bet_cost_per_contract_cents = int(round(predicted_proba_yes * 100))
        
        elif (1 - predicted_proba_yes) > live_cfg.PROBABILITY_THRESHOLD_NO:
            action = "BUY_NO"
            prob_of_winning_bet = 1 - predicted_proba_yes
            bet_cost_per_contract_cents = int(round((1 - predicted_proba_yes) * 100))
        
        # Ensure bet_cost_per_contract_cents is valid (1-99)
        if not (1 <= bet_cost_per_contract_cents <= 99):
            action = "HOLD" # Cannot trade at 0 or 100 cents effectively, or if thresholds are too loose

        # --- Sizing Logic ---
        if action != "HOLD":
            if live_cfg.USE_KELLY_CRITERION:
                if current_capital_cents <= 0: # Can't bet if bankrupt
                    engine_logger.warning(f"Kelly Sizing: Capital is {current_capital_cents}, cannot place trade for {market_ticker}")
                    action = "HOLD"
                else:
                    kelly_f_star = calculate_kelly_fraction(prob_of_winning_bet, bet_cost_per_contract_cents)
                    target_risk_fraction = kelly_f_star * live_cfg.KELLY_FRACTION
                    
                    # Apply cap based on MAX_PCT_CAPITAL_PER_TRADE
                    target_risk_fraction = min(target_risk_fraction, live_cfg.MAX_PCT_CAPITAL_PER_TRADE)
                    
                    capital_to_risk_cents = int(round(current_capital_cents * target_risk_fraction))
                    
                    if bet_cost_per_contract_cents > 0:
                        contracts_traded = math.floor(capital_to_risk_cents / bet_cost_per_contract_cents)
                    else:
                        contracts_traded = 0

                    # Apply min/max contract limits
                    contracts_traded = max(live_cfg.MIN_CONTRACTS_TO_TRADE if capital_to_risk_cents > bet_cost_per_contract_cents else 0, contracts_traded) # ensure enough capital for min contracts
                    contracts_traded = min(contracts_traded, live_cfg.MAX_CONTRACTS_TO_TRADE)
                    
                    # Ensure we don't risk more than available capital even with rounding
                    if (contracts_traded * bet_cost_per_contract_cents) > current_capital_cents :
                        contracts_traded = math.floor(current_capital_cents / bet_cost_per_contract_cents)

                    if contracts_traded < live_cfg.MIN_CONTRACTS_TO_TRADE: # Final check
                        action = "HOLD" # Not enough contracts to make it worthwhile or affordable
                        contracts_traded = 0
                    else:
                        trade_value_cents = contracts_traded * bet_cost_per_contract_cents
            else: # Fixed 1 contract if not using Kelly
                contracts_traded = 1
                trade_value_cents = contracts_traded * bet_cost_per_contract_cents

        # --- P&L Calculation ---
        if action != "HOLD" and contracts_traded > 0:
            pnl_per_contract_cents = 0
            if action == "BUY_YES":
                if actual_market_outcome == 1: pnl_per_contract_cents = 100 - bet_cost_per_contract_cents; trade_stats['yes_won'] += contracts_traded
                else: pnl_per_contract_cents = -bet_cost_per_contract_cents; trade_stats['yes_lost'] += contracts_traded
            elif action == "BUY_NO":
                if actual_market_outcome == 0: pnl_per_contract_cents = 100 - bet_cost_per_contract_cents; trade_stats['no_won'] += contracts_traded
                else: pnl_per_contract_cents = -bet_cost_per_contract_cents; trade_stats['no_lost'] += contracts_traded
            
            pnl_for_this_trade_cents = pnl_per_contract_cents * contracts_traded
            overall_pnl_cents += pnl_for_this_trade_cents # Track simple P&L

            if live_cfg.USE_KELLY_CRITERION:
                current_capital_cents += pnl_for_this_trade_cents # Update capital based on Kelly-sized trade
        
        # --- Logging Trade ---
        if action != "HOLD":
            if live_cfg.ONE_BET_PER_KALSHI_MARKET: active_market_bets[market_ticker] = action
            
            trade_log_entry_dict = {
                "trade_execution_time_utc": decision_date_obj.isoformat(),
                "market_ticker": market_ticker, "strike_price": strike_price,
                "resolution_time_ts": resolution_time_ts, "decision_timestamp_s": decision_timestamp_s,
                "time_to_resolution_minutes": f"{time_to_resolution_min:.2f}", "action": action,
                "predicted_prob_yes": f"{predicted_proba_yes:.4f}",
                "bet_cost_cents_per_contract": bet_cost_per_contract_cents,
                "contracts_traded": contracts_traded,
                "actual_outcome_target": actual_market_outcome,
                "pnl_cents": pnl_for_this_trade_cents,
                "current_capital_before_trade_cents": capital_before_trade_cents if live_cfg.USE_KELLY_CRITERION else 0,
                "kelly_fraction_f_star": f"{kelly_f_star:.4f}" if live_cfg.USE_KELLY_CRITERION else 0,
                "capital_to_risk_cents": capital_to_risk_cents if live_cfg.USE_KELLY_CRITERION else 0,
                "trade_value_cents": trade_value_cents
            }
            all_trades_log_list.append(trade_log_entry_dict)

            trade_log_entry_csv_str = (
                f"{decision_date_obj.isoformat()},"
                f"{market_ticker},{strike_price},{resolution_time_ts},"
                f"{decision_timestamp_s},{time_to_resolution_min:.2f},"
                f"{action},{predicted_proba_yes:.4f},{bet_cost_per_contract_cents},{contracts_traded},"
                f"{actual_market_outcome},{pnl_for_this_trade_cents},"
                f"{capital_before_trade_cents if live_cfg.USE_KELLY_CRITERION else ''},"
                f"{kelly_f_star:.4f} if live_cfg.USE_KELLY_CRITERION else '',"
                f"{capital_to_risk_cents if live_cfg.USE_KELLY_CRITERION else ''},"
                f"{trade_value_cents}\n"
            )
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
    engine_logger.info(f"Total P&L (Simple Sum of Trade P&Ls): {overall_pnl_cents / 100:.2f} USD") # For comparison
    
    total_yes_bets = trade_stats['yes_won'] + trade_stats['yes_lost']
    total_no_bets = trade_stats['no_won'] + trade_stats['no_lost']
    engine_logger.info(f"YES Bets: {trade_stats['yes_won']} Won Contracts, {trade_stats['yes_lost']} Lost Contracts (Total: {total_yes_bets} contracts)")
    engine_logger.info(f"NO Bets: {trade_stats['no_won']} Won Contracts, {trade_stats['no_lost']} Lost Contracts (Total: {total_no_bets} contracts)")
    total_contracts_traded = sum(trade_stats.values())
    engine_logger.info(f"Total Contracts Traded: {total_contracts_traded}")
    
    trade_log_df = pd.DataFrame(all_trades_log_list)
    final_capital = current_capital_cents if live_cfg.USE_KELLY_CRITERION else (starting_capital_cents + overall_pnl_cents)
    return overall_pnl_cents, total_contracts_traded, trade_log_df, final_capital