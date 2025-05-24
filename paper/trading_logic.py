# /paper/trading_logic.py

import logging
import pandas as pd
import numpy as np
import math
from typing import Dict, Any, Optional, Tuple, List

import paper.config as cfg
from paper.portfolio_manager import PortfolioManager # For Kelly sizing context

logger = logging.getLogger("paper_trading_logic")

# Kelly Criterion calculation (can be moved to utils if used elsewhere, but fits here for now)
def calculate_kelly_fraction(prob_win: float, bet_price_cents: int) -> float:
    """
    Calculates the Kelly fraction (f*).
    prob_win: Model's predicted probability of the contract settling at 100.
    bet_price_cents: Cost of the contract in cents (e.g., 60 for a 60c contract).
    """
    if not (0 < bet_price_cents < 100): # Price must be between 1 and 99 cents
        # logger.debug(f"Kelly: Invalid bet_price_cents: {bet_price_cents}")
        return 0.0
    if not (0 < prob_win < 1): # Probability must be between 0 and 1 (exclusive for strict Kelly)
        # logger.debug(f"Kelly: Invalid prob_win: {prob_win}")
        return 0.0

    p = prob_win  # Probability of winning (contract settles at 100)
    q = 1.0 - p   # Probability of losing (contract settles at 0)

    # 'b' is net odds received on the wager; payout (100-cost) / cost
    # If you pay `bet_price_cents` and win, you get 100 back. Profit is 100 - bet_price_cents.
    # Odds b = (Profit if Win) / (Amount Lost if Loss, which is bet_price_cents)
    b = (100.0 - bet_price_cents) / bet_price_cents
    
    if b <= 0: # Should not happen if bet_price_cents is < 100
        # logger.debug(f"Kelly: Invalid odds b: {b} from bet_price_cents: {bet_price_cents}")
        return 0.0

    f_star = (b * p - q) / b
    return max(0.0, f_star) # Kelly fraction cannot be negative

def get_trade_decision(
    features: Dict[str, Any], # Dictionary of features for one decision point
    model, # Trained sklearn model
    scaler, # Trained sklearn scaler
    model_feature_names: List[str],
    portfolio_manager: PortfolioManager # Needed for Kelly sizing
) -> Optional[Tuple[str, float, int, int, Optional[float]]]: # (action, entry_price_c, contracts, prob_yes, kelly_f*)
    """
    Makes a trading decision based on features and model prediction.
    Returns: (action, entry_price_cents, contracts_to_trade, predicted_prob_yes, kelly_f_star) or None
    """
    
    # 1. Prepare features for model input
    feature_values_for_model = []
    for f_name in model_feature_names:
        val = features.get(f_name)
        if pd.isna(val):
            logger.warning(f"NaN value for feature '{f_name}' in market {features.get('market_ticker', 'N/A')}. Cannot predict.")
            return None
        feature_values_for_model.append(val)
    
    if not feature_values_for_model:
        logger.error("No feature values collected. Cannot predict.")
        return None

    try:
        # Reshape for single sample prediction
        single_decision_point_features_np = np.array(feature_values_for_model).reshape(1, -1)
        scaled_features = scaler.transform(single_decision_point_features_np)
        predicted_proba_yes = model.predict_proba(scaled_features)[0, 1] # Prob of class '1' (YES)
    except Exception as e:
        logger.error(f"Error during scaling/prediction for {features.get('market_ticker', 'N/A')}: {e}", exc_info=True)
        return None

    # 2. Determine Action based on thresholds
    action: Optional[str] = None
    entry_price_cents: int = 0 # Cost of the contract in cents
    prob_of_winning_this_bet: float = 0.0

    # Model predicts P(Yes).
    # For BUY_YES, price is P(Yes)*100, prob_win is P(Yes)
    # For BUY_NO, price is P(No)*100, prob_win is P(No)
    
    prob_no = 1.0 - predicted_proba_yes

    if predicted_proba_yes > cfg.PROBABILITY_THRESHOLD_YES:
        action = "BUY_YES"
        # Entry price for BUY_YES is based on the current ask, or predicted_proba_yes if no live ask
        # For paper trading, we can assume we get filled at a price reflecting our prediction,
        # or use the latest Kalshi ask from features.
        # Using `predicted_proba_yes` makes it consistent with historical backtest bet_cost.
        entry_price_cents = int(round(predicted_proba_yes * 100))
        prob_of_winning_this_bet = predicted_proba_yes
        # More realistic: use features['current_kalshi_yes_ask'] * 100 if available and not NaN
        # live_ask_price_cents = features.get('current_kalshi_yes_ask', np.nan) * 100
        # if pd.notna(live_ask_price_cents) and 1 <= live_ask_price_cents <= 99:
        # entry_price_cents = int(round(live_ask_price_cents))


    elif prob_no > cfg.PROBABILITY_THRESHOLD_NO:
        action = "BUY_NO"
        entry_price_cents = int(round(prob_no * 100))
        prob_of_winning_this_bet = prob_no
        # More realistic: use (1 - features['current_kalshi_yes_bid']) * 100 if available
        # live_bid_price_cents = features.get('current_kalshi_yes_bid', np.nan) * 100
        # if pd.notna(live_bid_price_cents) and 1 <= live_bid_price_cents <= 99:
        #    entry_price_cents = int(round(100 - live_bid_price_cents))


    if action is None or not (1 <= entry_price_cents <= 99): # Invalid price, no action
        return None

    # 3. Determine Position Size
    contracts_to_trade = 0
    kelly_f_star: Optional[float] = None

    if cfg.USE_KELLY_CRITERION:
        current_capital_cents = portfolio_manager.get_current_capital_cents()
        if current_capital_cents <= 0:
            logger.info("Kelly Sizing: Zero or negative capital, cannot trade.")
            return None # No capital to trade

        kelly_f_star = calculate_kelly_fraction(prob_of_winning_this_bet, entry_price_cents)
        if kelly_f_star <= 0:
            # logger.debug(f"Kelly fraction is {kelly_f_star:.4f}, no edge. No trade.")
            return None

        target_risk_fraction = min(kelly_f_star * cfg.KELLY_FRACTION, cfg.MAX_PCT_CAPITAL_PER_TRADE)
        capital_to_risk_cents = int(round(current_capital_cents * target_risk_fraction))
        
        if entry_price_cents > 0:
            proposed_contracts = math.floor(capital_to_risk_cents / entry_price_cents)
        else:
            proposed_contracts = 0
        
        # Apply min/max contract constraints
        if proposed_contracts < cfg.MIN_CONTRACTS_TO_TRADE:
            # If capital_to_risk is high enough for min_contracts, trade min_contracts
            if capital_to_risk_cents >= (cfg.MIN_CONTRACTS_TO_TRADE * entry_price_cents):
                 contracts_to_trade = cfg.MIN_CONTRACTS_TO_TRADE
            else:
                # logger.debug(f"Kelly: Proposed contracts {proposed_contracts} < min {cfg.MIN_CONTRACTS_TO_TRADE} and not enough capital for min. No trade.")
                return None # Not enough edge or capital for min trade size
        else:
            contracts_to_trade = proposed_contracts
            
        contracts_to_trade = min(contracts_to_trade, cfg.MAX_CONTRACTS_TO_TRADE)

        # Final check: ensure total cost doesn't exceed available capital
        if (contracts_to_trade * entry_price_cents) > current_capital_cents:
            contracts_to_trade = math.floor(current_capital_cents / entry_price_cents)
            logger.info(f"Kelly: Adjusted contracts to {contracts_to_trade} due to capital limit.")

        if contracts_to_trade < cfg.MIN_CONTRACTS_TO_TRADE: # Check again after capital limit adjustment
            # logger.debug(f"Kelly: Final contracts {contracts_to_trade} < min {cfg.MIN_CONTRACTS_TO_TRADE}. No trade.")
            return None
    else: # Not using Kelly, trade fixed size (e.g., 1 contract for paper trading)
        contracts_to_trade = 1 # Default for non-Kelly paper trading
        # Ensure we can afford even 1 contract
        if (contracts_to_trade * entry_price_cents) > portfolio_manager.get_current_capital_cents():
            logger.info(f"Non-Kelly: Insufficient capital for 1 contract of {features.get('market_ticker', 'N/A')}.")
            return None


    if contracts_to_trade > 0:
        return action, entry_price_cents, contracts_to_trade, predicted_proba_yes, kelly_f_star
    else:
        # logger.debug(f"No contracts to trade for {features.get('market_ticker', 'N/A')} after sizing.")
        return None