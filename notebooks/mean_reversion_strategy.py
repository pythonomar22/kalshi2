# mean_reversion_strategy.py
import pandas as pd
import logging
import numpy as np

logger = logging.getLogger(__name__)

# --- Strategy Parameters ---
LOW_PRICE_THRESHOLD_CENTS = 20     # For buying YES: if yes_ask is below this
HIGH_PRICE_THRESHOLD_CENTS = 80    # For buying NO: if yes_bid is above this (meaning YES is overvalued)

# Max distance BTC can be from strike for a mean reversion trade
# e.g., if YES ask is 15c, but BTC is $500 below strike, maybe don't buy YES
MAX_ADVERSE_DISTANCE_TO_STRIKE_BUY_YES = -100 # BTC can be up to $100 below strike when buying low-priced YES
MAX_ADVERSE_DISTANCE_TO_STRIKE_BUY_NO = 100   # BTC can be up to $100 above strike when buying high-priced NO (i.e. selling YES)

MAX_BTC_VOLATILITY_THRESHOLD = 100 # Example: If 15-min BTC price stdev > $100, too volatile, no trade.

# Thresholds for taking action (these are more like hard gates for this strategy)
# The "score" will be more implicit based on meeting conditions.
# We can still generate a pseudo-score for sizing.
MODEL_SCORE_FOR_TRADE = 1.0 # A dummy score if conditions are met, to trigger base size
MODEL_SCORE_NO_TRADE = 0.0

def get_trade_decision(btc_features: dict, kalshi_strike_price: float, kalshi_prices: dict):
    """
    Determines trade action based on mean reversion logic.

    Args:
        btc_features (dict): Dictionary of BTC features including 'btc_price', 'btc_volatility_15m'.
        kalshi_strike_price (float): The strike price of the Kalshi market.
        kalshi_prices (dict): Dictionary with 'yes_bid' and 'yes_ask' for Kalshi market.

    Returns:
        tuple: (trade_action_str or None, model_prediction_score_float or None)
    """
    if btc_features is None or pd.isna(btc_features.get('btc_price')) or \
       pd.isna(btc_features.get('btc_volatility_15m')) or kalshi_prices is None or \
       pd.isna(kalshi_prices.get('yes_ask')) or pd.isna(kalshi_prices.get('yes_bid')):
        logger.debug("Missing necessary data for mean reversion strategy decision.")
        return None, None

    btc_price = btc_features['btc_price']
    btc_volatility = btc_features['btc_volatility_15m']
    yes_ask = kalshi_prices['yes_ask']
    yes_bid = kalshi_prices['yes_bid']
    distance_to_strike = btc_price - kalshi_strike_price

    trade_action = None
    model_score = MODEL_SCORE_NO_TRADE # Default to no trade score

    # Check volatility filter first
    if btc_volatility > MAX_BTC_VOLATILITY_THRESHOLD:
        logger.debug(f"BTC volatility {btc_volatility:.2f} exceeds threshold {MAX_BTC_VOLATILITY_THRESHOLD}. No mean reversion trade.")
        return None, None

    # Check for BUY YES (mean reversion from low price)
    if yes_ask < LOW_PRICE_THRESHOLD_CENTS:
        if distance_to_strike >= MAX_ADVERSE_DISTANCE_TO_STRIKE_BUY_YES: # BTC not too far below strike
            trade_action = "BUY_YES"
            # Simple score: further below threshold = higher score (more "undervalued")
            model_score = (LOW_PRICE_THRESHOLD_CENTS - yes_ask) / LOW_PRICE_THRESHOLD_CENTS 
            logger.debug(f"MeanReversion Signal: BUY_YES. YesAsk={yes_ask}c, DistToStrike={distance_to_strike:.2f}, Vol={btc_volatility:.2f}, Score={model_score:.2f}")
        else:
            logger.debug(f"MeanReversion BUY_YES condition: YesAsk ({yes_ask}c) low, but DistToStrike ({distance_to_strike:.2f}) too adverse.")

    # Check for BUY NO (mean reversion from high price for YES contract)
    # This happens if a BUY_YES signal was not already found
    if trade_action is None and yes_bid > HIGH_PRICE_THRESHOLD_CENTS:
        if distance_to_strike <= MAX_ADVERSE_DISTANCE_TO_STRIKE_BUY_NO: # BTC not too far above strike
            trade_action = "BUY_NO"
            # Simple score: further above threshold = higher score (more "overvalued" for YES)
            model_score = (yes_bid - HIGH_PRICE_THRESHOLD_CENTS) / (100 - HIGH_PRICE_THRESHOLD_CENTS)
            logger.debug(f"MeanReversion Signal: BUY_NO. YesBid={yes_bid}c, DistToStrike={distance_to_strike:.2f}, Vol={btc_volatility:.2f}, Score={model_score:.2f}")
        else:
            logger.debug(f"MeanReversion BUY_NO condition: YesBid ({yes_bid}c) high, but DistToStrike ({distance_to_strike:.2f}) too adverse.")

    if trade_action:
        return trade_action, model_score # Return the calculated score for sizing
    else:
        return None, None # No trade signal based on these rules