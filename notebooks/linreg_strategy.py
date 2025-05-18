# linreg_strategy.py
import pandas as pd
import logging

logger = logging.getLogger(__name__)

# --- Simulated Linear Regression Model Parameters ---
# These would normally be loaded from a trained model file or a config
# For now, they are hardcoded here for this specific strategy module.
MODEL_PARAMS = {
    'intercept': 0.0,
    'coef_btc_price_change_1m': 0.05,
    'coef_btc_price_change_5m': 0.1,
    'coef_btc_price_change_15m': 0.02,
    'coef_btc_volatility_15m': -0.01,
    'coef_distance_to_strike': 0.0001 
}
# Thresholds for taking action based on model score
MODEL_SCORE_THRESHOLD_BUY_YES = 0.2 
MODEL_SCORE_THRESHOLD_BUY_NO = -0.2

FEATURE_KEYS = [ # Ensure this order matches how coefficients are used if doing dot product
    'btc_price_change_1m', 
    'btc_price_change_5m', 
    'btc_price_change_15m', 
    'btc_volatility_15m',
    'distance_to_strike' # This one is calculated separately
]


def calculate_model_score(btc_features: dict, kalshi_strike_price: float) -> float | None:
    """
    Calculates the prediction score based on BTC features and Kalshi strike.
    """
    if btc_features is None:
        logger.warning("BTC features are None, cannot calculate model score.")
        return None

    # Ensure all required features are present
    for key in FEATURE_KEYS:
        if key == 'distance_to_strike': continue # Calculated next
        if key not in btc_features or pd.isna(btc_features[key]):
            logger.warning(f"Missing or NaN feature '{key}' for model score calculation.")
            return None
    if pd.isna(btc_features.get('btc_price')): # btc_price needed for distance_to_strike
        logger.warning("Missing btc_price for distance_to_strike calculation.")
        return None

    distance_to_strike = btc_features['btc_price'] - kalshi_strike_price

    score = MODEL_PARAMS['intercept'] + \
            (MODEL_PARAMS['coef_btc_price_change_1m'] * btc_features['btc_price_change_1m']) + \
            (MODEL_PARAMS['coef_btc_price_change_5m'] * btc_features['btc_price_change_5m']) + \
            (MODEL_PARAMS['coef_btc_price_change_15m'] * btc_features['btc_price_change_15m']) + \
            (MODEL_PARAMS['coef_btc_volatility_15m'] * btc_features['btc_volatility_15m']) + \
            (MODEL_PARAMS['coef_distance_to_strike'] * distance_to_strike)
    
    feature_str = (f"BTC Last={btc_features['btc_price']:.2f}, Chg1m={btc_features['btc_price_change_1m']:.2f}, "
                   f"Chg5m={btc_features['btc_price_change_5m']:.2f}, Chg15m={btc_features['btc_price_change_15m']:.2f}, "
                   f"Vol15m={btc_features['btc_volatility_15m']:.2f}, DistToStrike={distance_to_strike:.2f}")
    logger.debug(f"Model Features: {feature_str}, Calculated Score={score:.4f}")
    return score


def get_trade_decision(model_prediction_score: float | None):
    """
    Determines trade action based on the model's prediction score.

    Args:
        model_prediction_score (float | None): The score from the model.

    Returns:
        tuple: (trade_action_str or None, model_prediction_score)
               trade_action_str can be "BUY_YES", "BUY_NO".
    """
    if model_prediction_score is None:
        return None, None

    trade_action = None
    if model_prediction_score > MODEL_SCORE_THRESHOLD_BUY_YES:
        trade_action = "BUY_YES"
    elif model_prediction_score < MODEL_SCORE_THRESHOLD_BUY_NO:
        trade_action = "BUY_NO"
    
    return trade_action, model_prediction_score