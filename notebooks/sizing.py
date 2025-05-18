# sizing.py
import numpy as np
import logging

logger = logging.getLogger(__name__)

# --- Sizing Configuration ---
BASE_POSITION_SIZE = 1  # Number of contracts for a baseline signal
MAX_POSITION_SIZE = 5   # Maximum number of contracts to trade
CONFIDENCE_SCALING_FACTOR = 0.5 # How much the model score influences size beyond base
                               # e.g. score of 2.0 * 0.5 = 1.0 extra contracts
MIN_MODEL_SCORE_FOR_SCALING = 1.0 # Minimum absolute score to start scaling size beyond base

def calculate_position_size(model_score: float) -> int:
    """
    Calculates position size based on model score.
    
    Args:
        model_score (float): The raw score from the prediction model.
                             Positive for YES, negative for NO.
    
    Returns:
        int: The number of contracts to trade.
    """
    absolute_score = abs(model_score)
    
    if absolute_score < min(abs(MODEL_SCORE_THRESHOLD_BUY_YES), abs(MODEL_SCORE_THRESHOLD_BUY_NO)): # Using global model thresholds from backtest.py
        # This case should ideally be caught by the trade decision logic in backtest.py already
        logger.debug(f"Model score {model_score:.2f} is below minimum trade threshold. Defaulting to 0 size (should not happen if called after trade decision).")
        return 0

    # Start with base size
    calculated_size = BASE_POSITION_SIZE
    
    # Add scaled confidence
    # Only scale up if score significantly exceeds the basic threshold
    if absolute_score > MIN_MODEL_SCORE_FOR_SCALING:
        additional_size = (absolute_score - MIN_MODEL_SCORE_FOR_SCALING) * CONFIDENCE_SCALING_FACTOR
        calculated_size += additional_size
        
    # Cap at max position size and ensure it's an integer
    final_size = min(MAX_POSITION_SIZE, int(round(calculated_size)))
    
    # Ensure minimum size is at least BASE_POSITION_SIZE if a trade is triggered
    final_size = max(BASE_POSITION_SIZE, final_size) 
    
    logger.debug(f"Model score: {model_score:.2f}, Calculated raw size: {calculated_size:.2f}, Final rounded size: {final_size}")
    return final_size

# These need to be accessible by the sizing function if it's to be self-contained
# Or, pass them as arguments. For now, assume they are defined in the calling scope (backtest.py)
# This is a bit of a hack; ideally, sizing would be more decoupled or take thresholds.
# Let's make it so that backtest.py calls this *after* deciding to trade.
MODEL_SCORE_THRESHOLD_BUY_YES = 0.5 # Placeholder, will be overwritten by backtest.py's value
MODEL_SCORE_THRESHOLD_BUY_NO = -0.5 # Placeholder