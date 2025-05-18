# sizing.py
import numpy as np
import logging

logger = logging.getLogger(__name__)

# --- Sizing Configuration ---
# These can be overridden by backtest.py by directly setting sizing.VARIABLE_NAME
BASE_POSITION_SIZE = 1
MAX_POSITION_SIZE = 5
CONFIDENCE_SCALING_FACTOR = 0.5
MIN_MODEL_SCORE_FOR_SCALING = 1.0 

# Model thresholds needed for one of the checks
MODEL_SCORE_THRESHOLD_BUY_YES = 0.5 # Will be updated by backtest.py
MODEL_SCORE_THRESHOLD_BUY_NO = -0.5 # Will be updated by backtest.py

def calculate_position_size(model_score: float) -> int:
    absolute_score = abs(model_score)
    
    # This check ensures we don't try to size a trade that shouldn't have been triggered.
    # It relies on MODEL_SCORE_THRESHOLD_BUY_YES and MODEL_SCORE_THRESHOLD_BUY_NO being set correctly
    # by the calling module (backtest.py) to match its decision logic.
    if (model_score > 0 and model_score < MODEL_SCORE_THRESHOLD_BUY_YES) or \
       (model_score < 0 and model_score > MODEL_SCORE_THRESHOLD_BUY_NO) or \
       (model_score == 0 and (MODEL_SCORE_THRESHOLD_BUY_YES > 0 or MODEL_SCORE_THRESHOLD_BUY_NO < 0)): # Handle score of 0
        logger.debug(f"Model score {model_score:.2f} is below effective trade threshold. Defaulting to 0 size.")
        return 0

    calculated_size = float(BASE_POSITION_SIZE) # Start as float for accumulation
    
    if absolute_score > MIN_MODEL_SCORE_FOR_SCALING:
        # Scale based on how much the score exceeds the MIN_MODEL_SCORE_FOR_SCALING, not the entry threshold
        excess_score_for_scaling = absolute_score - MIN_MODEL_SCORE_FOR_SCALING
        additional_size = excess_score_for_scaling * CONFIDENCE_SCALING_FACTOR
        calculated_size += additional_size
        
    final_size = min(MAX_POSITION_SIZE, int(round(calculated_size)))
    final_size = max(BASE_POSITION_SIZE if model_score !=0 else 0, final_size) # Ensure at least base size if any trade, or 0 if score was 0
    
    logger.debug(f"Sizing: Score={model_score:.2f}, RawCalcSize={calculated_size:.2f}, FinalIntSize={final_size}")
    return final_size