# sizing.py
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

# --- Sizing Configuration ---
# These can be overridden by backtest.py or other calling scripts
MAX_CAPITAL_ALLOCATION_PER_TRADE_USD = 20.0 # Max $ to allocate if max confidence (e.g., score of 5)
BASE_CAPITAL_ALLOCATION_PER_TRADE_USD = 5.0 # Min $ to allocate if trade signal met (e.g., score of 1)
MAX_CONTRACTS_PER_TRADE = 100 # Overall cap on number of contracts, regardless of capital

# Thresholds for the mapped model score (e.g., 0-5 scale)
# These are set by backtest_v2.py to align with map_prediction_to_sizing_score output
MODEL_SCORE_THRESHOLD_BUY_YES = 1.0 # Minimum score (after mapping) to consider a trade
MODEL_SCORE_THRESHOLD_BUY_NO = 1.0  # Minimum absolute score (after mapping)

# This defines the upper end of the score range for linear scaling of capital.
# If map_prediction_to_sizing_score outputs up to 5, this should be 5.
PRACTICAL_MAX_MODEL_SCORE_FOR_SCALING = 5.0


def calculate_position_size_capital_based(
    model_score: float, # This is now the mapped score (e.g., 0-5)
    contract_cost_cents: float,
    available_capital_usd: float
) -> int:
    """
    Calculates position size based on a mapped model score, contract cost, and available capital.
    The model_score is expected to be a confidence score (e.g., 0-5), where higher means more confident.
    """
    if pd.isna(contract_cost_cents) or contract_cost_cents <= 0 or contract_cost_cents >= 100:
        logger.debug(f"Sizing: Invalid contract cost ({contract_cost_cents}c). Size = 0.")
        return 0

    allocation_usd = 0
    abs_mapped_score = abs(model_score) # The input score is already mapped and its sign indicates direction

    # Check if the mapped score meets the minimum threshold for any trade
    # MODEL_SCORE_THRESHOLD_BUY_YES/NO now refer to the mapped score's minimum.
    # The direction (YES/NO) is already decided by the strategy before calling sizing.
    # Here, we just care if the confidence (abs_mapped_score) is high enough.
    
    # If mapped score is 0 (meaning original prediction was below strategy.PRED_THRESHOLD_BUY_YES/NO),
    # or if it's below the sizing module's own minimum threshold.
    # Note: backtest_v2 calls map_prediction_to_sizing_score which returns 0 if original pred < strategy threshold.
    # So, if abs_mapped_score < MODEL_SCORE_THRESHOLD_BUY_YES (e.g. < 1.0), it means no scaled allocation.
    if abs_mapped_score < MODEL_SCORE_THRESHOLD_BUY_YES: # Using one threshold for min confidence for sizing
        logger.debug(f"Sizing: Mapped model score {abs_mapped_score:.2f} is below sizing threshold {MODEL_SCORE_THRESHOLD_BUY_YES:.2f}. Size = 0.")
        return 0

    # Scale allocation based on the mapped score:
    # If score is at MODEL_SCORE_THRESHOLD_BUY_YES, use BASE_CAPITAL_ALLOCATION_PER_TRADE_USD.
    # If score is at PRACTICAL_MAX_MODEL_SCORE_FOR_SCALING, use MAX_CAPITAL_ALLOCATION_PER_TRADE_USD.
    # Linearly interpolate.
    
    min_score_for_scaling = MODEL_SCORE_THRESHOLD_BUY_YES # e.g., 1.0
    max_score_for_scaling = PRACTICAL_MAX_MODEL_SCORE_FOR_SCALING # e.g., 5.0

    if abs_mapped_score >= max_score_for_scaling:
        allocation_usd = MAX_CAPITAL_ALLOCATION_PER_TRADE_USD
    elif abs_mapped_score <= min_score_for_scaling: # Should be caught by the check above, but good for clarity
        allocation_usd = BASE_CAPITAL_ALLOCATION_PER_TRADE_USD
    else: # Score is between min_score_for_scaling and max_score_for_scaling
        scale_range = max_score_for_scaling - min_score_for_scaling
        if scale_range > 0:
            proportion_of_scaling = (abs_mapped_score - min_score_for_scaling) / scale_range
            allocation_usd = BASE_CAPITAL_ALLOCATION_PER_TRADE_USD + \
                             proportion_of_scaling * (MAX_CAPITAL_ALLOCATION_PER_TRADE_USD - BASE_CAPITAL_ALLOCATION_PER_TRADE_USD)
        else: # min and max scores for scaling are the same
            allocation_usd = BASE_CAPITAL_ALLOCATION_PER_TRADE_USD
    
    # Ensure allocation does not exceed available capital
    allocation_usd = min(allocation_usd, available_capital_usd)
    
    num_contracts_float = (allocation_usd * 100) / contract_cost_cents
    num_contracts = int(np.floor(num_contracts_float))

    num_contracts = min(num_contracts, MAX_CONTRACTS_PER_TRADE)

    if allocation_usd > 0 and num_contracts == 0:
        logger.debug(f"Sizing: Allocation ${allocation_usd:.2f} with cost {contract_cost_cents:.0f}c results in <1 contract. Size = 0.")
    
    logger.debug(f"Sizing (Capital): Mapped Score={model_score:.2f}, AllocUSD=${allocation_usd:.2f}, Cost/Cont={contract_cost_cents:.0f}c, NumContFloat={num_contracts_float:.2f}, FinalNumCont={num_contracts}")
    return num_contracts