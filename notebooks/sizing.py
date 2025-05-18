# sizing.py
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

# --- Sizing Configuration ---
# These can be overridden by backtest.py
MAX_CAPITAL_ALLOCATION_PER_TRADE_USD = 20.0 # Max $ to allocate if max confidence
BASE_CAPITAL_ALLOCATION_PER_TRADE_USD = 5.0 # Min $ to allocate if trade signal met
MAX_CONTRACTS_PER_TRADE = 100 # Overall cap on number of contracts, regardless of capital

# Model thresholds (set by backtest.py) - needed for determining min/max score for scaling
MODEL_SCORE_THRESHOLD_BUY_YES = 0.5 
MODEL_SCORE_THRESHOLD_BUY_NO = -0.5 
# Define a practical max score for scaling (e.g., beyond this score, confidence is considered maxed out for sizing)
# This prevents extreme scores from leading to disproportionately large (though still capped) allocations.
# This would typically be observed from your model's score distribution during training/validation.
PRACTICAL_MAX_MODEL_SCORE_FOR_SCALING = 5.0 


def calculate_position_size_capital_based(
    model_score: float, 
    contract_cost_cents: float, # Cost of one contract in cents (e.g., yes_ask or (100-yes_bid))
    available_capital_usd: float # Total trading capital
) -> int:
    """
    Calculates position size based on model score, contract cost, and available capital.
    This version tries to allocate a portion of capital based on confidence.
    """
    if pd.isna(contract_cost_cents) or contract_cost_cents <= 0 or contract_cost_cents >= 100:
        logger.debug(f"Invalid contract cost ({contract_cost_cents}c). Cannot size position.")
        return 0

    # 1. Determine the dollar amount to allocate based on confidence
    #    This logic maps model_score to a capital allocation.
    
    allocation_usd = 0
    abs_score = abs(model_score)
    
    # Check if score meets minimum threshold for any trade
    trade_signal_exists = (model_score > MODEL_SCORE_THRESHOLD_BUY_YES) or \
                          (model_score < MODEL_SCORE_THRESHOLD_BUY_NO)

    if not trade_signal_exists:
        logger.debug(f"Model score {model_score:.2f} below trade threshold. Size = 0.")
        return 0

    # Scale allocation:
    # If score is just at threshold, use BASE_CAPITAL_ALLOCATION_PER_TRADE_USD
    # If score reaches PRACTICAL_MAX_MODEL_SCORE_FOR_SCALING, use MAX_CAPITAL_ALLOCATION_PER_TRADE_USD
    # Linearly interpolate between these points.
    
    min_abs_threshold = min(abs(MODEL_SCORE_THRESHOLD_BUY_YES), abs(MODEL_SCORE_THRESHOLD_BUY_NO))
    if min_abs_threshold == 0 and PRACTICAL_MAX_MODEL_SCORE_FOR_SCALING == 0: # Avoid division by zero if thresholds are 0
        scale_range = 1.0
    else:
        scale_range = PRACTICAL_MAX_MODEL_SCORE_FOR_SCALING - min_abs_threshold

    if abs_score <= min_abs_threshold: # Should have been caught by trade_signal_exists
        allocation_usd = BASE_CAPITAL_ALLOCATION_PER_TRADE_USD 
    elif abs_score >= PRACTICAL_MAX_MODEL_SCORE_FOR_SCALING:
        allocation_usd = MAX_CAPITAL_ALLOCATION_PER_TRADE_USD
    else: # Score is between min_abs_threshold and PRACTICAL_MAX_MODEL_SCORE_FOR_SCALING
        if scale_range > 0:
            proportion_of_scaling = (abs_score - min_abs_threshold) / scale_range
            allocation_usd = BASE_CAPITAL_ALLOCATION_PER_TRADE_USD + \
                             proportion_of_scaling * (MAX_CAPITAL_ALLOCATION_PER_TRADE_USD - BASE_CAPITAL_ALLOCATION_PER_TRADE_USD)
        else: # min_abs_threshold and PRACTICAL_MAX_MODEL_SCORE are the same
            allocation_usd = BASE_CAPITAL_ALLOCATION_PER_TRADE_USD


    # Ensure allocation does not exceed available capital
    allocation_usd = min(allocation_usd, available_capital_usd)
    
    # 2. Convert allocated dollar amount to number of contracts
    if contract_cost_cents <= 0: # Should be caught earlier
        logger.warning("Contract cost is zero or negative, cannot calculate contracts.")
        return 0
        
    num_contracts_float = (allocation_usd * 100) / contract_cost_cents
    num_contracts = int(np.floor(num_contracts_float)) # Always round down to not exceed allocation

    # 3. Apply overall max contracts cap
    num_contracts = min(num_contracts, MAX_CONTRACTS_PER_TRADE)

    # 4. Ensure at least 1 contract if any allocation was made (unless rounded down to 0)
    if allocation_usd > 0 and num_contracts == 0:
        # This can happen if contract_cost_cents is very high relative to allocation
        # e.g., allocate $5 (500c), contract cost 99c -> 500/99 = 5.05 -> 5 contracts
        # e.g., allocate $0.5 (50c), contract cost 70c -> 50/70 = 0.71 -> 0 contracts
        logger.debug(f"Allocation {allocation_usd:.2f} with cost {contract_cost_cents:.0f}c results in <1 contract. Setting to 0.")
        # Or you could force it to 1 if any allocation > min_contract_cost, but floor is safer.
        # num_contracts = 1 # if you want to always trade at least 1 if any capital is allocated and affordable
    
    logger.debug(f"Sizing (Capital): Score={model_score:.2f}, AllocUSD={allocation_usd:.2f}, Cost/Cont={contract_cost_cents:.0f}c, NumContFloat={num_contracts_float:.2f}, FinalNumCont={num_contracts}")
    return num_contracts