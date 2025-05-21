# sizing.py
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

# --- Sizing Configuration ---
# These can be overridden by the calling backtest script if needed.

# For Kelly Criterion Sizing
KELLY_FRACTION = 0.1  # Use a fraction of the calculated Kelly stake (e.g., 0.1 = 10% Kelly, 0.5 = Half Kelly)
                    # Full Kelly can be too aggressive.
MIN_CONTRACT_PRICE_CENTS_FOR_KELLY = 1 # Avoid division by zero or extreme odds for prices like 0 or 100
MAX_CONTRACT_PRICE_CENTS_FOR_KELLY = 99

# Fallback / General Caps (still useful with Kelly)
MAX_CAPITAL_ALLOCATION_PERCENTAGE_OF_TOTAL = 0.10 # Max 10% of total capital per trade, even if Kelly says more
MAX_CONTRACTS_PER_TRADE = 200 # Overall hard cap on number of contracts

# These are no longer directly used if Kelly determines allocation,
# but kept for reference or if you add fallback mechanisms.
# MAX_CAPITAL_ALLOCATION_PER_TRADE_USD = 100.0
# BASE_CAPITAL_ALLOCATION_PER_TRADE_USD = 10.0
# MODEL_SCORE_THRESHOLD_BUY_YES = 1.0
# MODEL_SCORE_THRESHOLD_BUY_NO = 1.0
# PRACTICAL_MAX_MODEL_SCORE_FOR_SCALING = 5.0


def calculate_kelly_position_size(
    model_prob_win: float,        # Model's probability of the chosen side winning (0 to 1)
    entry_price_cents: float,     # Cost of one contract for the chosen side in cents
    available_capital_usd: float,
    max_allocation_pct_override: float = None, # Optional override for MAX_CAPITAL_ALLOCATION_PERCENTAGE_OF_TOTAL
    kelly_fraction_override: float = None      # Optional override for KELLY_FRACTION
) -> int:
    """
    Calculates position size based on Kelly Criterion.

    Args:
        model_prob_win: Probability of the chosen bet winning (e.g., P(YES) if buying YES).
        entry_price_cents: The price in cents to enter the contract for the chosen side.
                           (e.g., yes_ask if buying YES; 100-yes_bid if buying NO).
        available_capital_usd: Current total available capital.
        max_allocation_pct_override: Override for the maximum percentage of capital to allocate.
        kelly_fraction_override: Override for the fraction of Kelly stake to use.

    Returns:
        Number of contracts to trade.
    """
    if pd.isna(model_prob_win) or not (0 <= model_prob_win <= 1):
        logger.debug(f"Sizing (Kelly): Invalid model_prob_win ({model_prob_win}). Size = 0.")
        return 0
    
    if pd.isna(entry_price_cents) or \
       not (MIN_CONTRACT_PRICE_CENTS_FOR_KELLY <= entry_price_cents <= MAX_CONTRACT_PRICE_CENTS_FOR_KELLY):
        logger.debug(f"Sizing (Kelly): Invalid entry_price_cents ({entry_price_cents}). Must be between {MIN_CONTRACT_PRICE_CENTS_FOR_KELLY} and {MAX_CONTRACT_PRICE_CENTS_FOR_KELLY}. Size = 0.")
        return 0

    if available_capital_usd <= 0:
        logger.debug("Sizing (Kelly): No available capital. Size = 0.")
        return 0

    # Calculate net odds (b)
    # If you win, you get 100 cents. You paid 'entry_price_cents'.
    # Profit if win = 100 - entry_price_cents
    # Amount risked = entry_price_cents
    # b = (Profit if win) / (Amount risked)
    profit_if_win_cents = 100.0 - entry_price_cents
    amount_risked_cents = entry_price_cents
    
    if amount_risked_cents <= 0: # Should be caught by price check, but defensive
        logger.debug("Sizing (Kelly): Amount risked is zero or negative. Cannot calculate odds. Size = 0.")
        return 0
    
    b_odds = profit_if_win_cents / amount_risked_cents

    if b_odds <= 0: # This means profit_if_win is not positive (e.g. entry_price is 100)
        logger.debug(f"Sizing (Kelly): Net odds 'b' ({b_odds:.2f}) are not positive. No bet. (Price: {entry_price_cents}c)")
        return 0

    # Kelly Criterion formula: f* = p - q/b  (where p=prob_win, q=prob_loss=1-p)
    prob_loss = 1.0 - model_prob_win
    kelly_f_star = model_prob_win - (prob_loss / b_odds)

    current_kelly_fraction = kelly_fraction_override if kelly_fraction_override is not None else KELLY_FRACTION

    if kelly_f_star <= 0:
        logger.debug(f"Sizing (Kelly): f* ({kelly_f_star:.4f}) is not positive. No bet. (p={model_prob_win:.2f}, b={b_odds:.2f})")
        return 0
        
    # Apply the chosen fraction of Kelly
    effective_stake_fraction = kelly_f_star * current_kelly_fraction
    
    # Capital to allocate based on Kelly
    capital_to_allocate_usd = effective_stake_fraction * available_capital_usd
    
    # Apply a cap based on a percentage of total capital if set
    current_max_alloc_pct = max_allocation_pct_override if max_allocation_pct_override is not None else MAX_CAPITAL_ALLOCATION_PERCENTAGE_OF_TOTAL
    max_capital_by_pct_limit = available_capital_usd * current_max_alloc_pct
    capital_to_allocate_usd = min(capital_to_allocate_usd, max_capital_by_pct_limit)

    if capital_to_allocate_usd <= 0:
        return 0 # No capital to allocate after limits

    # Convert allocated capital to number of contracts
    num_contracts_float = (capital_to_allocate_usd * 100.0) / entry_price_cents # entry_price_cents is cost per contract
    num_contracts = int(np.floor(num_contracts_float))

    # Apply hard cap on number of contracts
    num_contracts = min(num_contracts, MAX_CONTRACTS_PER_TRADE)
    
    # Ensure we don't try to bet more than available capital, even after fractional Kelly
    # This check should ideally be redundant if MAX_CAPITAL_ALLOCATION_PERCENTAGE_OF_TOTAL <= 1.0
    # and effective_stake_fraction <=1.0, but good for safety.
    total_cost_of_trade_usd = (num_contracts * entry_price_cents) / 100.0
    if total_cost_of_trade_usd > available_capital_usd:
        num_contracts = int(np.floor((available_capital_usd * 100.0) / entry_price_cents))
        # Re-apply max contracts cap if sizing down due to capital limit
        num_contracts = min(num_contracts, MAX_CONTRACTS_PER_TRADE)


    if capital_to_allocate_usd > 0 and num_contracts == 0:
        logger.debug(f"Sizing (Kelly): Allocation ${capital_to_allocate_usd:.2f} with cost {entry_price_cents:.0f}c results in <1 contract. Size = 0.")
    
    logger.debug(f"Sizing (Kelly): p_win={model_prob_win:.3f}, entry_price={entry_price_cents:.0f}c, b_odds={b_odds:.3f} -> "
                 f"f*={kelly_f_star:.3f}, eff_stake_frac={effective_stake_fraction:.3f}, "
                 f"AllocUSD=${capital_to_allocate_usd:.2f}, NumContFloat={num_contracts_float:.2f}, FinalNumCont={num_contracts}")
    
    return max(0, num_contracts) # Ensure non-negative