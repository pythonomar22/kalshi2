# model.py
import logging
import math
from datetime import datetime, timezone

logger = logging.getLogger(__name__) # Will be configured by the main script

# --- Constants for a placeholder model ---
VOLATILITY_PLACEHOLDER_ANNUALIZED = 1.0 # Example: 100% annualized volatility
RISK_FREE_RATE_PLACEHOLDER = 0.05 # Example: 5% risk-free rate

def get_time_to_expiry_years(kalshi_contract_expiry_dt_utc: datetime) -> float:
    """Calculates time to expiry in fractional years."""
    now_utc = datetime.now(timezone.utc)
    if kalshi_contract_expiry_dt_utc <= now_utc:
        return 0.0
    time_delta = kalshi_contract_expiry_dt_utc - now_utc
    return time_delta.total_seconds() / (365.25 * 24 * 60 * 60)

def black_scholes_prob_above_strike(
    current_price: float,
    strike_price: float,
    time_to_expiry_yrs: float,
    volatility_annualized: float,
    risk_free_rate: float = RISK_FREE_RATE_PLACEHOLDER
) -> float | None:
    """
    Calculates the Black-Scholes probability of the asset price being above the strike at expiry.
    This is N(d2) in Black-Scholes, which is the risk-neutral probability of S_T > K.
    Note: Kalshi settles on an average, this model is for spot at expiry. It's a simplification.
    """
    if time_to_expiry_yrs <= 0 or volatility_annualized <= 0 or current_price <=0 or strike_price <= 0:
        # If already past expiry or invalid inputs, make a simple determination
        if current_price > strike_price: return 0.99 # Highly likely if past expiry and above
        elif current_price < strike_price: return 0.01 # Highly unlikely if past expiry and below
        else: return 0.5 # Indeterminate for edge cases or bad inputs before expiry
    
    d1 = (math.log(current_price / strike_price) + \
         (risk_free_rate + 0.5 * volatility_annualized**2) * time_to_expiry_yrs) / \
         (volatility_annualized * math.sqrt(time_to_expiry_yrs))
    
    d2 = d1 - volatility_annualized * math.sqrt(time_to_expiry_yrs)
    
    # N(d2) - Cumulative standard normal distribution
    prob_above_strike = 0.5 * (1 + math.erf(d2 / math.sqrt(2)))
    return prob_above_strike

def calculate_p_model(
    underlying_asset_symbol: str, # e.g., "ETHUSDT"
    current_asset_price: float | None,
    kalshi_contract_strike_price: float, # The actual numerical strike, e.g., 2629.99
    kalshi_contract_expiry_dt_utc: datetime,
    # Placeholder for more advanced model inputs
    recent_volatility_data=None, # Could be ATR, stddev from klines
    recent_momentum_data=None   # Could be MACD, RSI from klines
) -> float | None:
    """
    Calculates the bot's estimated probability (P_model) for the Kalshi contract.
    """
    logger.debug(
        f"P_Model Input: Asset={underlying_asset_symbol}, CurrentPrice={current_asset_price}, "
        f"KalshiStrike={kalshi_contract_strike_price}, ExpiryUTC={kalshi_contract_expiry_dt_utc}"
    )

    if current_asset_price is None:
        logger.warning(f"P_Model: Current asset price for {underlying_asset_symbol} is None. Cannot calculate.")
        return None

    # --- Placeholder Model: Uses simplified Black-Scholes for N(d2) ---
    # This is a very rough approximation because:
    # 1. Kalshi settles on an *average* of 60s of CF Benchmarks RTI, not spot price at expiry.
    # 2. We are using Binance spot as proxy for CF Benchmarks RTI.
    # 3. Volatility is a placeholder. Real volatility should be derived from market data.

    time_to_expiry_yrs = get_time_to_expiry_years(kalshi_contract_expiry_dt_utc)
    
    # Estimate short-term volatility - very crudely for now
    # A better approach would be to calculate historical or implied volatility from Binance klines
    # For example, using ATR or std dev of log returns from recent 1-minute klines.
    # If 'recent_volatility_data' (e.g. from Binance klines) is provided, use it.
    # For this placeholder, we use a constant.
    annualized_volatility = VOLATILITY_PLACEHOLDER_ANNUALIZED 
    if recent_volatility_data and isinstance(recent_volatility_data, dict): # e.g. from binance kline H/L
        # Example crude vol from 1-min kline: (high-low)/close annualized
        # This is NOT a good volatility measure, just for illustration
        try:
            k_h = float(recent_volatility_data.get('h', current_asset_price))
            k_l = float(recent_volatility_data.get('l', current_asset_price))
            k_c = float(recent_volatility_data.get('c', current_asset_price))
            if k_c > 0:
                one_min_range = (k_h - k_l) / k_c
                # Annualize: multiply by sqrt(Num_periods_in_year)
                # Roughly sqrt(365*24*60) for 1-minute periods
                # This is a very aggressive annualization for such a short period.
                # A more common way is std dev of log returns.
                # For simplicity, let's just use a slightly adjusted placeholder if data is extreme
                # annualized_volatility = max(0.2, min(2.0, one_min_range * math.sqrt(525600) ))
                pass # Sticking to placeholder for now unless a better short-term vol method is added
        except: # Broad except for parsing float errors
            pass


    p_model = black_scholes_prob_above_strike(
        current_price=current_asset_price,
        strike_price=kalshi_contract_strike_price,
        time_to_expiry_yrs=time_to_expiry_yrs,
        volatility_annualized=annualized_volatility
    )

    if p_model is not None:
        logger.info(
            f"P_Model Output for {underlying_asset_symbol} vs Kalshi Strike {kalshi_contract_strike_price:.2f}: "
            f"P_model={p_model:.4f} (TTE: {time_to_expiry_yrs*365.25:.2f} days, Vol: {annualized_volatility:.2f})"
        )
    else:
        logger.warning(f"P_Model for {underlying_asset_symbol} returned None.")
        
    return p_model