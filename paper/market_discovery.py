# /paper/market_discovery.py

import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import List, Optional, Dict, Any
import re
from collections import defaultdict

import httpx 

import paper.config as cfg
from paper.utils import load_private_key, get_kalshi_auth_headers
from paper.data_models import KalshiMarketInfo

logger = logging.getLogger("paper_market_discovery")

KALSHI_PRIVATE_KEY = load_private_key(cfg.KALSHI_PRIVATE_KEY_PATH)

def parse_strike_from_ticker(ticker: str) -> Optional[float]:
    match = re.search(r"-T([0-9]+\.?[0-9]*)$", ticker)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            logger.warning(f"Could not parse strike from ticker segment '{match.group(1)}' in ticker '{ticker}'")
            return None
    logger.debug(f"Could not find strike pattern in ticker '{ticker}'")
    return None

def round_to_kalshi_strike_step(price: float, step: int = cfg.KALSHI_STRIKE_INCREMENT) -> float:
    """Rounds a price to the nearest Kalshi strike step (e.g., $250)."""
    return round(price / step) * step

async def fetch_active_markets(
    client: httpx.AsyncClient, 
    target_series_ticker: str,
    current_btc_price_for_ntm: Optional[float] # Pass current BTC price for NTM selection
) -> List[KalshiMarketInfo]:
    if not KALSHI_PRIVATE_KEY:
        logger.error("Kalshi private key not loaded. Cannot fetch active markets.")
        return []

    endpoint = "/trade-api/v2/markets"
    params = {
        "status": "open", 
        "series_ticker": target_series_ticker, 
        "limit": 100 
    }

    auth_headers = get_kalshi_auth_headers("GET", endpoint, KALSHI_PRIVATE_KEY, cfg.KALSHI_API_KEY_ID)
    if not auth_headers:
        logger.error("Failed to generate auth headers for market discovery.")
        return []

    all_raw_markets_data: List[Dict[str, Any]] = [] # Renamed for clarity
    cursor = None
    page_count = 0
    max_pages = 20 

    logger.debug(f"Attempting to fetch markets from endpoint: {cfg.KALSHI_API_BASE_URL}{endpoint} with params: {params}")

    while page_count < max_pages : 
        current_params = params.copy()
        if cursor:
            current_params["cursor"] = cursor
        
        try:
            response = await client.get(f"{cfg.KALSHI_API_BASE_URL}{endpoint}", headers=auth_headers, params=current_params, timeout=30.0)
            response.raise_for_status()
            data = response.json()
            
            markets_on_page = data.get("markets", [])
            if markets_on_page:
                all_raw_markets_data.extend(markets_on_page)
            
            cursor = data.get("cursor")
            page_count += 1
            if not cursor or not markets_on_page: 
                break
            await asyncio.sleep(0.3) 

        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error fetching markets for series {target_series_ticker}: {e.response.status_code} - {e.response.text}")
            return [] 
        except httpx.RequestError as e:
            logger.error(f"Request error fetching markets for series {target_series_ticker}: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error fetching markets for series {target_series_ticker}: {e}", exc_info=True)
            return []
            
    logger.info(f"Fetched {len(all_raw_markets_data)} raw market entries for series '{target_series_ticker}' with status 'open' after {page_count} pages.")
    
    # --- Preliminary Parsing and Grouping by Event ---
    candidate_markets_by_event: Dict[str, List[KalshiMarketInfo]] = defaultdict(list)
    now_utc = datetime.now(timezone.utc)

    for m_data in all_raw_markets_data:
        try:
            market_ticker = m_data.get('ticker')
            event_ticker = m_data.get('event_ticker')
            if not market_ticker or not event_ticker:
                continue

            # Initial filter for series (API should handle this, but good to double-check)
            api_series_ticker = m_data.get('series_ticker')
            if api_series_ticker is not None and api_series_ticker != target_series_ticker:
                continue

            open_time = datetime.fromisoformat(m_data['open_time'].replace('Z', '+00:00'))
            close_time = datetime.fromisoformat(m_data['close_time'].replace('Z', '+00:00'))
            market_status_from_api = m_data.get('status', '').lower()

            time_until_close = close_time - now_utc
            min_time_to_trade = timedelta(minutes=cfg.MIN_MINUTES_BEFORE_RESOLUTION_FOR_DECISION + 1)
            max_time_to_trade = timedelta(hours=2, minutes=30) 

            if not (market_status_from_api in ['open', 'active'] and \
                    open_time <= now_utc < close_time and \
                    min_time_to_trade < time_until_close < max_time_to_trade):
                continue
            
            strike_val = parse_strike_from_ticker(market_ticker)
            if strike_val is None: # If ticker doesn't encode strike, try API field
                strike_str_from_api = m_data.get('strike')
                if strike_str_from_api is not None:
                    try: strike_val = float(strike_str_from_api)
                    except ValueError: pass
            
            if strike_val is None:
                logger.debug(f"Skipping {market_ticker}: Could not determine strike price.")
                continue
            
            series_ticker_for_model = api_series_ticker if api_series_ticker is not None else target_series_ticker
            
            market_info = KalshiMarketInfo(
                ticker=market_ticker,
                series_ticker=series_ticker_for_model,
                event_ticker=event_ticker,
                strike_price=strike_val, 
                open_time_utc=open_time,
                close_time_utc=close_time,
                status=market_status_from_api,
                last_price_yes_cents=m_data.get('last_price')
            )
            candidate_markets_by_event[event_ticker].append(market_info)

        except Exception as e:
            logger.warning(f"Error pre-parsing market data for ticker {m_data.get('ticker', 'N/A')}: {e}. Data: {m_data}")
            continue
    
    # --- NTM Selection per Event ---
    selected_ntm_markets: List[KalshiMarketInfo] = []
    if not current_btc_price_for_ntm:
        logger.warning("No current BTC price provided for NTM selection. Cannot select NTM markets effectively.")
        # Fallback: maybe just take the first few markets per event up to MAX_MARKETS_TO_MONITOR
        # For now, we'll just return empty if no BTC price
        return []

    for event_ticker, markets_in_event in candidate_markets_by_event.items():
        if not markets_in_event:
            continue

        # Sort markets by strike price
        markets_in_event.sort(key=lambda m: m.strike_price)
        
        # Find markets closest to current_btc_price_for_ntm
        markets_with_diff = [
            (m, abs(m.strike_price - current_btc_price_for_ntm)) for m in markets_in_event
        ]
        markets_with_diff.sort(key=lambda x: x[1]) # Sort by difference to current price

        # Select NTM_MARKETS_PER_EVENT closest markets
        count_to_select = cfg.NUM_NTM_MARKETS_PER_EVENT
        
        # Ensure we have an odd number for a central "at-the-money" feel if possible
        if count_to_select % 2 == 0 and count_to_select > 0:
            count_to_select = max(1, count_to_select -1) # Make it odd, e.g. 4->3, 6->5. If 0, stays 0.
        
        for market_obj, diff in markets_with_diff[:count_to_select]:
            if len(selected_ntm_markets) < cfg.MAX_MARKETS_TO_MONITOR:
                selected_ntm_markets.append(market_obj)
                logger.debug(f"NTM Selected for {event_ticker}: {market_obj.ticker} (Strike: {market_obj.strike_price}, Diff: {diff:.2f})")
            else:
                logger.warning(f"Hit MAX_MARKETS_TO_MONITOR ({cfg.MAX_MARKETS_TO_MONITOR}) during NTM selection. Halting selection.")
                break
        if len(selected_ntm_markets) >= cfg.MAX_MARKETS_TO_MONITOR:
            break 
            
    if not selected_ntm_markets and len(all_raw_markets_data) > 0:
        logger.info(f"No NTM markets selected for series '{target_series_ticker}' despite fetching {len(all_raw_markets_data)} candidates. BTC Price: {current_btc_price_for_ntm}")
    elif selected_ntm_markets:
        logger.info(f"Selected {len(selected_ntm_markets)} NTM markets across all events for series '{target_series_ticker}'.")
        
    return selected_ntm_markets


async def get_market_details_rest(client: httpx.AsyncClient, market_ticker: str) -> Optional[KalshiMarketInfo]:
    if not KALSHI_PRIVATE_KEY:
        logger.error("Kalshi private key not loaded. Cannot fetch market details.")
        return None

    endpoint = f"/trade-api/v2/markets/{market_ticker}"
    auth_headers = get_kalshi_auth_headers("GET", endpoint, KALSHI_PRIVATE_KEY, cfg.KALSHI_API_KEY_ID)
    if not auth_headers: return None

    try:
        response = await client.get(f"{cfg.KALSHI_API_BASE_URL}{endpoint}", headers=auth_headers, timeout=15.0)
        response.raise_for_status()
        m_data = response.json().get("market")
        if m_data:
            open_time = datetime.fromisoformat(m_data['open_time'].replace('Z', '+00:00'))
            close_time = datetime.fromisoformat(m_data['close_time'].replace('Z', '+00:00'))
            
            strike_val: Optional[float] = None
            strike_str_from_api = m_data.get('strike')
            if strike_str_from_api is not None:
                try: strike_val = float(strike_str_from_api)
                except ValueError: logger.warning(f"Could not parse strike '{strike_str_from_api}' from API in get_market_details_rest for {market_ticker}.")
            
            if strike_val is None: strike_val = parse_strike_from_ticker(market_ticker)
            if strike_val is None: 
                logger.warning(f"CRITICAL: Strike still None after all attempts for {market_ticker} in get_market_details_rest.")
                return None

            return KalshiMarketInfo(
                ticker=market_ticker, 
                series_ticker=m_data.get('series_ticker', cfg.TARGET_EVENT_SERIES_TICKER),
                event_ticker=m_data['event_ticker'],
                strike_price=strike_val,
                open_time_utc=open_time,
                close_time_utc=close_time,
                status=m_data.get('status','').lower(),
                last_price_yes_cents=m_data.get('last_price')
            )
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error fetching details for {market_ticker}: {e.response.status_code} - {e.response.text}")
    except Exception as e:
        logger.error(f"Error fetching details for {market_ticker}: {e}", exc_info=True)
    return None

async def get_resolved_market_outcome(client: httpx.AsyncClient, market_ticker: str) -> Optional[str]:
    endpoint = f"/trade-api/v2/markets/{market_ticker}" 
    if not KALSHI_PRIVATE_KEY:
        logger.error(f"Kalshi private key not loaded. Cannot fetch outcome for {market_ticker}.")
        return None
        
    auth_headers = get_kalshi_auth_headers("GET", endpoint, KALSHI_PRIVATE_KEY, cfg.KALSHI_API_KEY_ID)
    if not auth_headers: return None
    try:
        response = await client.get(f"{cfg.KALSHI_API_BASE_URL}{endpoint}", headers=auth_headers, timeout=15.0)
        response.raise_for_status() 
        market_data = response.json().get("market")
        if market_data and market_data.get("status", "").lower() in ["settled", "finalized"]:
            result = market_data.get("result")
            if result and result.upper() in ["YES", "NO"]:
                return result.upper()
            else:
                logger.warning(f"Market {market_ticker} is {market_data.get('status')} but result field is missing/invalid: {result}")
        else:
            current_status = market_data.get('status', 'N/A') if market_data else 'N/A (no market_data)'
            logger.debug(f"Market {market_ticker} not yet settled/finalized or data missing. Status: {current_status}")
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            logger.warning(f"Market {market_ticker} not found when trying to fetch outcome (404).")
        else:
            logger.error(f"HTTP error fetching outcome for {market_ticker}: {e.response.status_code} - {e.response.text}")
    except Exception as e:
        logger.error(f"Unexpected error fetching outcome for {market_ticker}: {e}", exc_info=True)
    return None