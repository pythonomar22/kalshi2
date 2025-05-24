# /paper/trade_logger.py

import csv
from pathlib import Path
from datetime import datetime
import logging

import paper.config as cfg
from paper.data_models import PaperTrade # Import your Pydantic model

logger = logging.getLogger("paper_trade_logger")

# CSV Header, align with PaperTrade model and any extra info
CSV_FIELDNAMES = [
    "trade_id", "market_ticker", "decision_timestamp_utc", "action",
    "predicted_prob_yes", "entry_price_cents", "contracts_traded", "trade_value_cents",
    "resolution_time_utc", "actual_outcome", "pnl_cents", "status",
    "kelly_f_star", "capital_at_trade_cents"
]

def get_trade_log_filepath(decision_time_utc: datetime) -> Path:
    """Determines the filepath for the hourly trade log."""
    date_str = decision_time_utc.strftime("%Y-%m-%d")
    hour_str = decision_time_utc.strftime("%H")
    
    date_specific_log_dir = cfg.LOG_DIR / date_str
    date_specific_log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file_path = date_specific_log_dir / f"paper_trades_{hour_str}.csv"
    return log_file_path

def log_paper_trade(trade: PaperTrade):
    """Logs a PaperTrade object to the appropriate CSV file."""
    log_file_path = get_trade_log_filepath(trade.decision_timestamp_utc)
    file_exists = log_file_path.exists()

    try:
        with open(log_file_path, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=CSV_FIELDNAMES, extrasaction='ignore')
            if not file_exists or log_file_path.stat().st_size == 0:
                writer.writeheader()
            
            # Convert datetime objects to ISO strings for CSV
            trade_dict = trade.model_dump(mode='json') # Pydantic's way to get dict, handles Optional
            
            # Ensure datetime objects are ISO formatted if not handled by model_dump
            for key in ['decision_timestamp_utc', 'resolution_time_utc']:
                if isinstance(trade_dict.get(key), datetime):
                     trade_dict[key] = trade_dict[key].isoformat()
                elif trade_dict.get(key) is None and key in CSV_FIELDNAMES: # Handle Optional fields that are None
                    trade_dict[key] = ''


            writer.writerow(trade_dict)
        logger.info(f"Logged paper trade {trade.trade_id} for {trade.market_ticker} to {log_file_path.name}")
    except IOError as e:
        logger.error(f"IOError writing trade {trade.trade_id} to CSV {log_file_path}: {e}")
    except Exception as e:
        logger.error(f"Unexpected error writing trade {trade.trade_id} to CSV: {e}", exc_info=True)

def update_trade_in_log(updated_trade: PaperTrade):
    """
    Updates an existing trade in the log file.
    This is more complex as it requires reading, finding, and rewriting.
    For simplicity in paper trading, you might log closures as new entries
    or handle P&L updates in memory and a separate summary.
    A true update would involve:
    1. Reading all trades from the relevant hourly CSV.
    2. Finding the trade by trade_id.
    3. Updating its fields (status, pnl_cents, actual_outcome).
    4. Rewriting the entire CSV file.
    This can be slow. A simpler approach for now:
    - Log initial trade placement.
    - When a trade closes, log a "closure event" or update a portfolio summary.
    - The current log_paper_trade is for new trades.
    We can enhance this later if precise in-place CSV updates are critical.
    For now, we'll focus on logging the placement and then managing P&L in PortfolioManager.
    """
    logger.info(f"Trade update requested for {updated_trade.trade_id}. "
                f"Consider logging closure events separately or managing P&L via PortfolioManager for now.")
    # Placeholder: Actual update logic would be implemented here if needed.
    # For now, we can log an updated full row if necessary, potentially leading to duplicates if not careful.
    # Or, a separate log file for "trade_updates" or "closures".
    # Let's just re-log it for simplicity for now, understanding it might create a new line.
    # A better way would be to have a portfolio manager update its internal state and persist that.
    log_paper_trade(updated_trade) # This will append a new line with updated info.