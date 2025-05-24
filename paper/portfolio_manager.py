# /paper/portfolio_manager.py

import logging
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional

import paper.config as cfg
from paper.data_models import PaperTrade
from paper.trade_logger import log_paper_trade, update_trade_in_log # For logging updates

logger = logging.getLogger("paper_portfolio_manager")

class PortfolioManager:
    def __init__(self, initial_capital_cents: int):
        self.current_capital_cents: int = initial_capital_cents
        self.initial_capital_cents: int = initial_capital_cents
        self.open_trades: Dict[str, PaperTrade] = {} # trade_id: PaperTrade object
        self.closed_trades: List[PaperTrade] = []
        self.total_pnl_cents: int = 0
        logger.info(f"PortfolioManager initialized with paper capital: {self.current_capital_cents / 100:.2f} USD")

    def get_current_capital_cents(self) -> int:
        return self.current_capital_cents

    def can_afford_trade(self, trade_value_cents: int) -> bool:
        return self.current_capital_cents >= trade_value_cents

    def open_paper_trade(
        self,
        market_ticker: str,
        decision_timestamp_utc: datetime,
        action: str, # "BUY_YES" or "BUY_NO"
        predicted_prob_yes: float,
        entry_price_cents: int,
        contracts_traded: int,
        resolution_time_utc: datetime,
        kelly_f_star: Optional[float] = None
    ) -> Optional[PaperTrade]:
        
        trade_value_cents = entry_price_cents * contracts_traded
        if not self.can_afford_trade(trade_value_cents) and cfg.USE_KELLY_CRITERION:
            logger.warning(f"Paper trade opening: Insufficient capital for {contracts_traded} contracts "
                           f"of {market_ticker} @ {entry_price_cents}c. Value: {trade_value_cents}c, "
                           f"Capital: {self.current_capital_cents}c")
            return None

        trade_id = str(uuid.uuid4())
        capital_before_trade = self.current_capital_cents

        trade = PaperTrade(
            trade_id=trade_id,
            market_ticker=market_ticker,
            decision_timestamp_utc=decision_timestamp_utc,
            action=action,
            predicted_prob_yes=predicted_prob_yes,
            entry_price_cents=entry_price_cents,
            contracts_traded=contracts_traded,
            trade_value_cents=trade_value_cents,
            resolution_time_utc=resolution_time_utc,
            status="OPEN",
            kelly_f_star=kelly_f_star,
            capital_at_trade_cents=capital_before_trade if cfg.USE_KELLY_CRITERION else None
        )

        self.open_trades[trade_id] = trade
        if cfg.USE_KELLY_CRITERION: # Only deduct if using Kelly, as it implies managing a balance
            self.current_capital_cents -= trade_value_cents
        
        log_paper_trade(trade) # Log the newly opened trade
        logger.info(f"Opened paper trade {trade_id}: {action} {contracts_traded} of {market_ticker} "
                    f"@ {entry_price_cents}c. Cost: {trade_value_cents/100:.2f} USD. "
                    f"New paper capital: {self.current_capital_cents/100:.2f} USD (if Kelly).")
        return trade

    def close_paper_trade(self, trade_id: str, actual_outcome: str) -> bool: # actual_outcome: "YES" or "NO"
        if trade_id not in self.open_trades:
            logger.warning(f"Attempted to close non-existent or already closed trade_id: {trade_id}")
            return False

        trade = self.open_trades.pop(trade_id)
        trade.actual_outcome = actual_outcome.upper()
        trade.status = "CLOSED"
        
        pnl_per_contract_cents = 0
        if trade.action == "BUY_YES":
            if trade.actual_outcome == "YES":
                pnl_per_contract_cents = 100 - trade.entry_price_cents
            else: # Outcome was NO
                pnl_per_contract_cents = -trade.entry_price_cents
        elif trade.action == "BUY_NO":
            if trade.actual_outcome == "NO":
                pnl_per_contract_cents = 100 - trade.entry_price_cents
            else: # Outcome was YES
                pnl_per_contract_cents = -trade.entry_price_cents
        
        trade.pnl_cents = pnl_per_contract_cents * trade.contracts_traded
        self.total_pnl_cents += trade.pnl_cents # Aggregate P&L
        
        if cfg.USE_KELLY_CRITERION: # Add back the value of the contract + PNL
            # The cost was already deducted. So, if won, add 100*contracts. If lost, add 0.
            # Or, more simply, add the PNL to the capital.
            self.current_capital_cents += trade.pnl_cents + trade.trade_value_cents # Add back original cost + PNL
                                                                                   # This is: current_capital - cost + cost + pnl
                                                                                   # which simplifies to current_capital + pnl
            # Let's re-verify Kelly capital update:
            # Capital was: C_before_trade.
            # Trade cost: Cost = entry_price * contracts.
            # Capital after open: C_after_open = C_before_trade - Cost.
            # PNL = (outcome_price - entry_price) * contracts.
            # Outcome_price is 100 if win, 0 if loss (for the contract itself).
            # So PNL = (100 - entry_price) * contracts if BUY_YES wins.
            # PNL = (0 - entry_price) * contracts if BUY_YES loses.
            # Capital after close = C_after_open + outcome_price * contracts
            # Capital after close = (C_before_trade - Cost) + outcome_price * contracts
            # Capital after close = C_before_trade - (entry_price * contracts) + (outcome_price * contracts)
            # Capital after close = C_before_trade + (outcome_price - entry_price) * contracts
            # Capital after close = C_before_trade + PNL_from_this_trade
            # So, the self.current_capital_cents at this point is C_after_open.
            # To get C_after_close, we need to add (outcome_price * contracts).
            # Or, we have trade.pnl_cents.  And we have trade.trade_value_cents (which is the cost).
            # current_capital_cents (which is C_after_open) should become C_after_open + PNL + cost_of_trade
            # self.current_capital_cents += trade.pnl_cents # This is simpler: new_capital = old_capital + pnl
            # The line `self.current_capital_cents -= trade_value_cents` was correct on open.
            # So on close, we just add the PNL: `self.current_capital_cents += trade.pnl_cents`
            # Let's correct the line above.
            # When opening, capital becomes C - cost.
            # When closing, capital becomes (C - cost) + PNL + cost = C + PNL. This is correct.
            # The current line is: self.current_capital_cents += trade.pnl_cents
            # Example: Capital 500. Cost 60. PNL 40.
            # Open: Capital = 500 - 60 = 440.
            # Close: Capital = 440 + 40 = 480. This is INCORRECT.
            # If BUY_YES @ 60 wins (outcome is 100): PNL = (100-60) = 40. You get 100 back.
            # Capital after close = (Capital_before_open - Cost) + (Contracts * 100)
            # Capital after close = Capital_before_open + PNL
            # So, my initial update was `self.current_capital_cents += trade.pnl_cents`.
            # Let's trace:
            # C_start = 50000
            # Trade 1: BUY_YES @ 60c, 10 contracts. Cost = 600c.
            #   current_capital_cents = 50000 - 600 = 49400
            # Trade 1 Wins: PNL per_contract = 100 - 60 = 40c. Total PNL = 40 * 10 = 400c.
            #   current_capital_cents = 49400 + 400 = 49800. (This should be 50000 + 400 = 50400)
            # The PNL is the *change* in value. The capital should reflect the total value.
            # When a trade is opened, the cash decreases by `trade_value_cents`.
            # When a trade closes:
            #   If WON: cash increases by `contracts_traded * 100`.
            #   If LOST: cash increases by `0`.
            # So, the change in capital is `(contracts_traded * settlement_price_of_contract) - trade_value_cents`.
            # `trade.pnl_cents` already calculates this correctly.
            # So, `self.current_capital_cents += trade.pnl_cents` is correct if `current_capital_cents`
            # reflects the cash balance *after* the cost of the trade was deducted.
            
            # Let's simplify:
            # On Open: self.current_capital_cents -= trade.trade_value_cents (Cash out for contracts)
            # On Close (Win): self.current_capital_cents += trade.contracts_traded * 100 (Cash in from settlement)
            # On Close (Loss): self.current_capital_cents += 0 (Contracts expire worthless, cash already out)
            # The PNL is (contracts_traded * 100) - trade.trade_value_cents for a win.
            # The PNL is (0) - trade.trade_value_cents for a loss.
            # So, self.current_capital_cents (after open) + (contracts_traded * settlement_value)
            # is the new capital.
            # The current `self.current_capital_cents` ALREADY had `trade.trade_value_cents` removed.
            # So if the trade wins, we add `trade.contracts_traded * 100` to it.
            # If the trade loses, we add `0` to it.
            
            if trade.action == "BUY_YES" and trade.actual_outcome == "YES":
                self.current_capital_cents += trade.contracts_traded * 100
            elif trade.action == "BUY_NO" and trade.actual_outcome == "NO":
                self.current_capital_cents += trade.contracts_traded * 100
            # If lost, no cash comes back, the cost was already deducted.

        self.closed_trades.append(trade)
        update_trade_in_log(trade) # Log the updated (closed) trade details

        logger.info(f"Closed paper trade {trade_id} ({trade.market_ticker}). Outcome: {actual_outcome}, "
                    f"P&L: {trade.pnl_cents/100:.2f} USD. "
                    f"New paper capital: {self.current_capital_cents/100:.2f} USD (if Kelly).")
        return True

    def get_open_trades_for_market(self, market_ticker: str) -> List[PaperTrade]:
        return [t for t in self.open_trades.values() if t.market_ticker == market_ticker]

    def get_summary(self) -> dict:
        num_open = len(self.open_trades)
        num_closed = len(self.closed_trades)
        wins = sum(1 for t in self.closed_trades if t.pnl_cents is not None and t.pnl_cents > 0)
        losses = sum(1 for t in self.closed_trades if t.pnl_cents is not None and t.pnl_cents < 0)
        
        return {
            "initial_capital_usd": self.initial_capital_cents / 100,
            "current_capital_usd": self.current_capital_cents / 100,
            "total_pnl_usd": self.total_pnl_cents / 100,
            "num_open_trades": num_open,
            "num_closed_trades": num_closed,
            "num_wins": wins,
            "num_losses": losses,
            "win_rate": wins / num_closed if num_closed > 0 else 0
        }

    def check_and_close_resolved_trades(self, current_utc_time: datetime, get_market_outcome_func):
        """
        Iterates through open trades and closes them if they have resolved.
        `get_market_outcome_func` is a function callback that takes a market_ticker
        and returns its outcome ("YES", "NO", or None if not resolved/error).
        """
        resolved_trade_ids = []
        for trade_id, trade in self.open_trades.items():
            # Add a small buffer (e.g., 5 minutes) past resolution time before trying to get outcome
            if current_utc_time > (trade.resolution_time_utc + timedelta(minutes=5)):
                logger.info(f"Trade {trade_id} for {trade.market_ticker} (resolves {trade.resolution_time_utc}) is past resolution. Attempting to fetch outcome.")
                outcome = get_market_outcome_func(trade.market_ticker)
                if outcome:
                    self.close_paper_trade(trade_id, outcome)
                    # Don't add to resolved_trade_ids here, close_paper_trade pops it
                else:
                    logger.warning(f"Could not get outcome for resolved market {trade.market_ticker} yet.")
            # No need to collect IDs to pop later, close_paper_trade handles it.