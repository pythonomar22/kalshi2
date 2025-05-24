# /paper/data_models.py

from pydantic import BaseModel, Field, validator, model_validator
from typing import Dict, Optional, List, Any, Deque 
from datetime import datetime, timezone, timedelta # Import timedelta

class BinanceKlineInternal(BaseModel): 
    kline_start_time: int = Field(alias="t")    
    kline_end_time: int = Field(alias="T")      
    symbol_k: str = Field(alias="s")            
    interval: str = Field(alias="i")            
    open_price: str = Field(alias="o")          
    close_price: str = Field(alias="c")         
    high_price: str = Field(alias="h")          
    low_price: str = Field(alias="l")           
    base_volume: str = Field(alias="v")         
    num_trades: int = Field(alias="n")          
    is_closed: bool = Field(alias="x")          
    quote_volume: str = Field(alias="q")        
    taker_buy_base_volume: str = Field(alias="V") 
    taker_buy_quote_volume: str = Field(alias="Q")

class BinanceKline(BaseModel):
    stream_event_type: Optional[str] = Field(default=None, alias="e") 
    stream_event_time: Optional[int] = Field(default=None, alias="E")   
    stream_symbol: Optional[str] = Field(default=None, alias="s")       
    kline_data: Optional[BinanceKlineInternal] = Field(default=None, alias="k") 

    rest_kline_start_time: Optional[int] = None
    rest_open_price: Optional[str] = None
    rest_high_price: Optional[str] = None
    rest_low_price: Optional[str] = None
    rest_close_price: Optional[str] = None
    rest_base_volume: Optional[str] = None
    rest_kline_close_time: Optional[int] = None # This will be derived for HashKey
    rest_quote_volume: Optional[str] = None
    rest_num_trades: Optional[int] = None
    rest_taker_buy_base_volume: Optional[str] = None
    rest_taker_buy_quote_volume: Optional[str] = None
    rest_interval: Optional[str] = None 
    rest_symbol: Optional[str] = None
    rest_is_closed: bool = True 

    reception_timestamp_utc: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @classmethod
    def from_hashkey_api_list(cls, kline_data_list: list, symbol: str, interval_str: str) -> 'BinanceKline':
        """
        Creates a BinanceKline instance from the HashKey REST API list format.
        HashKey format: [kline_open_time(0), open(1), high(2), low(3), close(4), volume(5), 
                         kline_close_time(6, but seems to be 0 in sample), quote_asset_volume(7), 
                         number_of_trades(8), taker_buy_base_asset_volume(9), 
                         taker_buy_quote_asset_volume(10)]
        """
        open_time_ms = int(kline_data_list[0])
        
        # Derive close_time_ms based on interval
        # Assuming interval_str is like "1min", "5min", "1h" etc.
        # For "1min", duration is 60,000 ms. Close time is open_time + duration - 1ms.
        duration_ms = 0
        if interval_str == "1min":
            duration_ms = 60 * 1000
        elif interval_str == "5min":
            duration_ms = 5 * 60 * 1000
        # Add other intervals as needed
        else: # Default to 1 min if interval unknown for calculation
            duration_ms = 60 * 1000 
            
        derived_close_time_ms = open_time_ms + duration_ms - 1

        return cls(
            rest_kline_start_time=open_time_ms,
            rest_open_price=str(kline_data_list[1]),
            rest_high_price=str(kline_data_list[2]),
            rest_low_price=str(kline_data_list[3]),
            rest_close_price=str(kline_data_list[4]),
            rest_base_volume=str(kline_data_list[5]),
            rest_kline_close_time=derived_close_time_ms, # Use derived close time
            rest_quote_volume=str(kline_data_list[7]),
            rest_num_trades=int(kline_data_list[8]),
            rest_taker_buy_base_volume=str(kline_data_list[9]),
            rest_taker_buy_quote_volume=str(kline_data_list[10]),
            rest_symbol=symbol,
            rest_interval=interval_str,
            rest_is_closed=True, 
            reception_timestamp_utc=datetime.now(timezone.utc) 
        )

    # Properties remain the same
    @property
    def kline_start_time(self) -> int: 
        val = self.kline_data.kline_start_time if self.kline_data else self.rest_kline_start_time
        if val is None: raise ValueError("kline_start_time is None")
        return val
    @property
    def kline_end_time(self) -> int: 
        val = self.kline_data.kline_end_time if self.kline_data else self.rest_kline_close_time
        if val is None: raise ValueError("kline_end_time is None")
        return val
    @property
    def symbol(self) -> str: 
        val = self.kline_data.symbol_k if self.kline_data else self.rest_symbol
        if val is None: raise ValueError("symbol is None")
        return val
    @property
    def interval(self) -> str: 
        val = self.kline_data.interval if self.kline_data else self.rest_interval
        if val is None: raise ValueError("interval is None")
        return val
    @property
    def open_price(self) -> str: 
        val = self.kline_data.open_price if self.kline_data else self.rest_open_price
        if val is None: raise ValueError("open_price is None")
        return val
    @property
    def close_price(self) -> str: 
        val = self.kline_data.close_price if self.kline_data else self.rest_close_price
        if val is None: raise ValueError("close_price is None")
        return val
    @property
    def high_price(self) -> str: 
        val = self.kline_data.high_price if self.kline_data else self.rest_high_price
        if val is None: raise ValueError("high_price is None")
        return val
    @property
    def low_price(self) -> str: 
        val = self.kline_data.low_price if self.kline_data else self.rest_low_price
        if val is None: raise ValueError("low_price is None")
        return val
    @property
    def base_volume(self) -> str: 
        val = self.kline_data.base_volume if self.kline_data else self.rest_base_volume
        if val is None: raise ValueError("base_volume is None")
        return val
    @property
    def num_trades(self) -> int: 
        val = self.kline_data.num_trades if self.kline_data else self.rest_num_trades
        if val is None: raise ValueError("num_trades is None")
        return val
    @property
    def is_closed(self) -> bool: 
        val = self.kline_data.is_closed if self.kline_data is not None else self.rest_is_closed
        if val is None: raise ValueError("is_closed is None") # Should always be true for REST, or present for WS
        return val
    
    @property
    def timestamp_s_utc(self) -> int: 
        return self.kline_start_time // 1000
    @property
    def open(self) -> float: return float(self.open_price)
    @property
    def high(self) -> float: return float(self.high_price)
    @property
    def low(self) -> float: return float(self.low_price)
    @property
    def close(self) -> float: return float(self.close_price)
    @property
    def volume(self) -> float: return float(self.base_volume)

# ... (KalshiMarketState, PaperTrade, KalshiMarketInfo remain the same) ...
class KalshiOrderBookLevel(BaseModel):
    price: int 
    quantity: int

class KalshiMarketState(BaseModel):
    market_ticker: str
    timestamp_utc: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    yes_book: Dict[int, int] = Field(default_factory=dict) 
    no_book: Dict[int, int] = Field(default_factory=dict)  
    sequence_num: int = 0
    ui_yes_bid_cents: Optional[int] = None
    ui_yes_bid_qty: Optional[int] = None
    ui_yes_ask_cents: Optional[int] = None
    ui_yes_ask_qty_on_no_side: Optional[int] = None 

    def update_ui_bid_ask(self):
        yes_bids_prices = sorted([p for p, q in self.yes_book.items() if q > 0], reverse=True)
        no_bids_prices = sorted([p for p, q in self.no_book.items() if q > 0], reverse=True)
        self.ui_yes_bid_cents = yes_bids_prices[0] if yes_bids_prices else None
        self.ui_yes_bid_qty = self.yes_book.get(self.ui_yes_bid_cents, 0) if self.ui_yes_bid_cents is not None else 0
        self.ui_yes_ask_cents = (100 - no_bids_prices[0]) if no_bids_prices else None
        self.ui_yes_ask_qty_on_no_side = 0
        if self.ui_yes_ask_cents is not None:
            original_no_bid_price = 100 - self.ui_yes_ask_cents
            self.ui_yes_ask_qty_on_no_side = self.no_book.get(original_no_bid_price, 0)

class PaperTrade(BaseModel):
    trade_id: str 
    market_ticker: str
    decision_timestamp_utc: datetime
    action: str 
    predicted_prob_yes: float
    entry_price_cents: int 
    contracts_traded: int
    trade_value_cents: int 
    resolution_time_utc: datetime
    actual_outcome: Optional[str] = None 
    pnl_cents: Optional[int] = None
    status: str = "OPEN" 
    kelly_f_star: Optional[float] = None
    capital_at_trade_cents: Optional[int] = None
    @validator('action')
    def action_must_be_valid(cls, v):
        if v not in ["BUY_YES", "BUY_NO"]:
            raise ValueError('action must be BUY_YES or BUY_NO')
        return v

class KalshiMarketInfo(BaseModel):
    ticker: str
    series_ticker: str
    event_ticker: str
    strike_price: float
    open_time_utc: datetime
    close_time_utc: datetime 
    status: str 
    last_price_yes_cents: Optional[int] = None