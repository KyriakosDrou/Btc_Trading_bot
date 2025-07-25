from app import db
from datetime import datetime, timezone, timedelta
from sqlalchemy import func
import pytz

# Athens timezone (UTC+2)
ATHENS_TZ = pytz.timezone('Europe/Athens')

def athens_now():
    """Get current Athens time"""
    return datetime.now(ATHENS_TZ)

class Portfolio(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    symbol = db.Column(db.String(10), nullable=False)
    quantity = db.Column(db.Float, nullable=False, default=0.0)  # Positive = Long, Negative = Short
    avg_price = db.Column(db.Float, nullable=False, default=0.0)
    current_price = db.Column(db.Float, nullable=False, default=0.0)
    last_updated = db.Column(db.DateTime, default=athens_now)
    
    @property
    def is_long(self):
        return self.quantity > 0
    
    @property
    def is_short(self):
        return self.quantity < 0
    
    @property
    def position_type(self):
        if self.quantity > 0:
            return "LONG"
        elif self.quantity < 0:
            return "SHORT"
        else:
            return "FLAT"
    
    @property
    def market_value(self):
        # For long positions: quantity * current_price (asset value)
        # For short positions: -(abs(quantity) * current_price) (liability value)
        if self.is_long:
            return self.quantity * self.current_price
        else:
            # Short position shows the liability (negative value)
            return -(abs(self.quantity) * self.current_price)
    
    @property
    def total_cost(self):
        # Cost basis for the position
        return abs(self.quantity) * self.avg_price
    
    @property
    def unrealized_pnl(self):
        if self.quantity == 0:
            return 0
        
        if self.is_long:
            # Long P&L: (current_price - entry_price) * quantity
            return (self.current_price - self.avg_price) * self.quantity
        else:
            # Short P&L: (entry_price - current_price) * abs(quantity)
            return (self.avg_price - self.current_price) * abs(self.quantity)
    
    @property
    def unrealized_pnl_percent(self):
        if self.total_cost == 0:
            return 0
        return (self.unrealized_pnl / self.total_cost) * 100

class Trade(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    symbol = db.Column(db.String(10), nullable=False)
    side = db.Column(db.String(15), nullable=False)  # 'BUY', 'SELL', 'SHORT', 'COVER'
    quantity = db.Column(db.Float, nullable=False)
    price = db.Column(db.Float, nullable=False)
    total_value = db.Column(db.Float, nullable=False)
    strategy = db.Column(db.String(50), nullable=True)
    timestamp = db.Column(db.DateTime, default=athens_now)
    status = db.Column(db.String(20), default='COMPLETED')
    duration = db.Column(db.Integer, nullable=True)  # Trade duration in seconds
    pnl = db.Column(db.Float, nullable=True)  # Profit/Loss in EUR
    
    # New fields for trade reasoning
    signal_details = db.Column(db.JSON, nullable=True)  # Technical indicators and values
    reasoning = db.Column(db.Text, nullable=True)  # Human-readable trade logic
    market_conditions = db.Column(db.JSON, nullable=True)  # Market state at trade time
    risk_trigger = db.Column(db.String(100), nullable=True)  # Risk management reason (stop loss, take profit, etc.)
    
    def __repr__(self):
        return f'<Trade {self.side} {self.quantity} {self.symbol} @ {self.price}>'

class TradingStrategy(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), nullable=False)
    symbol = db.Column(db.String(10), nullable=False)
    enabled = db.Column(db.Boolean, default=True)
    parameters = db.Column(db.JSON, nullable=True)
    created_at = db.Column(db.DateTime, default=athens_now)
    last_executed = db.Column(db.DateTime, nullable=True)

class MarketData(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    symbol = db.Column(db.String(10), nullable=False)
    price = db.Column(db.Float, nullable=False)
    volume = db.Column(db.Integer, nullable=True)
    timestamp = db.Column(db.DateTime, default=athens_now)
    
    def __repr__(self):
        return f'<MarketData {self.symbol}: {self.price}>'

class BotSettings(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    key = db.Column(db.String(50), nullable=False, unique=True)
    value = db.Column(db.String(500), nullable=False)
    updated_at = db.Column(db.DateTime, default=athens_now)

class RealTimeMarketData(db.Model):
    """Real-time market data logging with technical indicators"""
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=athens_now, nullable=False)
    symbol = db.Column(db.String(10), nullable=False, default='BTC')
    price = db.Column(db.Float, nullable=False)
    volume_24h = db.Column(db.Float, nullable=True)
    price_change_24h = db.Column(db.Float, nullable=True)
    
    # Technical Indicators
    rsi_14 = db.Column(db.Float, nullable=True)
    ma_fast = db.Column(db.Float, nullable=True)  # 10-period MA
    ma_slow = db.Column(db.Float, nullable=True)  # 20-period MA
    bb_upper = db.Column(db.Float, nullable=True)  # Bollinger Band Upper
    bb_middle = db.Column(db.Float, nullable=True)  # Bollinger Band Middle
    bb_lower = db.Column(db.Float, nullable=True)  # Bollinger Band Lower
    
    # Market Conditions
    market_trend = db.Column(db.String(20), nullable=True)  # 'BULLISH', 'BEARISH', 'NEUTRAL'
    volatility = db.Column(db.Float, nullable=True)
    api_source = db.Column(db.String(50), nullable=True, default='CoinGecko')
    
    def __repr__(self):
        return f'<RealTimeMarketData {self.symbol} {self.price} @ {self.timestamp}>'

class StrategyStatus(db.Model):
    """Current status and trigger levels for each trading strategy"""
    id = db.Column(db.Integer, primary_key=True)
    strategy_name = db.Column(db.String(50), nullable=False)
    symbol = db.Column(db.String(10), nullable=False, default='BTC')
    current_state = db.Column(db.String(30), nullable=False)  # 'MONITORING', 'SIGNAL_TRIGGERED', 'WAITING'
    next_action = db.Column(db.String(100), nullable=True)  # Human-readable next action
    trigger_levels = db.Column(db.JSON, nullable=True)  # Current trigger conditions
    last_evaluation = db.Column(db.DateTime, default=athens_now)
    next_check_time = db.Column(db.DateTime, nullable=True)
    signal_strength = db.Column(db.Float, nullable=True)  # 0-100 signal confidence
    position_bias = db.Column(db.String(20), nullable=True)  # 'BULLISH', 'BEARISH', 'NEUTRAL'
    
    def __repr__(self):
        return f'<StrategyStatus {self.strategy_name}: {self.current_state}>'

class BotActivityLog(db.Model):
    """Comprehensive bot activity and system health monitoring"""
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=athens_now, nullable=False)
    activity_type = db.Column(db.String(30), nullable=False)  # 'STRATEGY_EVAL', 'API_CALL', 'TRADE_EXEC', 'ERROR', 'HEARTBEAT'
    description = db.Column(db.Text, nullable=False)
    details = db.Column(db.JSON, nullable=True)  # Additional structured data
    status = db.Column(db.String(20), default='SUCCESS')  # 'SUCCESS', 'ERROR', 'WARNING'
    execution_time_ms = db.Column(db.Integer, nullable=True)  # Performance tracking
    error_message = db.Column(db.Text, nullable=True)
    
    def __repr__(self):
        return f'<BotActivityLog {self.activity_type}: {self.status} @ {self.timestamp}>'

class DataExportLog(db.Model):
    """Track data exports and downloads for audit purposes"""
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=athens_now, nullable=False)
    export_type = db.Column(db.String(30), nullable=False)  # 'TRADES', 'MARKET_DATA', 'STRATEGY_STATUS', 'FULL_AUDIT'
    file_format = db.Column(db.String(10), nullable=False)  # 'CSV', 'EXCEL', 'JSON'
    date_range_start = db.Column(db.DateTime, nullable=True)
    date_range_end = db.Column(db.DateTime, nullable=True)
    filters_applied = db.Column(db.JSON, nullable=True)
    record_count = db.Column(db.Integer, nullable=False, default=0)
    file_size_kb = db.Column(db.Integer, nullable=True)
    
    def __repr__(self):
        return f'<DataExportLog {self.export_type} - {self.record_count} records>'

class TrailingStop(db.Model):
    """Trailing stop-loss tracking for active positions"""
    id = db.Column(db.Integer, primary_key=True)
    symbol = db.Column(db.String(10), nullable=False)
    position_type = db.Column(db.String(5), nullable=False)  # 'LONG' or 'SHORT'
    entry_price = db.Column(db.Float, nullable=False)
    stop_price = db.Column(db.Float, nullable=False)
    highest_price = db.Column(db.Float, nullable=True)  # For long positions
    lowest_price = db.Column(db.Float, nullable=True)   # For short positions
    trail_percent = db.Column(db.Float, nullable=False, default=2.0)  # 2% trailing
    created_at = db.Column(db.DateTime, default=athens_now)
    last_updated = db.Column(db.DateTime, default=athens_now)
    is_active = db.Column(db.Boolean, default=True)

class SupportResistanceLevel(db.Model):
    """Support and resistance levels detected by technical analysis"""
    id = db.Column(db.Integer, primary_key=True)
    symbol = db.Column(db.String(10), nullable=False, default='BTC')
    level_type = db.Column(db.String(10), nullable=False)  # 'SUPPORT' or 'RESISTANCE'
    price_level = db.Column(db.Float, nullable=False)
    strength = db.Column(db.Integer, nullable=False, default=1)  # Number of touches
    first_detected = db.Column(db.DateTime, default=athens_now)
    last_touched = db.Column(db.DateTime, default=athens_now)
    is_active = db.Column(db.Boolean, default=True)
    
    def __repr__(self):
        return f'<{self.level_type} {self.symbol} @ €{self.price_level}>'

class AdvancedSignal(db.Model):
    """Advanced trading signals from multiple timeframe and indicator analysis"""
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=athens_now, nullable=False)
    symbol = db.Column(db.String(10), nullable=False, default='BTC')
    signal_type = db.Column(db.String(10), nullable=False)  # 'BUY', 'SELL', 'NEUTRAL'
    confidence = db.Column(db.Float, nullable=False)  # 0-100 confidence score
    
    # Multi-timeframe analysis
    timeframe_1h_trend = db.Column(db.String(10), nullable=True)
    timeframe_4h_trend = db.Column(db.String(10), nullable=True)
    timeframe_1d_trend = db.Column(db.String(10), nullable=True)
    timeframe_confluence = db.Column(db.Boolean, default=False)
    
    # Volume analysis
    volume_signal = db.Column(db.String(10), nullable=True)
    volume_surge = db.Column(db.Boolean, default=False)
    volume_trend = db.Column(db.String(15), nullable=True)
    
    # Technical indicators
    macd_signal = db.Column(db.String(10), nullable=True)
    macd_line = db.Column(db.Float, nullable=True)
    macd_signal_line = db.Column(db.Float, nullable=True)
    macd_histogram = db.Column(db.Float, nullable=True)
    
    stoch_rsi_signal = db.Column(db.String(10), nullable=True)
    stoch_rsi_value = db.Column(db.Float, nullable=True)
    stoch_rsi_condition = db.Column(db.String(20), nullable=True)
    
    # Support/resistance context
    sr_context = db.Column(db.String(20), nullable=True)
    nearest_support = db.Column(db.Float, nullable=True)
    nearest_resistance = db.Column(db.Float, nullable=True)
    
    # Combined reasoning
    reasoning = db.Column(db.Text, nullable=True)
    components_json = db.Column(db.JSON, nullable=True)  # Detailed component analysis
    
    def __repr__(self):
        return f'<AdvancedSignal {self.signal_type} {self.symbol} confidence:{self.confidence}%>'

class Account(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    cash_balance = db.Column(db.Float, nullable=False, default=1000.0)  # Starting with €1k paper money
    total_value = db.Column(db.Float, nullable=False, default=1000.0)
    realized_pnl = db.Column(db.Float, nullable=False, default=0.0)
    unrealized_pnl = db.Column(db.Float, nullable=False, default=0.0)
    last_updated = db.Column(db.DateTime, default=athens_now)
