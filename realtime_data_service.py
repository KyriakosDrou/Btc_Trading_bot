"""
Real-time Trading Database Service
Continuously logs market data, strategy status, bot activities, and provides export capabilities.
"""

import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from app import db
from models import (
    RealTimeMarketData, StrategyStatus, BotActivityLog, DataExportLog,
    athens_now, Trade, Portfolio, Account
)
from market_data import MarketDataService
from trading_strategies import MovingAverageStrategy, RSIStrategy, BollingerBandsStrategy
import logging
import pandas as pd
import io

logger = logging.getLogger(__name__)

class RealTimeTradingDataService:
    def __init__(self):
        self.market_data_service = MarketDataService()
        self.strategies = {
            'RSI Strategy': RSIStrategy('BTC'),
            'MA Strategy': MovingAverageStrategy('BTC'),
            'Bollinger Bands Strategy': BollingerBandsStrategy('BTC')
        }
        
    def log_market_data_with_indicators(self, symbol: str = 'BTC') -> bool:
        """Log current market data with all technical indicators"""
        start_time = time.time()
        
        try:
            # Get real-time price data
            detailed_data = self.market_data_service.get_detailed_market_data(symbol)
            
            # Debug: Log what we received
            logger.debug(f"Market data received: {detailed_data}")
            
            if not detailed_data or 'price' not in detailed_data:
                # Fallback to simple price method
                simple_price = self.market_data_service.get_real_time_price(symbol)
                if simple_price:
                    detailed_data = {
                        'price': simple_price,
                        'volume_24h': 0,
                        'price_change_24h': 0
                    }
                    self.log_bot_activity('API_CALL', f'Using fallback price for {symbol}: €{simple_price:.2f}', 
                                        status='WARNING')
                else:
                    self.log_bot_activity('API_CALL', 'No price data available', 
                                        status='ERROR', error_message='Unable to fetch Bitcoin price')
                    return False
            
            # Get historical data for technical indicators
            historical_data = self.market_data_service.get_historical_data(symbol, period='1mo')
            if not historical_data or len(historical_data) < 20:
                # Use database prices as fallback
                recent_market_data = db.session.query(RealTimeMarketData).filter_by(symbol=symbol).order_by(
                    RealTimeMarketData.timestamp.desc()
                ).limit(30).all()
                
                if recent_market_data:
                    prices = [float(record.price) for record in reversed(recent_market_data)]
                else:
                    # Create minimal record with current price only
                    market_record = RealTimeMarketData(
                        symbol=symbol,
                        price=detailed_data['price'],
                        volume_24h=detailed_data.get('volume_24h'),
                        price_change_24h=detailed_data.get('price_change_24h'),
                        api_source='CoinGecko'
                    )
                    db.session.add(market_record)
                    db.session.commit()
                    
                    self.log_bot_activity('API_CALL', 'Market data logged without indicators (insufficient history)', 
                                        status='WARNING')
                    return True
            else:
                # Calculate technical indicators from API data
                # Debug: Check historical data structure
                logger.debug(f"Historical data sample: {historical_data[:2] if historical_data else 'None'}")
                try:
                    # Historical data uses 'close' prices, not 'price'
                    prices = [float(d['close']) for d in historical_data[-30:]]  # Last 30 prices
                except (KeyError, TypeError) as e:
                    logger.error(f"Error accessing close price in historical data: {e}")
                    # Fallback to using only current price
                    prices = [float(detailed_data['price'])]
            
            # Calculate technical indicators (for both API and database fallback data)
            if len(prices) >= 14:
                # RSI calculation
                rsi_value = self._calculate_rsi(prices)
                
                # Moving averages
                ma_fast = sum(prices[-10:]) / 10 if len(prices) >= 10 else None
                ma_slow = sum(prices[-20:]) / 20 if len(prices) >= 20 else None
                
                # Bollinger Bands
                bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(prices) if len(prices) >= 20 else (None, None, None)
                
                # Market trend analysis
                market_trend = self._determine_market_trend(prices, ma_fast, ma_slow, rsi_value)
                
                # Volatility calculation
                if len(prices) >= 10:
                    recent_prices = prices[-10:]
                    avg_price = sum(recent_prices) / len(recent_prices)
                    variance = sum((p - avg_price) ** 2 for p in recent_prices) / len(recent_prices)
                    volatility = (variance ** 0.5) / avg_price * 100  # Coefficient of variation as %
                else:
                    volatility = None
            else:
                # Minimal indicators for insufficient data
                rsi_value = None
                ma_fast = None
                ma_slow = None
                bb_upper, bb_middle, bb_lower = None, None, None
                market_trend = 'NEUTRAL'
                volatility = None

            
            # Create market data record
            market_record = RealTimeMarketData(
                symbol=symbol,
                price=detailed_data['price'],
                volume_24h=detailed_data.get('volume_24h'),
                price_change_24h=detailed_data.get('price_change_24h'),
                rsi_14=rsi_value,
                ma_fast=ma_fast,
                ma_slow=ma_slow,
                bb_upper=bb_upper,
                bb_middle=bb_middle,
                bb_lower=bb_lower,
                market_trend=market_trend,
                volatility=volatility,
                api_source='CoinGecko'
            )
            
            db.session.add(market_record)
            db.session.commit()
            
            execution_time = int((time.time() - start_time) * 1000)
            
            # Log successful market data collection
            rsi_display = f"{rsi_value:.1f}" if rsi_value is not None else "N/A"
            self.log_bot_activity(
                'API_CALL',
                f'Market data logged: {symbol} at €{detailed_data["price"]:.2f}, RSI: {rsi_display}',
                details={
                    'price': detailed_data['price'],
                    'rsi': rsi_value,
                    'ma_fast': ma_fast,
                    'ma_slow': ma_slow,
                    'trend': market_trend,
                    'volatility': volatility
                },
                execution_time_ms=execution_time
            )
            
            return True
            
        except Exception as e:
            execution_time = int((time.time() - start_time) * 1000)
            error_msg = f"Error logging market data: {str(e)}"
            logger.error(error_msg)
            
            self.log_bot_activity(
                'API_CALL',
                'Market data logging failed',
                status='ERROR',
                error_message=error_msg,
                execution_time_ms=execution_time
            )
            return False
    
    def update_strategy_status(self) -> None:
        """Update current status for all trading strategies"""
        start_time = time.time()
        
        try:
            # Get recent market data for analysis
            historical_data = self.market_data_service.get_historical_data('BTC', period='1mo')
            if not historical_data or len(historical_data) < 20:
                return
            
            # Use 'close' price from historical data, not 'price'
            current_price = float(historical_data[-1]['close'])
            
            # Get current portfolio position
            portfolio_item = db.session.query(Portfolio).filter_by(symbol='BTC').first()
            current_position = portfolio_item.quantity if portfolio_item else 0
            
            for strategy_name, strategy_obj in self.strategies.items():
                try:
                    # Calculate current signals
                    should_buy = strategy_obj.should_buy(historical_data)
                    should_sell = strategy_obj.should_sell(historical_data, current_position) if current_position > 0 else False
                    should_short = strategy_obj.should_short(historical_data)
                    should_cover = strategy_obj.should_cover(historical_data, current_position) if current_position < 0 else False
                    
                    # Determine current state and next action
                    if should_buy:
                        current_state = 'SIGNAL_TRIGGERED'
                        next_action = f'BUY signal active - Ready to purchase at €{current_price:.2f}'
                        signal_strength = 85
                        position_bias = 'BULLISH'
                    elif should_sell:
                        current_state = 'SIGNAL_TRIGGERED'
                        next_action = f'SELL signal active - Ready to close long position at €{current_price:.2f}'
                        signal_strength = 80
                        position_bias = 'BEARISH'
                    elif should_short:
                        current_state = 'SIGNAL_TRIGGERED'
                        next_action = f'SHORT signal active - Ready to short at €{current_price:.2f}'
                        signal_strength = 75
                        position_bias = 'BEARISH'
                    elif should_cover:
                        current_state = 'SIGNAL_TRIGGERED'
                        next_action = f'COVER signal active - Ready to close short position at €{current_price:.2f}'
                        signal_strength = 80
                        position_bias = 'BULLISH'
                    else:
                        current_state = 'MONITORING'
                        next_action = self._get_monitoring_description(strategy_name, historical_data)
                        signal_strength = 30
                        position_bias = 'NEUTRAL'
                    
                    # Get trigger levels specific to strategy
                    trigger_levels = self._get_strategy_trigger_levels(strategy_name, historical_data)
                    
                    # Update or create strategy status
                    status_record = db.session.query(StrategyStatus).filter_by(
                        strategy_name=strategy_name, symbol='BTC'
                    ).first()
                    
                    if status_record:
                        status_record.current_state = current_state
                        status_record.next_action = next_action
                        status_record.trigger_levels = trigger_levels
                        status_record.last_evaluation = athens_now()
                        status_record.next_check_time = athens_now() + timedelta(seconds=30)
                        status_record.signal_strength = signal_strength
                        status_record.position_bias = position_bias
                    else:
                        status_record = StrategyStatus(
                            strategy_name=strategy_name,
                            symbol='BTC',
                            current_state=current_state,
                            next_action=next_action,
                            trigger_levels=trigger_levels,
                            next_check_time=athens_now() + timedelta(seconds=30),
                            signal_strength=signal_strength,
                            position_bias=position_bias
                        )
                        db.session.add(status_record)
                    
                except Exception as e:
                    logger.error(f"Error updating strategy status for {strategy_name}: {e}")
                    continue
            
            db.session.commit()
            
            execution_time = int((time.time() - start_time) * 1000)
            self.log_bot_activity(
                'STRATEGY_EVAL',
                f'Updated status for {len(self.strategies)} strategies',
                details={'strategies_count': len(self.strategies)},
                execution_time_ms=execution_time
            )
            
        except Exception as e:
            execution_time = int((time.time() - start_time) * 1000)
            error_msg = f"Error updating strategy status: {str(e)}"
            logger.error(error_msg)
            
            self.log_bot_activity(
                'STRATEGY_EVAL',
                'Strategy status update failed',
                status='ERROR',
                error_message=error_msg,
                execution_time_ms=execution_time
            )
    
    def log_bot_activity(self, activity_type: str, description: str, 
                        details: Dict = None, status: str = 'SUCCESS', 
                        execution_time_ms: int = None, error_message: str = None) -> None:
        """Log bot activity with detailed information"""
        try:
            activity_log = BotActivityLog(
                activity_type=activity_type,
                description=description,
                details=details,
                status=status,
                execution_time_ms=execution_time_ms,
                error_message=error_message
            )
            
            db.session.add(activity_log)
            db.session.commit()
            
        except Exception as e:
            logger.error(f"Failed to log bot activity: {e}")
    
    def log_trade_execution(self, trade: 'Trade') -> None:
        """Log detailed trade execution information"""
        try:
            trade_details = {
                'trade_id': trade.id,
                'symbol': trade.symbol,
                'side': trade.side,
                'quantity': float(trade.quantity),
                'price': float(trade.price),
                'total_value': float(trade.total_value),
                'strategy': trade.strategy,
                'pnl': float(trade.pnl) if trade.pnl else None,
                'signal_details': trade.signal_details,
                'market_conditions': trade.market_conditions,
                'risk_trigger': trade.risk_trigger
            }
            
            self.log_bot_activity(
                'TRADE_EXEC',
                f'{trade.side} {trade.quantity:.8f} {trade.symbol} at €{trade.price:.2f}',
                details=trade_details
            )
            
        except Exception as e:
            logger.error(f"Failed to log trade execution: {e}")
    
    def get_current_bot_status(self) -> Dict[str, Any]:
        """Get comprehensive current bot status"""
        try:
            # Get latest activities
            recent_activities = db.session.query(BotActivityLog).order_by(
                BotActivityLog.timestamp.desc()
            ).limit(10).all()
            
            # Get strategy statuses
            strategy_statuses = db.session.query(StrategyStatus).filter_by(symbol='BTC').all()
            
            # Get latest market data
            latest_market_data = db.session.query(RealTimeMarketData).order_by(
                RealTimeMarketData.timestamp.desc()
            ).first()
            
            # System health metrics
            total_activities = db.session.query(BotActivityLog).count()
            error_count = db.session.query(BotActivityLog).filter_by(status='ERROR').count()
            success_rate = ((total_activities - error_count) / total_activities * 100) if total_activities > 0 else 0
            
            return {
                'system_health': {
                    'status': 'RUNNING',
                    'total_activities': total_activities,
                    'error_rate': round((error_count / total_activities * 100) if total_activities > 0 else 0, 2),
                    'success_rate': round(success_rate, 2),
                    'last_update': athens_now().isoformat()
                },
                'latest_market_data': {
                    'price': float(latest_market_data.price) if latest_market_data else None,
                    'rsi': float(latest_market_data.rsi_14) if latest_market_data and latest_market_data.rsi_14 else None,
                    'trend': latest_market_data.market_trend if latest_market_data else None,
                    'timestamp': latest_market_data.timestamp.isoformat() if latest_market_data else None
                },
                'strategy_overview': [
                    {
                        'name': status.strategy_name,
                        'state': status.current_state,
                        'next_action': status.next_action,
                        'signal_strength': float(status.signal_strength) if status.signal_strength else 0,
                        'bias': status.position_bias
                    } for status in strategy_statuses
                ],
                'recent_activities': [
                    {
                        'type': activity.activity_type,
                        'description': activity.description,
                        'status': activity.status,
                        'timestamp': activity.timestamp,  # Keep as datetime object for template
                        'execution_time_ms': activity.execution_time_ms
                    } for activity in recent_activities
                ]
            }
            
        except Exception as e:
            logger.error(f"Error getting bot status: {e}")
            return {
                'system_health': {'status': 'ERROR', 'error': str(e)},
                'latest_market_data': {},
                'strategy_overview': [],
                'recent_activities': []
            }
    
    def export_data(self, export_type: str, file_format: str = 'CSV', 
                   date_range_start: datetime = None, date_range_end: datetime = None,
                   filters: Dict = None) -> bytes:
        """Export trading data with comprehensive filtering"""
        try:
            filters = filters or {}
            
            if export_type == 'MARKET_DATA':
                data = self._export_market_data(date_range_start, date_range_end, filters)
            elif export_type == 'TRADES':
                data = self._export_trades(date_range_start, date_range_end, filters)
            elif export_type == 'STRATEGY_STATUS':
                data = self._export_strategy_status(date_range_start, date_range_end, filters)
            elif export_type == 'BOT_ACTIVITY':
                data = self._export_bot_activity(date_range_start, date_range_end, filters)
            elif export_type == 'FULL_AUDIT':
                data = self._export_full_audit(date_range_start, date_range_end, filters)
            else:
                raise ValueError(f"Unknown export type: {export_type}")
            
            # Convert to requested format
            if file_format.upper() == 'CSV':
                output = io.StringIO()
                data.to_csv(output, index=False)
                result = output.getvalue().encode('utf-8')
            elif file_format.upper() == 'EXCEL':
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    data.to_excel(writer, index=False, sheet_name=export_type)
                result = output.getvalue()
            else:
                raise ValueError(f"Unsupported file format: {file_format}")
            
            # Log export activity
            self._log_export_activity(export_type, file_format, date_range_start, 
                                    date_range_end, filters, len(data), len(result))
            
            return result
            
        except Exception as e:
            logger.error(f"Error exporting data: {e}")
            raise
    
    def _calculate_rsi(self, prices: List[float], period: int = 14) -> Optional[float]:
        """Calculate RSI indicator"""
        if len(prices) < period + 1:
            return None
        
        deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        gains = [delta if delta > 0 else 0 for delta in deltas]
        losses = [-delta if delta < 0 else 0 for delta in deltas]
        
        avg_gain = sum(gains[-period:]) / period
        avg_loss = sum(losses[-period:]) / period
        
        if avg_loss == 0:
            return 100
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return round(rsi, 2)
    
    def _calculate_bollinger_bands(self, prices: List[float], period: int = 20, std_dev: float = 2) -> tuple:
        """Calculate Bollinger Bands"""
        if len(prices) < period:
            return None, None, None
        
        recent_prices = prices[-period:]
        middle = sum(recent_prices) / period
        
        variance = sum((p - middle) ** 2 for p in recent_prices) / period
        std = variance ** 0.5
        
        upper = middle + (std_dev * std)
        lower = middle - (std_dev * std)
        
        return round(upper, 2), round(middle, 2), round(lower, 2)
    
    def _determine_market_trend(self, prices: List[float], ma_fast: float, 
                               ma_slow: float, rsi: float) -> str:
        """Determine overall market trend"""
        if not all([ma_fast, ma_slow, rsi]):
            return 'NEUTRAL'
        
        # Trend based on MA crossover and RSI
        if ma_fast > ma_slow and rsi < 70:
            return 'BULLISH'
        elif ma_fast < ma_slow and rsi > 30:
            return 'BEARISH'
        else:
            return 'NEUTRAL'
    
    def _get_monitoring_description(self, strategy_name: str, historical_data: List) -> str:
        """Get human-readable monitoring description for strategy"""
        current_price = float(historical_data[-1]['close'])
        
        if 'RSI' in strategy_name:
            prices = [float(d['close']) for d in historical_data[-14:]]
            rsi = self._calculate_rsi(prices) if len(prices) >= 14 else None
            if rsi:
                if rsi > 70:
                    return f'RSI: {rsi:.1f} (overbought) → Monitoring for SHORT signal'
                elif rsi < 30:
                    return f'RSI: {rsi:.1f} (oversold) → Monitoring for BUY signal'
                else:
                    return f'RSI: {rsi:.1f} (neutral) → Waiting for extreme levels'
        
        elif 'MA' in strategy_name:
            prices = [float(d['close']) for d in historical_data[-20:]]
            if len(prices) >= 20:
                ma_fast = sum(prices[-10:]) / 10
                ma_slow = sum(prices[-20:]) / 20
                if ma_fast > ma_slow:
                    return f'MA: Bullish crossover → Monitoring for continuation or reversal'
                else:
                    return f'MA: Bearish crossover → Monitoring for continuation or reversal'
        
        elif 'Bollinger' in strategy_name:
            prices = [float(d['close']) for d in historical_data[-20:]]
            bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(prices) if len(prices) >= 20 else (None, None, None)
            if all([bb_upper, bb_middle, bb_lower]):
                if current_price > bb_upper:
                    return f'Price above upper band (€{bb_upper:.2f}) → Monitoring for mean reversion'
                elif current_price < bb_lower:
                    return f'Price below lower band (€{bb_lower:.2f}) → Monitoring for bounce'
                else:
                    return f'Price in normal range → Monitoring for breakout'
        
        return f'Monitoring {strategy_name} → Waiting for signal conditions'
    
    def _get_strategy_trigger_levels(self, strategy_name: str, historical_data: List) -> Dict:
        """Get current trigger levels for strategy"""
        try:
            if 'RSI' in strategy_name:
                prices = [float(d['close']) for d in historical_data[-14:]]
                current_rsi = self._calculate_rsi(prices) if len(prices) >= 14 else None
                return {
                    'current_rsi': current_rsi,
                    'buy_trigger': 30,
                    'sell_trigger': 70,
                    'short_trigger': 70,
                    'cover_trigger': 30
                }
            
            elif 'MA' in strategy_name:
                prices = [float(d['close']) for d in historical_data[-20:]]
                if len(prices) >= 20:
                    ma_fast = sum(prices[-10:]) / 10
                    ma_slow = sum(prices[-20:]) / 20
                    return {
                        'ma_fast': round(ma_fast, 2),
                        'ma_slow': round(ma_slow, 2),
                        'crossover_signal': 'bullish' if ma_fast > ma_slow else 'bearish'
                    }
            
            elif 'Bollinger' in strategy_name:
                prices = [float(d['close']) for d in historical_data[-20:]]
                bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(prices) if len(prices) >= 20 else (None, None, None)
                current_price = float(historical_data[-1]['close'])
                return {
                    'current_price': current_price,
                    'bb_upper': bb_upper,
                    'bb_middle': bb_middle,
                    'bb_lower': bb_lower,
                    'buy_trigger': bb_lower,
                    'sell_trigger': bb_upper
                }
            
            return {}
            
        except Exception as e:
            logger.error(f"Error getting trigger levels for {strategy_name}: {e}")
            return {}
    
    def _export_market_data(self, start_date, end_date, filters) -> pd.DataFrame:
        """Export market data to DataFrame"""
        query = db.session.query(RealTimeMarketData)
        
        if start_date:
            query = query.filter(RealTimeMarketData.timestamp >= start_date)
        if end_date:
            query = query.filter(RealTimeMarketData.timestamp <= end_date)
        
        data = query.order_by(RealTimeMarketData.timestamp.desc()).all()
        
        return pd.DataFrame([
            {
                'Timestamp': record.timestamp.isoformat(),
                'Symbol': record.symbol,
                'Price (EUR)': record.price,
                'Volume 24h': record.volume_24h,
                'Price Change 24h': record.price_change_24h,
                'RSI': record.rsi_14,
                'MA Fast': record.ma_fast,
                'MA Slow': record.ma_slow,
                'BB Upper': record.bb_upper,
                'BB Middle': record.bb_middle,
                'BB Lower': record.bb_lower,
                'Market Trend': record.market_trend,
                'Volatility': record.volatility,
                'API Source': record.api_source
            } for record in data
        ])
    
    def _export_trades(self, start_date, end_date, filters) -> pd.DataFrame:
        """Export trade data to DataFrame"""
        query = db.session.query(Trade)
        
        if start_date:
            query = query.filter(Trade.timestamp >= start_date)
        if end_date:
            query = query.filter(Trade.timestamp <= end_date)
        
        if filters:
            if 'strategy' in filters:
                query = query.filter(Trade.strategy.like(f"%{filters['strategy']}%"))
            if 'side' in filters:
                query = query.filter(Trade.side == filters['side'])
        
        trades = query.order_by(Trade.timestamp.desc()).all()
        
        return pd.DataFrame([
            {
                'Timestamp': trade.timestamp.isoformat(),
                'Symbol': trade.symbol,
                'Side': trade.side,
                'Quantity': trade.quantity,
                'Price (EUR)': trade.price,
                'Total Value': trade.total_value,
                'Strategy': trade.strategy,
                'Status': trade.status,
                'Duration (sec)': trade.duration,
                'P&L (EUR)': trade.pnl,
                'Reasoning': trade.reasoning,
                'Signal Details': json.dumps(trade.signal_details) if trade.signal_details else None,
                'Market Conditions': json.dumps(trade.market_conditions) if trade.market_conditions else None,
                'Risk Trigger': trade.risk_trigger
            } for trade in trades
        ])
    
    def _export_strategy_status(self, start_date, end_date, filters) -> pd.DataFrame:
        """Export strategy status to DataFrame"""
        query = db.session.query(StrategyStatus)
        
        if start_date:
            query = query.filter(StrategyStatus.last_evaluation >= start_date)
        if end_date:
            query = query.filter(StrategyStatus.last_evaluation <= end_date)
        
        statuses = query.order_by(StrategyStatus.last_evaluation.desc()).all()
        
        return pd.DataFrame([
            {
                'Last Evaluation': status.last_evaluation.isoformat(),
                'Strategy Name': status.strategy_name,
                'Symbol': status.symbol,
                'Current State': status.current_state,
                'Next Action': status.next_action,
                'Signal Strength': status.signal_strength,
                'Position Bias': status.position_bias,
                'Trigger Levels': json.dumps(status.trigger_levels) if status.trigger_levels else None
            } for status in statuses
        ])
    
    def _export_bot_activity(self, start_date, end_date, filters) -> pd.DataFrame:
        """Export bot activity logs to DataFrame"""
        query = db.session.query(BotActivityLog)
        
        if start_date:
            query = query.filter(BotActivityLog.timestamp >= start_date)
        if end_date:
            query = query.filter(BotActivityLog.timestamp <= end_date)
        
        if filters:
            if 'activity_type' in filters:
                query = query.filter(BotActivityLog.activity_type == filters['activity_type'])
            if 'status' in filters:
                query = query.filter(BotActivityLog.status == filters['status'])
        
        activities = query.order_by(BotActivityLog.timestamp.desc()).all()
        
        return pd.DataFrame([
            {
                'Timestamp': activity.timestamp.isoformat(),
                'Activity Type': activity.activity_type,
                'Description': activity.description,
                'Status': activity.status,
                'Execution Time (ms)': activity.execution_time_ms,
                'Details': json.dumps(activity.details) if activity.details else None,
                'Error Message': activity.error_message
            } for activity in activities
        ])
    
    def _export_full_audit(self, start_date, end_date, filters) -> pd.DataFrame:
        """Export comprehensive audit trail"""
        # Combine all data sources for complete audit
        market_data = self._export_market_data(start_date, end_date, filters)
        trades = self._export_trades(start_date, end_date, filters)
        activities = self._export_bot_activity(start_date, end_date, filters)
        
        # Create comprehensive audit trail
        audit_records = []
        
        # Add market data records
        for _, row in market_data.iterrows():
            audit_records.append({
                'Timestamp': row['Timestamp'],
                'Type': 'MARKET_DATA',
                'Description': f"Price: €{row['Price (EUR)']} RSI: {row['RSI']} Trend: {row['Market Trend']}",
                'Details': json.dumps(row.to_dict())
            })
        
        # Add trade records
        for _, row in trades.iterrows():
            audit_records.append({
                'Timestamp': row['Timestamp'],
                'Type': 'TRADE',
                'Description': f"{row['Side']} {row['Quantity']} {row['Symbol']} at €{row['Price (EUR)']}",
                'Details': json.dumps(row.to_dict())
            })
        
        # Add activity records
        for _, row in activities.iterrows():
            audit_records.append({
                'Timestamp': row['Timestamp'],
                'Type': row['Activity Type'],
                'Description': row['Description'],
                'Details': json.dumps(row.to_dict())
            })
        
        # Sort by timestamp
        audit_df = pd.DataFrame(audit_records)
        if not audit_df.empty:
            audit_df = audit_df.sort_values('Timestamp', ascending=False)
        
        return audit_df
    
    def _log_export_activity(self, export_type: str, file_format: str, 
                           start_date, end_date, filters: Dict, 
                           record_count: int, file_size: int) -> None:
        """Log data export activity"""
        try:
            export_log = DataExportLog(
                export_type=export_type,
                file_format=file_format,
                date_range_start=start_date,
                date_range_end=end_date,
                filters_applied=filters,
                record_count=record_count,
                file_size_kb=file_size // 1024
            )
            
            db.session.add(export_log)
            db.session.commit()
            
            self.log_bot_activity(
                'DATA_EXPORT',
                f'Exported {record_count} {export_type} records as {file_format}',
                details={
                    'export_type': export_type,
                    'file_format': file_format,
                    'record_count': record_count,
                    'file_size_kb': file_size // 1024
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to log export activity: {e}")