"""
Advanced Analytics Engine for Bitcoin Trading Bot
Provides comprehensive performance metrics, risk analysis, and strategy comparison
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any
from sqlalchemy import func
from app import db
from models import Trade, Portfolio, Account, athens_now, ATHENS_TZ
import pytz
import logging

logger = logging.getLogger(__name__)

class TradingAnalytics:
    """Advanced analytics engine for trading performance analysis"""
    
    def __init__(self):
        self.risk_free_rate = 0.02  # 2% annual risk-free rate (ECB rates)
    
    def get_strategy_performance_comparison(self) -> Dict[str, Any]:
        """Compare performance across different trading strategies"""
        try:
            # Get all completed trades grouped by strategy
            strategy_stats = {}
            
            strategies = db.session.query(Trade.strategy).distinct().all()
            
            for (strategy_name,) in strategies:
                if not strategy_name:
                    continue
                    
                trades = db.session.query(Trade).filter_by(strategy=strategy_name).all()
                
                if not trades:
                    continue
                
                # Calculate strategy metrics
                total_trades = len(trades)
                profitable_trades = len([t for t in trades if t.pnl and t.pnl > 0])
                total_pnl = sum([t.pnl for t in trades if t.pnl]) or 0
                
                win_rate = (profitable_trades / total_trades * 100) if total_trades > 0 else 0
                avg_pnl = total_pnl / total_trades if total_trades > 0 else 0
                
                # Calculate average trade duration
                durations = [t.duration for t in trades if t.duration]
                avg_duration = sum(durations) / len(durations) if durations else 0
                
                # Risk metrics
                pnl_values = [t.pnl for t in trades if t.pnl is not None]
                if pnl_values:
                    volatility = np.std(pnl_values)
                    max_loss = min(pnl_values)
                    max_gain = max(pnl_values)
                else:
                    volatility = max_loss = max_gain = 0
                
                strategy_stats[strategy_name] = {
                    'total_trades': total_trades,
                    'win_rate': round(win_rate, 2),
                    'total_pnl': round(total_pnl, 2),
                    'avg_pnl': round(avg_pnl, 2),
                    'avg_duration_minutes': round(avg_duration / 60, 1) if avg_duration else 0,
                    'volatility': round(volatility, 2),
                    'max_loss': round(max_loss, 2),
                    'max_gain': round(max_gain, 2),
                    'profit_factor': round(max_gain / abs(max_loss), 2) if max_loss < 0 else 0
                }
            
            return strategy_stats
            
        except Exception as e:
            logger.error(f"Error calculating strategy performance: {e}")
            return {}
    
    def get_pnl_breakdown(self, period: str = 'daily') -> Dict[str, Any]:
        """Get P&L breakdown by time periods (daily/weekly/monthly)"""
        try:
            # Get date range for the last period
            end_date = athens_now()
            
            if period == 'daily':
                start_date = end_date - timedelta(days=30)
                date_format = '%Y-%m-%d'
                group_format = 'DATE(timestamp)'
            elif period == 'weekly':
                start_date = end_date - timedelta(weeks=12)
                date_format = '%Y-W%U'
                group_format = 'YEARWEEK(timestamp)'
            else:  # monthly
                start_date = end_date - timedelta(days=365)
                date_format = '%Y-%m'
                group_format = 'DATE_FORMAT(timestamp, "%Y-%m")'
            
            # Query trades within date range
            trades = db.session.query(Trade).filter(
                Trade.timestamp >= start_date,
                Trade.pnl.isnot(None)
            ).order_by(Trade.timestamp).all()
            
            # Group by time period
            period_data = {}
            cumulative_pnl = 0
            
            for trade in trades:
                # Convert to Athens time for grouping
                athens_time = trade.timestamp
                if hasattr(athens_time, 'tzinfo') and athens_time.tzinfo is None:
                    athens_time = pytz.utc.localize(athens_time).astimezone(ATHENS_TZ)
                
                if period == 'daily':
                    period_key = athens_time.strftime('%Y-%m-%d')
                elif period == 'weekly':
                    period_key = f"{athens_time.year}-W{athens_time.isocalendar()[1]:02d}"
                else:  # monthly
                    period_key = athens_time.strftime('%Y-%m')
                
                if period_key not in period_data:
                    period_data[period_key] = {
                        'pnl': 0,
                        'trades': 0,
                        'wins': 0,
                        'losses': 0
                    }
                
                period_data[period_key]['pnl'] += trade.pnl
                period_data[period_key]['trades'] += 1
                
                if trade.pnl > 0:
                    period_data[period_key]['wins'] += 1
                elif trade.pnl < 0:
                    period_data[period_key]['losses'] += 1
            
            # Calculate cumulative P&L and additional metrics
            sorted_periods = sorted(period_data.keys())
            for period_key in sorted_periods:
                cumulative_pnl += period_data[period_key]['pnl']
                period_data[period_key]['cumulative_pnl'] = round(cumulative_pnl, 2)
                period_data[period_key]['pnl'] = round(period_data[period_key]['pnl'], 2)
                period_data[period_key]['win_rate'] = round(
                    (period_data[period_key]['wins'] / period_data[period_key]['trades'] * 100) 
                    if period_data[period_key]['trades'] > 0 else 0, 2
                )
            
            return {
                'period': period,
                'data': period_data,
                'summary': {
                    'total_pnl': round(cumulative_pnl, 2),
                    'total_periods': len(period_data),
                    'profitable_periods': len([p for p in period_data.values() if p['pnl'] > 0]),
                    'avg_period_pnl': round(cumulative_pnl / len(period_data), 2) if period_data else 0
                }
            }
            
        except Exception as e:
            logger.error(f"Error calculating P&L breakdown: {e}")
            return {'period': period, 'data': {}, 'summary': {}}
    
    def calculate_maximum_drawdown(self) -> Dict[str, float]:
        """Calculate maximum drawdown from peak to trough"""
        try:
            # Get all trades ordered by time
            trades = db.session.query(Trade).filter(
                Trade.pnl.isnot(None)
            ).order_by(Trade.timestamp).all()
            
            if not trades:
                return {'max_drawdown': 0, 'max_drawdown_percent': 0, 'peak_value': 1000, 'trough_value': 1000}
            
            # Calculate cumulative portfolio value
            starting_value = 1000.0  # Starting balance
            portfolio_values = [starting_value]
            cumulative_pnl = 0
            
            for trade in trades:
                cumulative_pnl += trade.pnl
                portfolio_values.append(starting_value + cumulative_pnl)
            
            # Calculate drawdown
            peak = portfolio_values[0]
            max_drawdown = 0
            max_drawdown_percent = 0
            peak_value = starting_value
            trough_value = starting_value
            
            for value in portfolio_values:
                if value > peak:
                    peak = value
                
                drawdown = peak - value
                drawdown_percent = (drawdown / peak * 100) if peak > 0 else 0
                
                if drawdown > max_drawdown:
                    max_drawdown = drawdown
                    max_drawdown_percent = drawdown_percent
                    peak_value = peak
                    trough_value = value
            
            return {
                'max_drawdown': round(max_drawdown, 2),
                'max_drawdown_percent': round(max_drawdown_percent, 2),
                'peak_value': round(peak_value, 2),
                'trough_value': round(trough_value, 2)
            }
            
        except Exception as e:
            logger.error(f"Error calculating maximum drawdown: {e}")
            return {'max_drawdown': 0, 'max_drawdown_percent': 0, 'peak_value': 1000, 'trough_value': 1000}
    
    def calculate_sharpe_ratio(self, period_days: int = 30) -> float:
        """Calculate Sharpe ratio for risk-adjusted returns"""
        try:
            # Get trades from the specified period
            end_date = athens_now()
            start_date = end_date - timedelta(days=period_days)
            
            trades = db.session.query(Trade).filter(
                Trade.timestamp >= start_date,
                Trade.pnl.isnot(None)
            ).all()
            
            if not trades:
                return 0.0
            
            # Calculate daily returns
            daily_returns = []
            pnl_values = [trade.pnl for trade in trades]
            
            # Convert to daily returns (assuming 1000 EUR base)
            base_value = 1000.0
            for pnl in pnl_values:
                daily_return = pnl / base_value
                daily_returns.append(daily_return)
            
            if len(daily_returns) < 2:
                return 0.0
            
            # Calculate metrics
            avg_return = np.mean(daily_returns)
            return_std = np.std(daily_returns)
            
            # Daily risk-free rate (assuming 2% annual)
            daily_risk_free = self.risk_free_rate / 365
            
            # Sharpe ratio
            if return_std == 0:
                return 0.0
            
            sharpe_ratio = (avg_return - daily_risk_free) / return_std
            
            # Annualize (multiply by sqrt of trading days per year)
            annualized_sharpe = sharpe_ratio * np.sqrt(365)
            
            return round(annualized_sharpe, 3)
            
        except Exception as e:
            logger.error(f"Error calculating Sharpe ratio: {e}")
            return 0.0
    
    def get_time_based_performance(self) -> Dict[str, Any]:
        """Analyze performance by hours and days of week"""
        try:
            # Get all trades with P&L
            trades = db.session.query(Trade).filter(
                Trade.pnl.isnot(None)
            ).all()
            
            if not trades:
                return {'hourly': {}, 'daily': {}}
            
            # Initialize hour and day tracking
            hourly_performance = {hour: {'pnl': 0, 'trades': 0, 'wins': 0} for hour in range(24)}
            daily_performance = {day: {'pnl': 0, 'trades': 0, 'wins': 0} for day in range(7)}
            
            day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            
            for trade in trades:
                # Convert to Athens time
                athens_time = trade.timestamp
                if hasattr(athens_time, 'tzinfo') and athens_time.tzinfo is None:
                    athens_time = pytz.utc.localize(athens_time).astimezone(ATHENS_TZ)
                
                hour = athens_time.hour
                day = athens_time.weekday()  # 0 = Monday, 6 = Sunday
                
                # Update hourly stats
                hourly_performance[hour]['pnl'] += trade.pnl
                hourly_performance[hour]['trades'] += 1
                if trade.pnl > 0:
                    hourly_performance[hour]['wins'] += 1
                
                # Update daily stats
                daily_performance[day]['pnl'] += trade.pnl
                daily_performance[day]['trades'] += 1
                if trade.pnl > 0:
                    daily_performance[day]['wins'] += 1
            
            # Calculate win rates and format data
            formatted_hourly = {}
            for hour, data in hourly_performance.items():
                if data['trades'] > 0:
                    formatted_hourly[f"{hour:02d}:00"] = {
                        'pnl': round(data['pnl'], 2),
                        'trades': data['trades'],
                        'win_rate': round((data['wins'] / data['trades'] * 100), 1),
                        'avg_pnl': round(data['pnl'] / data['trades'], 2)
                    }
            
            formatted_daily = {}
            for day, data in daily_performance.items():
                if data['trades'] > 0:
                    formatted_daily[day_names[day]] = {
                        'pnl': round(data['pnl'], 2),
                        'trades': data['trades'],
                        'win_rate': round((data['wins'] / data['trades'] * 100), 1),
                        'avg_pnl': round(data['pnl'] / data['trades'], 2)
                    }
            
            # Find best and worst performing times
            best_hour = max(formatted_hourly.items(), key=lambda x: x[1]['pnl']) if formatted_hourly else None
            worst_hour = min(formatted_hourly.items(), key=lambda x: x[1]['pnl']) if formatted_hourly else None
            best_day = max(formatted_daily.items(), key=lambda x: x[1]['pnl']) if formatted_daily else None
            worst_day = min(formatted_daily.items(), key=lambda x: x[1]['pnl']) if formatted_daily else None
            
            return {
                'hourly': formatted_hourly,
                'daily': formatted_daily,
                'insights': {
                    'best_hour': {'time': best_hour[0], 'pnl': best_hour[1]['pnl']} if best_hour else None,
                    'worst_hour': {'time': worst_hour[0], 'pnl': worst_hour[1]['pnl']} if worst_hour else None,
                    'best_day': {'day': best_day[0], 'pnl': best_day[1]['pnl']} if best_day else None,
                    'worst_day': {'day': worst_day[0], 'pnl': worst_day[1]['pnl']} if worst_day else None
                }
            }
            
        except Exception as e:
            logger.error(f"Error calculating time-based performance: {e}")
            return {'hourly': {}, 'daily': {}, 'insights': {}}
    
    def get_risk_metrics(self) -> Dict[str, Any]:
        """Calculate comprehensive risk metrics"""
        try:
            trades = db.session.query(Trade).filter(Trade.pnl.isnot(None)).all()
            
            if not trades:
                return {}
            
            pnl_values = [trade.pnl for trade in trades]
            
            # Basic risk metrics
            total_trades = len(pnl_values)
            profitable_trades = len([pnl for pnl in pnl_values if pnl > 0])
            losing_trades = len([pnl for pnl in pnl_values if pnl < 0])
            
            avg_win = np.mean([pnl for pnl in pnl_values if pnl > 0]) if profitable_trades > 0 else 0
            avg_loss = np.mean([pnl for pnl in pnl_values if pnl < 0]) if losing_trades > 0 else 0
            
            # Risk ratios
            profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else 0
            win_loss_ratio = profitable_trades / losing_trades if losing_trades > 0 else float('inf')
            
            # Volatility metrics
            returns_std = np.std(pnl_values)
            max_consecutive_losses = self._calculate_max_consecutive_losses(pnl_values)
            
            # Value at Risk (95% confidence)
            var_95 = np.percentile(pnl_values, 5) if pnl_values else 0
            
            return {
                'total_trades': total_trades,
                'win_rate': round((profitable_trades / total_trades * 100), 2),
                'avg_win': round(avg_win, 2),
                'avg_loss': round(avg_loss, 2),
                'profit_factor': round(profit_factor, 2),
                'win_loss_ratio': round(win_loss_ratio, 2) if win_loss_ratio != float('inf') else 'N/A',
                'volatility': round(returns_std, 2),
                'max_consecutive_losses': max_consecutive_losses,
                'var_95': round(var_95, 2),
                'sharpe_ratio': self.calculate_sharpe_ratio()
            }
            
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {e}")
            return {}
    
    def _calculate_max_consecutive_losses(self, pnl_values: List[float]) -> int:
        """Calculate maximum consecutive losses"""
        max_consecutive = 0
        current_consecutive = 0
        
        for pnl in pnl_values:
            if pnl < 0:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        
        return max_consecutive
    
    def get_comprehensive_analytics(self) -> Dict[str, Any]:
        """Get all analytics in one comprehensive report"""
        try:
            return {
                'strategy_comparison': self.get_strategy_performance_comparison(),
                'pnl_breakdown': {
                    'daily': self.get_pnl_breakdown('daily'),
                    'weekly': self.get_pnl_breakdown('weekly'),
                    'monthly': self.get_pnl_breakdown('monthly')
                },
                'drawdown': self.calculate_maximum_drawdown(),
                'sharpe_ratio': self.calculate_sharpe_ratio(),
                'time_performance': self.get_time_based_performance(),
                'risk_metrics': self.get_risk_metrics(),
                'generated_at': athens_now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error generating comprehensive analytics: {e}")
            return {}