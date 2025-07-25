"""
Advanced Trading Strategies Module
Implements sophisticated trading algorithms with multiple timeframe analysis,
volume-based signals, support/resistance detection, and advanced indicators.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class TrailingStopLoss:
    """Trailing stop-loss that follows price movements automatically"""
    
    def __init__(self, initial_stop_pct: float = 0.05, trail_pct: float = 0.02):
        self.initial_stop_pct = initial_stop_pct  # Initial 5% stop loss
        self.trail_pct = trail_pct  # Trail by 2%
        self.stops = {}  # symbol -> {entry_price, stop_price, highest_price, position_type}
    
    def set_stop(self, symbol: str, entry_price: float, position_type: str):
        """Set initial trailing stop for a position"""
        if position_type == 'LONG':
            stop_price = entry_price * (1 - self.initial_stop_pct)
            self.stops[symbol] = {
                'entry_price': entry_price,
                'stop_price': stop_price,
                'highest_price': entry_price,
                'position_type': position_type
            }
        elif position_type == 'SHORT':
            stop_price = entry_price * (1 + self.initial_stop_pct)
            self.stops[symbol] = {
                'entry_price': entry_price,
                'stop_price': stop_price,
                'lowest_price': entry_price,
                'position_type': position_type
            }
    
    def update_stop(self, symbol: str, current_price: float) -> bool:
        """Update trailing stop and return True if stop is triggered"""
        if symbol not in self.stops:
            return False
            
        stop_data = self.stops[symbol]
        position_type = stop_data['position_type']
        
        if position_type == 'LONG':
            # For long positions, trail stop upward
            if current_price > stop_data['highest_price']:
                stop_data['highest_price'] = current_price
                new_stop = current_price * (1 - self.trail_pct)
                if new_stop > stop_data['stop_price']:
                    stop_data['stop_price'] = new_stop
            
            # Check if stop is triggered
            return current_price <= stop_data['stop_price']
            
        elif position_type == 'SHORT':
            # For short positions, trail stop downward
            if current_price < stop_data['lowest_price']:
                stop_data['lowest_price'] = current_price
                new_stop = current_price * (1 + self.trail_pct)
                if new_stop < stop_data['stop_price']:
                    stop_data['stop_price'] = new_stop
            
            # Check if stop is triggered
            return current_price >= stop_data['stop_price']
        
        return False
    
    def remove_stop(self, symbol: str):
        """Remove trailing stop for a symbol"""
        if symbol in self.stops:
            del self.stops[symbol]
    
    def get_stop_info(self, symbol: str) -> Optional[Dict]:
        """Get current stop information"""
        return self.stops.get(symbol)

class MultiTimeframeAnalysis:
    """Analyzes multiple timeframes for comprehensive market view"""
    
    def __init__(self):
        self.timeframes = {
            '1h': 24,    # 24 hours of hourly data
            '4h': 24,    # 24 * 4 hours of 4-hour data  
            '1d': 30     # 30 days of daily data
        }
    
    def analyze_timeframes(self, historical_data: List[Dict]) -> Dict:
        """Analyze trends across multiple timeframes"""
        if len(historical_data) < 30:
            return {'trend': 'NEUTRAL', 'strength': 0, 'confluence': False}
        
        # Convert to pandas for easier analysis
        df = pd.DataFrame(historical_data)
        df['price'] = df['close'].astype(float)
        
        timeframe_signals = {}
        
        # 1-hour trend (last 24 data points)
        if len(df) >= 24:
            hourly_data = df.tail(24)
            timeframe_signals['1h'] = self._analyze_trend(hourly_data['price'].tolist())
        
        # 4-hour trend (every 4th point for last 96 points)
        if len(df) >= 96:
            four_hour_data = df.iloc[::4].tail(24)
            timeframe_signals['4h'] = self._analyze_trend(four_hour_data['price'].tolist())
        
        # Daily trend (full dataset)
        timeframe_signals['1d'] = self._analyze_trend(df['price'].tolist())
        
        # Determine confluence
        bullish_count = sum(1 for signal in timeframe_signals.values() if signal['trend'] == 'BULLISH')
        bearish_count = sum(1 for signal in timeframe_signals.values() if signal['trend'] == 'BEARISH')
        
        if bullish_count >= 2:
            overall_trend = 'BULLISH'
            confluence = True
        elif bearish_count >= 2:
            overall_trend = 'BEARISH'
            confluence = True
        else:
            overall_trend = 'NEUTRAL'
            confluence = False
        
        # Calculate average strength
        avg_strength = np.mean([signal['strength'] for signal in timeframe_signals.values()])
        
        return {
            'trend': overall_trend,
            'strength': avg_strength,
            'confluence': confluence,
            'timeframes': timeframe_signals
        }
    
    def _analyze_trend(self, prices: List[float]) -> Dict:
        """Analyze trend for a single timeframe"""
        if len(prices) < 10:
            return {'trend': 'NEUTRAL', 'strength': 0}
        
        # Calculate moving averages
        short_ma = np.mean(prices[-10:])
        long_ma = np.mean(prices[-20:]) if len(prices) >= 20 else np.mean(prices)
        
        # Determine trend
        if short_ma > long_ma * 1.01:  # 1% threshold
            trend = 'BULLISH'
            strength = min((short_ma / long_ma - 1) * 100, 100)
        elif short_ma < long_ma * 0.99:  # 1% threshold
            trend = 'BEARISH'
            strength = min((1 - short_ma / long_ma) * 100, 100)
        else:
            trend = 'NEUTRAL'
            strength = 0
        
        return {'trend': trend, 'strength': strength}

class VolumeAnalysis:
    """Volume-based trading signal analysis"""
    
    def __init__(self):
        self.volume_ma_period = 20
    
    def analyze_volume_signals(self, historical_data: List[Dict]) -> Dict:
        """Analyze volume patterns for trading signals"""
        if len(historical_data) < self.volume_ma_period:
            return {'signal': 'NEUTRAL', 'strength': 0, 'volume_trend': 'NEUTRAL'}
        
        # Extract price and volume data
        volumes = [float(d.get('volume', 0)) for d in historical_data]
        prices = [float(d['close']) for d in historical_data]
        
        if not any(volumes):  # No volume data available
            return {'signal': 'NEUTRAL', 'strength': 0, 'volume_trend': 'NEUTRAL'}
        
        # Calculate volume moving average
        volume_ma = np.mean(volumes[-self.volume_ma_period:])
        current_volume = volumes[-1]
        
        # Volume surge detection (current volume > 150% of average)
        volume_surge = current_volume > volume_ma * 1.5
        
        # Price-volume relationship
        price_change = (prices[-1] - prices[-2]) / prices[-2] * 100
        volume_change = (volumes[-1] - volumes[-2]) / volumes[-2] * 100 if volumes[-2] > 0 else 0
        
        # Determine signal based on price-volume divergence
        signal = 'NEUTRAL'
        strength = 0
        
        if volume_surge:
            if price_change > 0.5:  # Price up with high volume
                signal = 'BUY'
                strength = min(abs(price_change) * 2, 100)
            elif price_change < -0.5:  # Price down with high volume
                signal = 'SELL'
                strength = min(abs(price_change) * 2, 100)
        
        # Volume trend analysis
        recent_volumes = volumes[-5:]
        older_volumes = volumes[-10:-5] if len(volumes) >= 10 else volumes[:-5]
        
        if len(older_volumes) > 0:
            recent_avg = np.mean(recent_volumes)
            older_avg = np.mean(older_volumes)
            
            if recent_avg > older_avg * 1.2:
                volume_trend = 'INCREASING'
            elif recent_avg < older_avg * 0.8:
                volume_trend = 'DECREASING'
            else:
                volume_trend = 'STABLE'
        else:
            volume_trend = 'NEUTRAL'
        
        return {
            'signal': signal,
            'strength': strength,
            'volume_trend': volume_trend,
            'volume_surge': volume_surge,
            'current_volume': current_volume,
            'volume_ma': volume_ma
        }

class SupportResistanceDetector:
    """Detects support and resistance levels using pivot points"""
    
    def __init__(self, lookback_period: int = 20, min_touches: int = 2):
        self.lookback_period = lookback_period
        self.min_touches = min_touches
    
    def detect_levels(self, historical_data: List[Dict]) -> Dict:
        """Detect support and resistance levels"""
        if len(historical_data) < self.lookback_period:
            return {'support_levels': [], 'resistance_levels': [], 'current_level': None}
        
        # Extract price data
        highs = [float(d['high']) for d in historical_data]
        lows = [float(d['low']) for d in historical_data]
        closes = [float(d['close']) for d in historical_data]
        
        current_price = closes[-1]
        
        # Find pivot highs (resistance levels)
        resistance_levels = self._find_pivot_points(highs, 'high')
        
        # Find pivot lows (support levels)
        support_levels = self._find_pivot_points(lows, 'low')
        
        # Determine current level context
        current_level = self._determine_current_level(current_price, support_levels, resistance_levels)
        
        return {
            'support_levels': sorted(support_levels),
            'resistance_levels': sorted(resistance_levels, reverse=True),
            'current_level': current_level,
            'nearest_support': max([s for s in support_levels if s < current_price], default=None),
            'nearest_resistance': min([r for r in resistance_levels if r > current_price], default=None)
        }
    
    def _find_pivot_points(self, prices: List[float], point_type: str) -> List[float]:
        """Find pivot points in price data"""
        pivot_points = []
        window = 3  # Look 3 periods left and right
        
        for i in range(window, len(prices) - window):
            current_price = prices[i]
            left_prices = prices[i-window:i]
            right_prices = prices[i+1:i+window+1]
            
            if point_type == 'high':
                # Pivot high: current price higher than surrounding prices
                if all(current_price >= p for p in left_prices + right_prices):
                    pivot_points.append(current_price)
            else:
                # Pivot low: current price lower than surrounding prices
                if all(current_price <= p for p in left_prices + right_prices):
                    pivot_points.append(current_price)
        
        # Remove duplicates and similar levels (within 1% of each other)
        filtered_points = []
        for point in sorted(set(pivot_points)):
            if not any(abs(point - existing) / existing < 0.01 for existing in filtered_points):
                filtered_points.append(point)
        
        return filtered_points
    
    def _determine_current_level(self, current_price: float, support_levels: List[float], 
                                resistance_levels: List[float]) -> Optional[str]:
        """Determine if current price is near a significant level"""
        tolerance = 0.005  # 0.5% tolerance
        
        # Check if near support
        for support in support_levels:
            if abs(current_price - support) / support < tolerance:
                return 'AT_SUPPORT'
        
        # Check if near resistance
        for resistance in resistance_levels:
            if abs(current_price - resistance) / resistance < tolerance:
                return 'AT_RESISTANCE'
        
        return None

class MACDStrategy:
    """MACD (Moving Average Convergence Divergence) strategy"""
    
    def __init__(self, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9):
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
    
    def calculate_macd(self, prices: List[float]) -> Dict:
        """Calculate MACD indicator"""
        if len(prices) < self.slow_period + self.signal_period:
            return {'macd_line': None, 'signal_line': None, 'histogram': None, 'signal': 'NEUTRAL'}
        
        # Calculate EMAs
        fast_ema = self._calculate_ema(prices, self.fast_period)
        slow_ema = self._calculate_ema(prices, self.slow_period)
        
        # MACD line = Fast EMA - Slow EMA
        macd_line = fast_ema - slow_ema
        
        # Calculate signal line (EMA of MACD line)
        macd_history = []
        for i in range(len(prices) - self.slow_period + 1):
            if i >= len(prices) - len(prices) + self.slow_period - 1:
                fast = self._calculate_ema(prices[:self.slow_period + i], self.fast_period)
                slow = self._calculate_ema(prices[:self.slow_period + i], self.slow_period)
                macd_history.append(fast - slow)
        
        if len(macd_history) >= self.signal_period:
            signal_line = self._calculate_ema(macd_history, self.signal_period)
        else:
            signal_line = 0
        
        # Histogram = MACD - Signal
        histogram = macd_line - signal_line
        
        # Determine signal
        signal = self._determine_macd_signal(macd_line, signal_line, histogram, macd_history)
        
        return {
            'macd_line': macd_line,
            'signal_line': signal_line,
            'histogram': histogram,
            'signal': signal
        }
    
    def _calculate_ema(self, prices: List[float], period: int) -> float:
        """Calculate Exponential Moving Average"""
        if len(prices) < period:
            return np.mean(prices)
        
        multiplier = 2 / (period + 1)
        ema = np.mean(prices[:period])  # Start with SMA
        
        for price in prices[period:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))
        
        return ema
    
    def _determine_macd_signal(self, macd_line: float, signal_line: float, 
                              histogram: float, macd_history: List[float]) -> str:
        """Determine MACD trading signal"""
        # Bullish crossover: MACD crosses above signal line
        if len(macd_history) >= 2:
            prev_macd = macd_history[-2] if len(macd_history) >= 2 else macd_line
            prev_signal = signal_line  # Simplified for this example
            
            if macd_line > signal_line and prev_macd <= prev_signal:
                return 'BUY'
            elif macd_line < signal_line and prev_macd >= prev_signal:
                return 'SELL'
        
        return 'NEUTRAL'

class StochasticRSIStrategy:
    """Stochastic RSI strategy for overbought/oversold conditions"""
    
    def __init__(self, rsi_period: int = 14, stoch_period: int = 14):
        self.rsi_period = rsi_period
        self.stoch_period = stoch_period
    
    def calculate_stochastic_rsi(self, prices: List[float]) -> Dict:
        """Calculate Stochastic RSI indicator"""
        if len(prices) < self.rsi_period + self.stoch_period:
            return {'stoch_rsi': None, 'signal': 'NEUTRAL', 'condition': 'NORMAL'}
        
        # First calculate RSI
        rsi_values = []
        for i in range(self.rsi_period, len(prices)):
            period_prices = prices[i - self.rsi_period:i]
            rsi = self._calculate_rsi(period_prices)
            rsi_values.append(rsi)
        
        if len(rsi_values) < self.stoch_period:
            return {'stoch_rsi': None, 'signal': 'NEUTRAL', 'condition': 'NORMAL'}
        
        # Calculate Stochastic of RSI
        recent_rsi = rsi_values[-self.stoch_period:]
        highest_rsi = max(recent_rsi)
        lowest_rsi = min(recent_rsi)
        current_rsi = rsi_values[-1]
        
        if highest_rsi == lowest_rsi:
            stoch_rsi = 50  # Neutral when no variation
        else:
            stoch_rsi = ((current_rsi - lowest_rsi) / (highest_rsi - lowest_rsi)) * 100
        
        # Determine signal and condition
        signal, condition = self._determine_stoch_rsi_signal(stoch_rsi)
        
        return {
            'stoch_rsi': stoch_rsi,
            'signal': signal,
            'condition': condition,
            'current_rsi': current_rsi
        }
    
    def _calculate_rsi(self, prices: List[float]) -> float:
        """Calculate RSI for a period"""
        if len(prices) < 2:
            return 50
        
        gains = []
        losses = []
        
        for i in range(1, len(prices)):
            change = prices[i] - prices[i-1]
            if change > 0:
                gains.append(change)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(abs(change))
        
        avg_gain = np.mean(gains) if gains else 0
        avg_loss = np.mean(losses) if losses else 0
        
        if avg_loss == 0:
            return 100
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _determine_stoch_rsi_signal(self, stoch_rsi: float) -> Tuple[str, str]:
        """Determine Stochastic RSI signal and market condition"""
        if stoch_rsi >= 80:
            return 'SELL', 'OVERBOUGHT'
        elif stoch_rsi <= 20:
            return 'BUY', 'OVERSOLD'
        elif stoch_rsi >= 70:
            return 'NEUTRAL', 'APPROACHING_OVERBOUGHT'
        elif stoch_rsi <= 30:
            return 'NEUTRAL', 'APPROACHING_OVERSOLD'
        else:
            return 'NEUTRAL', 'NORMAL'

class AdvancedTradingEngine:
    """Main engine that coordinates all advanced trading strategies"""
    
    def __init__(self):
        self.trailing_stops = TrailingStopLoss()
        self.timeframe_analyzer = MultiTimeframeAnalysis()
        self.volume_analyzer = VolumeAnalysis()
        self.support_resistance = SupportResistanceDetector()
        self.macd_strategy = MACDStrategy()
        self.stoch_rsi_strategy = StochasticRSIStrategy()
    
    def analyze_comprehensive_signals(self, symbol: str, historical_data: List[Dict]) -> Dict:
        """Generate comprehensive trading signals using all advanced strategies"""
        if len(historical_data) < 30:
            return self._neutral_response()
        
        # Extract price data
        prices = [float(d['close']) for d in historical_data]
        
        # Run all analyses
        timeframe_analysis = self.timeframe_analyzer.analyze_timeframes(historical_data)
        volume_analysis = self.volume_analyzer.analyze_volume_signals(historical_data)
        sr_analysis = self.support_resistance.detect_levels(historical_data)
        macd_analysis = self.macd_strategy.calculate_macd(prices)
        stoch_rsi_analysis = self.stoch_rsi_strategy.calculate_stochastic_rsi(prices)
        
        # Weight and combine signals
        signals = []
        
        # Timeframe confluence (high weight)
        if timeframe_analysis['confluence']:
            if timeframe_analysis['trend'] == 'BULLISH':
                signals.append(('BUY', timeframe_analysis['strength'], 0.3))
            elif timeframe_analysis['trend'] == 'BEARISH':
                signals.append(('SELL', timeframe_analysis['strength'], 0.3))
        
        # Volume signals (medium weight)
        if volume_analysis['signal'] != 'NEUTRAL':
            signals.append((volume_analysis['signal'], volume_analysis['strength'], 0.2))
        
        # MACD signals (medium weight)
        if macd_analysis['signal'] != 'NEUTRAL':
            signals.append((macd_analysis['signal'], 75, 0.2))
        
        # Stochastic RSI signals (lower weight, contrarian)
        if stoch_rsi_analysis['signal'] != 'NEUTRAL':
            signals.append((stoch_rsi_analysis['signal'], 60, 0.15))
        
        # Support/Resistance context (modifier)
        sr_modifier = self._get_sr_modifier(sr_analysis, prices[-1])
        
        # Calculate weighted signal
        final_signal = self._calculate_weighted_signal(signals, sr_modifier)
        
        return {
            'signal': final_signal['action'],
            'confidence': final_signal['confidence'],
            'reasoning': final_signal['reasoning'],
            'components': {
                'timeframe': timeframe_analysis,
                'volume': volume_analysis,
                'support_resistance': sr_analysis,
                'macd': macd_analysis,
                'stochastic_rsi': stoch_rsi_analysis
            }
        }
    
    def _neutral_response(self) -> Dict:
        """Return neutral response when insufficient data"""
        return {
            'signal': 'NEUTRAL',
            'confidence': 0,
            'reasoning': 'Insufficient data for advanced analysis',
            'components': {}
        }
    
    def _get_sr_modifier(self, sr_analysis: Dict, current_price: float) -> Dict:
        """Get support/resistance context modifier"""
        nearest_support = sr_analysis.get('nearest_support')
        nearest_resistance = sr_analysis.get('nearest_resistance')
        current_level = sr_analysis.get('current_level')
        
        modifier = {'buy_boost': 0, 'sell_boost': 0, 'context': 'NEUTRAL'}
        
        if current_level == 'AT_SUPPORT':
            modifier['buy_boost'] = 0.1
            modifier['context'] = 'AT_SUPPORT'
        elif current_level == 'AT_RESISTANCE':
            modifier['sell_boost'] = 0.1
            modifier['context'] = 'AT_RESISTANCE'
        elif nearest_support and current_price / nearest_support < 1.02:  # Within 2% of support
            modifier['buy_boost'] = 0.05
            modifier['context'] = 'NEAR_SUPPORT'
        elif nearest_resistance and nearest_resistance / current_price < 1.02:  # Within 2% of resistance
            modifier['sell_boost'] = 0.05
            modifier['context'] = 'NEAR_RESISTANCE'
        
        return modifier
    
    def _calculate_weighted_signal(self, signals: List[Tuple], sr_modifier: Dict) -> Dict:
        """Calculate final weighted trading signal"""
        if not signals:
            return {'action': 'NEUTRAL', 'confidence': 0, 'reasoning': 'No clear signals detected'}
        
        buy_score = 0
        sell_score = 0
        total_weight = 0
        
        reasoning_parts = []
        
        for signal, strength, weight in signals:
            total_weight += weight
            weighted_strength = (strength / 100) * weight
            
            if signal == 'BUY':
                buy_score += weighted_strength
                reasoning_parts.append(f"Bullish signal (strength: {strength:.1f}%)")
            elif signal == 'SELL':
                sell_score += weighted_strength
                reasoning_parts.append(f"Bearish signal (strength: {strength:.1f}%)")
        
        # Apply support/resistance modifiers
        buy_score += sr_modifier['buy_boost']
        sell_score += sr_modifier['sell_boost']
        
        if sr_modifier['context'] != 'NEUTRAL':
            reasoning_parts.append(f"Price {sr_modifier['context']}")
        
        # Determine final signal
        if buy_score > sell_score + 0.1:  # Require significant edge
            action = 'BUY'
            confidence = min(buy_score * 100, 100)
        elif sell_score > buy_score + 0.1:
            action = 'SELL'
            confidence = min(sell_score * 100, 100)
        else:
            action = 'NEUTRAL'
            confidence = 0
            reasoning_parts = ['Conflicting signals - staying neutral']
        
        return {
            'action': action,
            'confidence': confidence,
            'reasoning': ' | '.join(reasoning_parts)
        }