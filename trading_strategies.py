import numpy as np
import pandas as pd
from typing import Dict, List, Any
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class TradingStrategy:
    def __init__(self, name: str, symbol: str, parameters: Dict[str, Any]):
        self.name = name
        self.symbol = symbol
        self.parameters = parameters
        
    def should_buy(self, price_data: List[Dict]) -> bool:
        """Override in subclasses"""
        raise NotImplementedError
        
    def should_sell(self, price_data: List[Dict], current_position: float) -> bool:
        """Override in subclasses"""
        raise NotImplementedError
    
    def should_short(self, price_data: List[Dict]) -> bool:
        """Override in subclasses for short selling signals"""
        raise NotImplementedError
    
    def should_cover(self, price_data: List[Dict], current_position: float) -> bool:
        """Override in subclasses for covering short positions"""
        raise NotImplementedError
        
    def get_position_size(self, account_value: float, current_price: float) -> float:
        """Calculate position size based on 10% of account balance for Bitcoin trading"""
        # Use 10% of total account balance for each trade
        position_value = account_value * 0.10  # €100 with €1,000 account
        position_size = position_value / current_price
        
        # Ensure minimum position size for Bitcoin (but prioritize 10% allocation)
        return max(position_size, 0.001)  # Should be ~0.001 BTC with current prices and €1K account

    def get_signal_details(self, price_data: List[Dict], signal_type: str) -> Dict:
        """Get detailed signal information - override in subclasses"""
        return {
            'signal_type': signal_type,
            'strategy': self.name,
            'timestamp': datetime.now().isoformat(),
            'current_price': price_data[-1]['close'] if price_data else None
        }

    def get_reasoning(self, price_data: List[Dict], signal_type: str) -> str:
        """Get human-readable reasoning - override in subclasses"""
        return f"{self.name} strategy generated {signal_type} signal"

    def get_market_conditions(self, price_data: List[Dict]) -> Dict:
        """Get market conditions at time of signal"""
        if not price_data:
            return {}
        
        current_price = float(price_data[-1]['close'])
        prev_price = float(price_data[-2]['close']) if len(price_data) > 1 else current_price
        
        return {
            'current_price': current_price,
            'previous_price': prev_price,
            'price_change': current_price - prev_price,
            'price_change_percent': ((current_price - prev_price) / prev_price * 100) if prev_price else 0,
            'volume': price_data[-1].get('volume', 0),
            'timestamp': price_data[-1].get('timestamp', datetime.now().isoformat())
        }

class MovingAverageStrategy(TradingStrategy):
    def __init__(self, symbol: str, parameters: Dict[str, Any] = None):
        if parameters is None:
            parameters = {
                'short_window': 10,
                'long_window': 30,
                'risk_percent': 0.02
            }
        super().__init__('Moving Average Crossover', symbol, parameters)
        
    def should_buy(self, price_data: List[Dict]) -> bool:
        if len(price_data) < self.parameters['long_window']:
            return False
            
        prices = [float(p['close']) for p in price_data]
        short_ma = np.mean(prices[-self.parameters['short_window']:])
        long_ma = np.mean(prices[-self.parameters['long_window']:])
        
        # Previous MAs to check for crossover
        prev_short_ma = np.mean(prices[-self.parameters['short_window']-1:-1])
        prev_long_ma = np.mean(prices[-self.parameters['long_window']-1:-1])
        
        # Buy signal: short MA crosses above long MA
        return (short_ma > long_ma and prev_short_ma <= prev_long_ma)

    def get_signal_details(self, price_data: List[Dict], signal_type: str) -> Dict:
        """Get detailed MA signal information"""
        if len(price_data) < self.parameters['long_window']:
            return super().get_signal_details(price_data, signal_type)
        
        prices = [float(p['close']) for p in price_data]
        short_ma = np.mean(prices[-self.parameters['short_window']:])
        long_ma = np.mean(prices[-self.parameters['long_window']:])
        prev_short_ma = np.mean(prices[-self.parameters['short_window']-1:-1])
        prev_long_ma = np.mean(prices[-self.parameters['long_window']-1:-1])
        
        return {
            'signal_type': signal_type,
            'strategy': self.name,
            'timestamp': datetime.now().isoformat(),
            'current_price': prices[-1],
            'short_ma': round(short_ma, 2),
            'long_ma': round(long_ma, 2),
            'prev_short_ma': round(prev_short_ma, 2),
            'prev_long_ma': round(prev_long_ma, 2),
            'ma_spread': round(short_ma - long_ma, 2),
            'crossover_strength': round(abs(short_ma - long_ma), 2),
            'short_window': self.parameters['short_window'],
            'long_window': self.parameters['long_window']
        }

    def get_reasoning(self, price_data: List[Dict], signal_type: str) -> str:
        """Get human-readable MA reasoning"""
        details = self.get_signal_details(price_data, signal_type)
        
        if signal_type == 'BUY':
            return f"MA {details['short_window']}-period ({details['short_ma']}) crossed above {details['long_window']}-period ({details['long_ma']}) → BULLISH crossover → BUY signal"
        elif signal_type == 'SELL':
            return f"MA {details['short_window']}-period ({details['short_ma']}) crossed below {details['long_window']}-period ({details['long_ma']}) → BEARISH crossover → SELL signal"
        elif signal_type == 'SHORT':
            return f"MA {details['short_window']}-period ({details['short_ma']}) crossed below {details['long_window']}-period ({details['long_ma']}) → BEARISH trend → SHORT signal"
        elif signal_type == 'COVER':
            return f"MA {details['short_window']}-period ({details['short_ma']}) crossed above {details['long_window']}-period ({details['long_ma']}) → BULLISH reversal → COVER signal"
        
        return super().get_reasoning(price_data, signal_type)
    
    def should_sell(self, price_data: List[Dict], current_position: float) -> bool:
        if current_position <= 0 or len(price_data) < self.parameters['long_window']:
            return False
            
        prices = [float(p['close']) for p in price_data]
        short_ma = np.mean(prices[-self.parameters['short_window']:])
        long_ma = np.mean(prices[-self.parameters['long_window']:])
        
        # Previous MAs to check for crossover
        prev_short_ma = np.mean(prices[-self.parameters['short_window']-1:-1])
        prev_long_ma = np.mean(prices[-self.parameters['long_window']-1:-1])
        
        # Sell signal: short MA crosses below long MA
        return (short_ma < long_ma and prev_short_ma >= prev_long_ma)
    
    def should_short(self, price_data: List[Dict]) -> bool:
        """Short when bearish MA crossover occurs"""
        if len(price_data) < self.parameters['long_window']:
            return False
            
        prices = [float(p['close']) for p in price_data]
        short_ma = np.mean(prices[-self.parameters['short_window']:])
        long_ma = np.mean(prices[-self.parameters['long_window']:])
        
        # Previous MAs to check for crossover
        prev_short_ma = np.mean(prices[-self.parameters['short_window']-1:-1])
        prev_long_ma = np.mean(prices[-self.parameters['long_window']-1:-1])
        
        # Short signal: short MA crosses below long MA (bearish crossover)
        return (short_ma < long_ma and prev_short_ma >= prev_long_ma)
    
    def should_cover(self, price_data: List[Dict], current_position: float) -> bool:
        """Cover short when bullish MA crossover occurs"""
        if current_position >= 0 or len(price_data) < self.parameters['long_window']:
            return False
            
        prices = [float(p['close']) for p in price_data]
        short_ma = np.mean(prices[-self.parameters['short_window']:])
        long_ma = np.mean(prices[-self.parameters['long_window']:])
        
        # Previous MAs to check for crossover
        prev_short_ma = np.mean(prices[-self.parameters['short_window']-1:-1])
        prev_long_ma = np.mean(prices[-self.parameters['long_window']-1:-1])
        
        # Cover signal: short MA crosses above long MA (bullish crossover)
        return (short_ma > long_ma and prev_short_ma <= prev_long_ma)

class RSIStrategy(TradingStrategy):
    def __init__(self, symbol: str, parameters: Dict[str, Any] = None):
        if parameters is None:
            parameters = {
                'rsi_period': 14,
                'oversold_level': 30,
                'overbought_level': 70,
                'risk_percent': 0.02
            }
        super().__init__('RSI Mean Reversion', symbol, parameters)
        
    def _calculate_rsi(self, prices: List[float]) -> float:
        """Calculate RSI indicator"""
        if len(prices) < self.parameters['rsi_period'] + 1:
            return 50  # Neutral RSI
            
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-self.parameters['rsi_period']:])
        avg_loss = np.mean(losses[-self.parameters['rsi_period']:])
        
        if avg_loss == 0:
            return 100
            
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def should_buy(self, price_data: List[Dict]) -> bool:
        if len(price_data) < self.parameters['rsi_period'] + 1:
            return False
            
        prices = [float(p['close']) for p in price_data]
        rsi = self._calculate_rsi(prices)
        
        # Buy when RSI is oversold
        return rsi < self.parameters['oversold_level']
    
    def should_sell(self, price_data: List[Dict], current_position: float) -> bool:
        if current_position <= 0 or len(price_data) < self.parameters['rsi_period'] + 1:
            return False
            
        prices = [float(p['close']) for p in price_data]
        rsi = self._calculate_rsi(prices)
        
        # Sell when RSI is overbought
        return rsi > self.parameters['overbought_level']
    
    def should_short(self, price_data: List[Dict]) -> bool:
        """Short when RSI is overbought (>70)"""
        if len(price_data) < self.parameters['rsi_period'] + 1:
            return False
            
        prices = [float(p['close']) for p in price_data]
        rsi = self._calculate_rsi(prices)
        
        # Short when RSI is overbought
        return rsi > self.parameters['overbought_level']
    
    def should_cover(self, price_data: List[Dict], current_position: float) -> bool:
        """Cover short when RSI is oversold (<30)"""
        if current_position >= 0 or len(price_data) < self.parameters['rsi_period'] + 1:
            return False
            
        prices = [float(p['close']) for p in price_data]
        rsi = self._calculate_rsi(prices)
        
        # Cover when RSI is oversold
        return rsi < self.parameters['oversold_level']

    def get_signal_details(self, price_data: List[Dict], signal_type: str) -> Dict:
        """Get detailed RSI signal information"""
        if len(price_data) < self.parameters['rsi_period'] + 1:
            return super().get_signal_details(price_data, signal_type)
        
        prices = [float(p['close']) for p in price_data]
        rsi = self._calculate_rsi(prices)
        
        return {
            'signal_type': signal_type,
            'strategy': self.name,
            'timestamp': datetime.now().isoformat(),
            'current_price': prices[-1],
            'rsi': round(rsi, 2),
            'rsi_period': self.parameters['rsi_period'],
            'overbought_level': self.parameters['overbought_level'],
            'oversold_level': self.parameters['oversold_level'],
            'rsi_condition': 'overbought' if rsi > self.parameters['overbought_level'] else 'oversold' if rsi < self.parameters['oversold_level'] else 'neutral'
        }

    def get_reasoning(self, price_data: List[Dict], signal_type: str) -> str:
        """Get human-readable RSI reasoning"""
        details = self.get_signal_details(price_data, signal_type)
        rsi = details['rsi']
        
        if signal_type == 'BUY':
            return f"RSI {rsi} < {details['oversold_level']} (oversold) → BULLISH reversal → BUY signal"
        elif signal_type == 'SELL':
            return f"RSI {rsi} > {details['overbought_level']} (overbought) → BEARISH reversal → SELL signal"
        elif signal_type == 'SHORT':
            return f"RSI {rsi} > {details['overbought_level']} (overbought) → BEARISH momentum → SHORT signal"
        elif signal_type == 'COVER':
            return f"RSI {rsi} < {details['oversold_level']} (oversold) → BULLISH reversal → COVER signal"
        
        return super().get_reasoning(price_data, signal_type)

class BollingerBandsStrategy(TradingStrategy):
    def __init__(self, symbol: str, parameters: Dict[str, Any] = None):
        if parameters is None:
            parameters = {
                'period': 20,
                'std_dev': 2.0,
                'risk_percent': 0.10
            }
        super().__init__('Bollinger Bands', symbol, parameters)
        
    def should_buy(self, price_data: List[Dict]) -> bool:
        if len(price_data) < self.parameters['period']:
            return False
            
        prices = [float(p['close']) for p in price_data]
        recent_prices = prices[-self.parameters['period']:]
        
        mean_price = np.mean(recent_prices)
        std_price = np.std(recent_prices)
        current_price = prices[-1]
        
        # Lower Bollinger Band
        lower_band = mean_price - (self.parameters['std_dev'] * std_price)
        
        # Buy signal: price touches or goes below lower band (oversold)
        return current_price <= lower_band
    
    def should_sell(self, price_data: List[Dict], current_position: float) -> bool:
        if current_position <= 0 or len(price_data) < self.parameters['period']:
            return False
            
        prices = [float(p['close']) for p in price_data]
        recent_prices = prices[-self.parameters['period']:]
        
        mean_price = np.mean(recent_prices)
        std_price = np.std(recent_prices)
        current_price = prices[-1]
        
        # Upper Bollinger Band
        upper_band = mean_price + (self.parameters['std_dev'] * std_price)
        
        # Sell signal: price touches or goes above upper band (overbought)
        return current_price >= upper_band
    
    def should_short(self, price_data: List[Dict]) -> bool:
        """Short when price breaks above upper Bollinger Band"""
        if len(price_data) < self.parameters['period']:
            return False
            
        prices = [float(p['close']) for p in price_data]
        recent_prices = prices[-self.parameters['period']:]
        
        mean_price = np.mean(recent_prices)
        std_price = np.std(recent_prices)
        current_price = prices[-1]
        
        # Upper Bollinger Band
        upper_band = mean_price + (self.parameters['std_dev'] * std_price)
        
        # Short when price breaks above upper band (expect reversion)
        return current_price >= upper_band
    
    def should_cover(self, price_data: List[Dict], current_position: float) -> bool:
        """Cover short when price hits lower Bollinger Band"""
        if current_position >= 0 or len(price_data) < self.parameters['period']:
            return False
            
        prices = [float(p['close']) for p in price_data]
        recent_prices = prices[-self.parameters['period']:]
        
        mean_price = np.mean(recent_prices)
        std_price = np.std(recent_prices)
        current_price = prices[-1]
        
        # Lower Bollinger Band
        lower_band = mean_price - (self.parameters['std_dev'] * std_price)
        
        # Cover when price hits lower band (mean reversion complete)
        return current_price <= lower_band

    def get_signal_details(self, price_data: List[Dict], signal_type: str) -> Dict:
        """Get detailed Bollinger Bands signal information"""
        if len(price_data) < self.parameters['period']:
            return super().get_signal_details(price_data, signal_type)
        
        prices = [float(p['close']) for p in price_data]
        recent_prices = prices[-self.parameters['period']:]
        
        mean_price = np.mean(recent_prices)
        std_price = np.std(recent_prices)
        current_price = prices[-1]
        
        upper_band = mean_price + (self.parameters['std_dev'] * std_price)
        lower_band = mean_price - (self.parameters['std_dev'] * std_price)
        
        # Calculate band position (0 = lower band, 0.5 = middle, 1 = upper band)
        band_position = (current_price - lower_band) / (upper_band - lower_band) if upper_band != lower_band else 0.5
        
        return {
            'signal_type': signal_type,
            'strategy': self.name,
            'timestamp': datetime.now().isoformat(),
            'current_price': current_price,
            'upper_band': round(upper_band, 2),
            'lower_band': round(lower_band, 2),
            'middle_band': round(mean_price, 2),
            'band_width': round(upper_band - lower_band, 2),
            'band_position': round(band_position, 3),
            'std_dev': self.parameters['std_dev'],
            'period': self.parameters['period'],
            'position_description': 'above upper band' if current_price > upper_band else 'below lower band' if current_price < lower_band else 'within bands'
        }

    def get_reasoning(self, price_data: List[Dict], signal_type: str) -> str:
        """Get human-readable Bollinger Bands reasoning"""
        details = self.get_signal_details(price_data, signal_type)
        
        if signal_type == 'BUY':
            return f"Price €{details['current_price']} ≤ Lower Band €{details['lower_band']} (oversold) → BULLISH reversal → BUY signal"
        elif signal_type == 'SELL':
            return f"Price €{details['current_price']} ≥ Upper Band €{details['upper_band']} (overbought) → BEARISH reversal → SELL signal"
        elif signal_type == 'SHORT':
            return f"Price €{details['current_price']} ≥ Upper Band €{details['upper_band']} (breakout) → BEARISH reversion → SHORT signal"
        elif signal_type == 'COVER':
            return f"Price €{details['current_price']} ≤ Lower Band €{details['lower_band']} (oversold) → BULLISH bounce → COVER signal"
        
        return super().get_reasoning(price_data, signal_type)

class StrategyManager:
    def __init__(self):
        self.available_strategies = {
            'moving_average': MovingAverageStrategy,
            'rsi': RSIStrategy,
            'bollinger_bands': BollingerBandsStrategy
        }
        
    def create_strategy(self, strategy_type: str, symbol: str, parameters: Dict[str, Any] = None):
        """Create a strategy instance"""
        if strategy_type not in self.available_strategies:
            raise ValueError(f"Unknown strategy type: {strategy_type}")
            
        strategy_class = self.available_strategies[strategy_type]
        return strategy_class(symbol, parameters)
    
    def get_strategy_parameters(self, strategy_type: str) -> Dict[str, Any]:
        """Get default parameters for a strategy type"""
        defaults = {
            'moving_average': {
                'short_window': 10,
                'long_window': 30,
                'risk_percent': 0.02
            },
            'rsi': {
                'rsi_period': 14,
                'oversold_level': 30,
                'overbought_level': 70,
                'risk_percent': 0.02
            },
            'bollinger_bands': {
                'period': 20,
                'std_dev': 2.0,
                'risk_percent': 0.02
            }
        }
        return defaults.get(strategy_type, {})
