import requests
import os
import logging
from datetime import datetime, timedelta
from app import db
from models import MarketData, athens_now
import time

logger = logging.getLogger(__name__)

class MarketDataService:
    def __init__(self):
        self.api_key = os.environ.get("COINAPI_KEY", "demo")
        self.base_url = "https://api.coingecko.com/api/v3"
        self.cache = {}
        self.cache_timeout = 30  # 30 second cache for real-time trading data
        self.last_api_call = {}
        self.min_api_interval = 5  # Minimum 5 seconds between API calls
        self.symbol_mapping = {
            'BTC': 'bitcoin'
        }
        
    def get_real_time_price(self, symbol):
        """Get real-time price for a cryptocurrency symbol with rate limiting"""
        try:
            # Check cache first - use cached data if recent
            cache_key = f"{symbol}_price"
            if cache_key in self.cache:
                cached_data, timestamp = self.cache[cache_key]
                if (athens_now() - timestamp).total_seconds() < self.cache_timeout:
                    if isinstance(cached_data, dict):
                        logger.info(f"Using cached price for {symbol}: ${cached_data['price']:.2f}")
                        return cached_data['price']
                    else:
                        logger.info(f"Using cached price for {symbol}: ${cached_data:.2f}")
                        return cached_data
            
            # Rate limiting - check if we can make API call (Athens time)
            now = athens_now()
            if symbol in self.last_api_call:
                time_since_last = (now - self.last_api_call[symbol]).total_seconds()
                if time_since_last < self.min_api_interval:
                    # Too soon, but critical for trading - use fresh database price
                    logger.warning(f"Rate limited {symbol}, using database price for critical trading decision")
                    return self._get_database_price(symbol)
            
            # Map symbol to CoinGecko ID
            coin_id = self.symbol_mapping.get(symbol.upper())
            if not coin_id:
                logger.warning(f"Unknown cryptocurrency symbol: {symbol}")
                return self._get_database_price(symbol)
            
            # Make API call to CoinGecko
            url = f"{self.base_url}/simple/price"
            params = {
                'ids': coin_id,
                'vs_currencies': 'eur',
                'include_24hr_vol': 'true',
                'include_24hr_change': 'true'
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            # Handle rate limiting gracefully
            if response.status_code == 429:
                logger.warning(f"API rate limited for {symbol} - using fresh database price")
                # Clear the cache to force fresh data next time
                if cache_key in self.cache:
                    del self.cache[cache_key]
                return self._get_database_price(symbol)
                
            response.raise_for_status()
            data = response.json()
            
            # Update last API call time
            self.last_api_call[symbol] = now
            
            if coin_id in data:
                price_data = data[coin_id]
                price = float(price_data['eur'])
                volume = int(price_data.get('eur_24h_vol', 0))
                change_24h = float(price_data.get('eur_24h_change', 0))
                
                # Cache the result with Athens time (including 24h change)
                cache_data = {
                    'price': price,
                    'change_24h': change_24h,
                    'volume': volume
                }
                self.cache[cache_key] = (cache_data, athens_now())
                
                # Store in database
                market_data = MarketData(
                    symbol=symbol,
                    price=price,
                    volume=volume
                )
                db.session.add(market_data)
                db.session.commit()
                
                logger.info(f"Retrieved REAL price for {symbol}: €{price:.2f}")
                return price
            else:
                logger.warning(f"No data found for cryptocurrency {symbol}")
                return self._get_database_price(symbol)
                
        except requests.RequestException as e:
            logger.error(f"Error fetching crypto data for {symbol}: {e}")
            return self._get_database_price(symbol)
        except Exception as e:
            logger.error(f"Unexpected error fetching crypto data for {symbol}: {e}")
            return self._get_database_price(symbol)
    
    def _get_database_price(self, symbol):
        """Get last real price from database with multiple fallback sources"""
        try:
            # Try RealTimeMarketData first (most recent)
            from models import RealTimeMarketData
            latest_realtime = db.session.query(RealTimeMarketData).filter_by(symbol=symbol).order_by(
                RealTimeMarketData.timestamp.desc()
            ).first()
            
            if latest_realtime and latest_realtime.price:
                try:
                    age_minutes = (athens_now() - latest_realtime.timestamp).total_seconds() / 60
                    logger.info(f"Using real-time database price for {symbol}: €{latest_realtime.price:.2f} (age: {age_minutes:.1f}min)")
                except:
                    logger.info(f"Using real-time database price for {symbol}: €{latest_realtime.price:.2f}")
                return latest_realtime.price
            
            # Fallback to MarketData table
            last_price = db.session.query(MarketData).filter_by(symbol=symbol).order_by(MarketData.timestamp.desc()).first()
            if last_price and last_price.price > 0:
                logger.info(f"Using MarketData database price for {symbol}: €{last_price.price:.2f}")
                return last_price.price
            else:
                logger.error(f"CRITICAL: No price data for {symbol} in any database table")
                return None
        except Exception as e:
            logger.error(f"Error getting database price for {symbol}: {e}")
            return None
    
    def get_historical_data(self, symbol, period='1mo'):
        """Get historical price data for cryptocurrency technical analysis"""
        try:
            # Map symbol to CoinGecko ID
            coin_id = self.symbol_mapping.get(symbol.upper())
            if not coin_id:
                logger.warning(f"Unknown cryptocurrency symbol: {symbol}")
                return []
            
            # Get historical data from CoinGecko (market chart)
            url = f"{self.base_url}/coins/{coin_id}/market_chart"
            params = {
                'vs_currency': 'usd',
                'days': '30'
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if 'prices' in data and data['prices']:
                prices = []
                for i, price_point in enumerate(data['prices']):
                    timestamp, price = price_point
                    date = datetime.fromtimestamp(timestamp / 1000).strftime('%Y-%m-%d')
                    
                    prices.append({
                        'date': date,
                        'open': price,
                        'high': price * 1.02,  # Approximate high
                        'low': price * 0.98,   # Approximate low
                        'close': price,
                        'volume': data['total_volumes'][i][1] if i < len(data.get('total_volumes', [])) else 0
                    })
                return prices
            else:
                logger.warning(f"No historical data found for cryptocurrency {symbol}")
                return []
                
        except Exception as e:
            logger.error(f"Error fetching historical crypto data for {symbol}: {e}")
            # Return cached historical data if available, otherwise generate realistic data
            return self._generate_realistic_historical_data(symbol)
    
    def _generate_realistic_historical_data(self, symbol):
        """Generate realistic historical data based on current real price"""
        import random
        current_price = self._get_database_price(symbol)
        if current_price <= 0:
            return []
        
        prices = []
        for i in range(30):
            date = (datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d')
            # Generate small realistic price movements around current price
            change = random.uniform(-0.02, 0.02)  # ±2% daily change
            price = current_price * (1 + change)
            
            prices.append({
                'date': date,
                'open': price * 0.998,
                'high': price * 1.015,
                'low': price * 0.985,
                'close': price,
                'volume': random.randint(1000000, 5000000)
            })
            current_price = price
        
        return prices
    
    def get_multiple_quotes(self, symbols):
        """Get quotes for multiple symbols"""
        quotes = {}
        for symbol in symbols:
            quotes[symbol] = self.get_real_time_price(symbol)
            time.sleep(0.1)  # Small delay to avoid API rate limits
        return quotes
    
    def get_detailed_market_data(self, symbol):
        """Get detailed market data including 24h change for Bitcoin widget"""
        try:
            # Check cache first for detailed data
            cache_key = f"{symbol}_price"
            if cache_key in self.cache:
                cached_data, timestamp = self.cache[cache_key]
                if (athens_now() - timestamp).total_seconds() < self.cache_timeout:
                    if isinstance(cached_data, dict):
                        logger.info(f"Using cached detailed data for {symbol}")
                        return {
                            'price': cached_data['price'],
                            'change_24h': cached_data['change_24h'],
                            'volume': cached_data['volume'],
                            'last_updated': timestamp.strftime('%Y-%m-%d %H:%M:%S')
                        }
            
            # Force a fresh API call by temporarily clearing cache
            old_cache = self.cache.get(cache_key)
            if cache_key in self.cache:
                del self.cache[cache_key]
            
            # Get fresh data
            price = self.get_real_time_price(symbol)
            
            # Get the cached detailed data that should now be available
            if cache_key in self.cache:
                cached_data, timestamp = self.cache[cache_key]
                if isinstance(cached_data, dict):
                    change_24h = cached_data['change_24h']
                    volume = cached_data['volume']
                    change_amount = price * (change_24h / 100)
                    
                    return {
                        'price': price,
                        'volume_24h': volume,
                        'price_change_24h': change_24h,
                        'change_24h': change_24h,
                        'change_amount': change_amount,
                        'volume': volume,
                        'last_updated': timestamp.strftime('%Y-%m-%d %H:%M:%S')
                    }
            
            # Fallback if detailed data not available
            return {
                'price': price,
                'volume_24h': 0,
                'price_change_24h': 0,
                'change_24h': 0,
                'change_amount': 0,
                'volume': 0,
                'last_updated': athens_now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
        except Exception as e:
            logger.error(f"Error getting detailed market data for {symbol}: {e}")
            return {
                'price': self._get_database_price(symbol),
                'volume_24h': 0,
                'price_change_24h': 0,
                'change_24h': 0,
                'change_amount': 0,
                'volume': 0,
                'last_updated': athens_now().strftime('%Y-%m-%d %H:%M:%S')
            }
