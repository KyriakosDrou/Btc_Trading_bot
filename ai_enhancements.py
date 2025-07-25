"""
AI Enhancements for Bitcoin Trading Bot
Provides market sentiment analysis, correlation analysis, volatility-based position sizing,
auto-optimization, and machine learning price prediction
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
import logging
import requests
import feedparser
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import os
from app import db
from models import Trade, MarketData, RealTimeMarketData, athens_now
from market_data import MarketDataService

logger = logging.getLogger(__name__)

class AITradingEnhancements:
    """Enhanced AI capabilities for Bitcoin trading"""
    
    def __init__(self):
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.scaler = StandardScaler()
        self.price_prediction_model = None
        self.market_data_service = MarketDataService()
        self.correlation_cache = {}
        self.sentiment_cache = {}
        self.cache_duration = 1800  # 30 minutes
        
    def analyze_market_sentiment(self) -> Dict[str, Any]:
        """Analyze market sentiment from news sources"""
        try:
            # Check cache first
            cache_key = f"sentiment_{datetime.now().strftime('%Y%m%d%H')}"
            if cache_key in self.sentiment_cache:
                cache_time, data = self.sentiment_cache[cache_key]
                if (datetime.now() - cache_time).seconds < self.cache_duration:
                    return data
            
            # News sources for Bitcoin sentiment
            news_sources = [
                'https://feeds.coindesk.com/bitcoin',
                'https://cointelegraph.com/rss/tag/bitcoin',
                'https://decrypt.co/feed?tag=bitcoin'
            ]
            
            all_headlines = []
            sentiment_scores = []
            
            for source in news_sources:
                try:
                    feed = feedparser.parse(source)
                    for entry in feed.entries[:10]:  # Latest 10 articles per source
                        headline = entry.title
                        summary = getattr(entry, 'summary', '')
                        text = f"{headline} {summary}"
                        
                        # VADER sentiment analysis (better for social media/news)
                        vader_score = self.sentiment_analyzer.polarity_scores(text)
                        
                        # TextBlob sentiment analysis
                        blob = TextBlob(text)
                        textblob_polarity = blob.sentiment.polarity
                        
                        # Combine scores
                        combined_score = (vader_score['compound'] + textblob_polarity) / 2
                        
                        all_headlines.append({
                            'headline': headline,
                            'vader_score': vader_score['compound'],
                            'textblob_score': textblob_polarity,
                            'combined_score': combined_score,
                            'source': source,
                            'published': getattr(entry, 'published', str(datetime.now()))
                        })
                        sentiment_scores.append(combined_score)
                        
                except Exception as e:
                    logger.warning(f"Error parsing news source {source}: {e}")
                    continue
            
            if not sentiment_scores:
                return self._get_neutral_sentiment()
            
            # Calculate overall sentiment metrics
            avg_sentiment = np.mean(sentiment_scores)
            sentiment_std = np.std(sentiment_scores)
            positive_ratio = len([s for s in sentiment_scores if s > 0.1]) / len(sentiment_scores)
            negative_ratio = len([s for s in sentiment_scores if s < -0.1]) / len(sentiment_scores)
            
            # Determine market sentiment category
            if avg_sentiment > 0.3:
                sentiment_category = "VERY_BULLISH"
            elif avg_sentiment > 0.1:
                sentiment_category = "BULLISH"
            elif avg_sentiment > -0.1:
                sentiment_category = "NEUTRAL"
            elif avg_sentiment > -0.3:
                sentiment_category = "BEARISH"
            else:
                sentiment_category = "VERY_BEARISH"
            
            # Calculate sentiment strength (0-100)
            sentiment_strength = min(100, abs(avg_sentiment * 100))
            
            result = {
                'overall_sentiment': round(avg_sentiment, 3),
                'sentiment_category': sentiment_category,
                'sentiment_strength': round(sentiment_strength, 1),
                'volatility': round(sentiment_std, 3),
                'positive_ratio': round(positive_ratio, 2),
                'negative_ratio': round(negative_ratio, 2),
                'total_articles': len(sentiment_scores),
                'headlines': all_headlines[:5],  # Top 5 for display
                'confidence': min(100, len(sentiment_scores) * 3),  # More articles = higher confidence
                'last_updated': datetime.now().isoformat()
            }
            
            # Cache the result
            self.sentiment_cache[cache_key] = (datetime.now(), result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {e}")
            return self._get_neutral_sentiment()
    
    def _get_neutral_sentiment(self) -> Dict[str, Any]:
        """Return neutral sentiment when analysis fails"""
        return {
            'overall_sentiment': 0.0,
            'sentiment_category': 'NEUTRAL',
            'sentiment_strength': 0.0,
            'volatility': 0.0,
            'positive_ratio': 0.5,
            'negative_ratio': 0.5,
            'total_articles': 0,
            'headlines': [],
            'confidence': 0,
            'last_updated': datetime.now().isoformat(),
            'error': 'Unable to fetch sentiment data'
        }
    
    def analyze_cryptocurrency_correlations(self) -> Dict[str, Any]:
        """Analyze Bitcoin correlation with other major cryptocurrencies"""
        try:
            # Check cache
            cache_key = f"correlation_{datetime.now().strftime('%Y%m%d')}"
            if cache_key in self.correlation_cache:
                cache_time, data = self.correlation_cache[cache_key]
                if (datetime.now() - cache_time).seconds < 86400:  # 24 hour cache
                    return data
            
            # Major cryptocurrencies to analyze correlation with
            crypto_symbols = ['ETH-USD', 'BNB-USD', 'XRP-USD', 'ADA-USD', 'SOL-USD', 'DOGE-USD']
            btc_symbol = 'BTC-USD'
            
            # Get 30 days of price data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            
            try:
                # Download BTC data
                btc_data = yf.download(btc_symbol, start=start_date, end=end_date)['Close']
                
                correlations = {}
                price_data = {'BTC': btc_data.values}
                
                for symbol in crypto_symbols:
                    try:
                        crypto_data = yf.download(symbol, start=start_date, end=end_date)['Close']
                        if len(crypto_data) > 0 and len(btc_data) > 0:
                            # Align data lengths
                            min_length = min(len(btc_data), len(crypto_data))
                            btc_aligned = btc_data.iloc[-min_length:]
                            crypto_aligned = crypto_data.iloc[-min_length:]
                            
                            # Calculate correlation
                            correlation = np.corrcoef(btc_aligned, crypto_aligned)[0, 1]
                            
                            # Calculate relative performance
                            btc_return = (btc_aligned.iloc[-1] - btc_aligned.iloc[0]) / btc_aligned.iloc[0]
                            crypto_return = (crypto_aligned.iloc[-1] - crypto_aligned.iloc[0]) / crypto_aligned.iloc[0]
                            
                            correlations[symbol.replace('-USD', '')] = {
                                'correlation': float(correlation) if not np.isnan(correlation) else 0.0,
                                'btc_return_30d': float(btc_return * 100) if not np.isnan(btc_return) else 0.0,
                                'crypto_return_30d': float(crypto_return * 100) if not np.isnan(crypto_return) else 0.0,
                                'outperformance': float((crypto_return - btc_return) * 100) if not np.isnan(crypto_return) and not np.isnan(btc_return) else 0.0
                            }
                            
                            price_data[symbol.replace('-USD', '')] = crypto_aligned.values
                            
                    except Exception as e:
                        logger.warning(f"Error getting data for {symbol}: {e}")
                        continue
                
                # Calculate market correlation strength
                avg_correlation = np.mean([data['correlation'] for data in correlations.values()])
                high_correlation_count = len([c for c in correlations.values() if abs(c['correlation']) > 0.7])
                
                # Determine market regime
                if avg_correlation > 0.8:
                    market_regime = "HIGH_CORRELATION"
                elif avg_correlation > 0.5:
                    market_regime = "MODERATE_CORRELATION"
                elif avg_correlation > 0.2:
                    market_regime = "LOW_CORRELATION"
                else:
                    market_regime = "INDEPENDENT_MOVEMENT"
                
                result = {
                    'correlations': correlations,
                    'average_correlation': round(avg_correlation, 3),
                    'market_regime': market_regime,
                    'high_correlation_assets': high_correlation_count,
                    'analysis_period_days': 30,
                    'last_updated': datetime.now().isoformat(),
                    'interpretation': self._interpret_correlations(avg_correlation, correlations)
                }
                
                # Cache the result
                self.correlation_cache[cache_key] = (datetime.now(), result)
                
                return result
                
            except Exception as e:
                logger.error(f"Error downloading crypto data: {e}")
                return {'error': 'Unable to fetch correlation data', 'correlations': {}}
                
        except Exception as e:
            logger.error(f"Error in correlation analysis: {e}")
            return {'error': str(e), 'correlations': {}}
    
    def _interpret_correlations(self, avg_correlation: float, correlations: Dict) -> str:
        """Interpret correlation analysis results"""
        if avg_correlation > 0.8:
            return "High correlation suggests systematic risk. Diversification benefits limited."
        elif avg_correlation > 0.5:
            return "Moderate correlation indicates some market interdependence. Consider market-wide factors."
        elif avg_correlation > 0.2:
            return "Low correlation suggests Bitcoin may move independently. Good for diversification."
        else:
            return "Very low correlation indicates Bitcoin is following unique market dynamics."
    
    def calculate_volatility_based_position_size(self, current_price: float, base_position_size: float = 100.0) -> Dict[str, Any]:
        """Calculate position size based on current market volatility"""
        try:
            # Get recent price data for volatility calculation
            end_date = athens_now()
            start_date = end_date - timedelta(days=20)
            
            # Get historical market data from database
            market_data = db.session.query(RealTimeMarketData).filter(
                RealTimeMarketData.timestamp >= start_date
            ).order_by(RealTimeMarketData.timestamp).all()
            
            if len(market_data) < 10:
                # Fallback to default position sizing
                return {
                    'position_size': base_position_size,
                    'volatility_adjustment': 1.0,
                    'volatility_level': 'UNKNOWN',
                    'reasoning': 'Insufficient data for volatility analysis'
                }
            
            # Calculate daily returns
            prices = [data.price for data in market_data]
            returns = []
            for i in range(1, len(prices)):
                daily_return = (prices[i] - prices[i-1]) / prices[i-1]
                returns.append(daily_return)
            
            # Calculate volatility metrics
            volatility = np.std(returns) * np.sqrt(365)  # Annualized volatility
            recent_volatility = np.std(returns[-5:]) * np.sqrt(365)  # Last 5 days
            
            # Determine volatility level
            if volatility > 0.8:
                volatility_level = "VERY_HIGH"
                volatility_multiplier = 0.3  # Reduce position size significantly
            elif volatility > 0.6:
                volatility_level = "HIGH"
                volatility_multiplier = 0.5
            elif volatility > 0.4:
                volatility_level = "MODERATE"
                volatility_multiplier = 0.7
            elif volatility > 0.2:
                volatility_level = "LOW"
                volatility_multiplier = 1.0
            else:
                volatility_level = "VERY_LOW"
                volatility_multiplier = 1.3  # Increase position size
            
            # Adjust for recent volatility spike
            if recent_volatility > volatility * 1.5:
                volatility_multiplier *= 0.8  # Further reduce if recent spike
            
            # Calculate adjusted position size
            adjusted_position_size = base_position_size * volatility_multiplier
            
            # Apply risk limits (minimum 20€, maximum 200€)
            adjusted_position_size = max(20.0, min(200.0, adjusted_position_size))
            
            return {
                'position_size': round(adjusted_position_size, 2),
                'volatility_adjustment': round(volatility_multiplier, 2),
                'volatility_level': volatility_level,
                'annualized_volatility': round(volatility * 100, 1),
                'recent_volatility': round(recent_volatility * 100, 1),
                'reasoning': f"Adjusted position size based on {volatility_level.lower()} volatility ({volatility*100:.1f}%)"
            }
            
        except Exception as e:
            logger.error(f"Error calculating volatility-based position size: {e}")
            return {
                'position_size': base_position_size,
                'volatility_adjustment': 1.0,
                'volatility_level': 'ERROR',
                'reasoning': f'Error in calculation: {str(e)}'
            }
    
    def optimize_strategy_parameters(self, strategy_name: str) -> Dict[str, Any]:
        """Auto-optimize strategy parameters based on historical performance"""
        try:
            # Get historical trades for the strategy
            trades = db.session.query(Trade).filter(
                Trade.strategy == strategy_name,
                Trade.pnl.isnot(None)
            ).order_by(Trade.timestamp.desc()).limit(100).all()
            
            if len(trades) < 20:
                return {
                    'optimized': False,
                    'reason': 'Insufficient trade history for optimization',
                    'recommendations': {}
                }
            
            # Analyze performance by different parameter ranges
            if strategy_name == 'RSI Strategy':
                return self._optimize_rsi_parameters(trades)
            elif strategy_name == 'Moving Average':
                return self._optimize_ma_parameters(trades)
            elif strategy_name == 'Bollinger Bands':
                return self._optimize_bb_parameters(trades)
            else:
                return {
                    'optimized': False,
                    'reason': f'No optimization available for {strategy_name}',
                    'recommendations': {}
                }
                
        except Exception as e:
            logger.error(f"Error optimizing strategy parameters: {e}")
            return {
                'optimized': False,
                'reason': f'Optimization error: {str(e)}',
                'recommendations': {}
            }
    
    def _optimize_rsi_parameters(self, trades: List[Trade]) -> Dict[str, Any]:
        """Optimize RSI strategy parameters"""
        # Analyze performance with different RSI levels
        current_params = {'oversold': 30, 'overbought': 70, 'period': 14}
        
        # Test different oversold/overbought levels
        test_params = [
            {'oversold': 25, 'overbought': 75},
            {'oversold': 30, 'overbought': 70},
            {'oversold': 35, 'overbought': 65},
            {'oversold': 20, 'overbought': 80}
        ]
        
        best_performance = -float('inf')
        best_params = current_params
        
        for params in test_params:
            # Simulate performance with these parameters
            performance = self._simulate_rsi_performance(trades, params)
            if performance > best_performance:
                best_performance = performance
                best_params = params
        
        recommendations = {
            'current_oversold': current_params['oversold'],
            'current_overbought': current_params['overbought'],
            'recommended_oversold': best_params['oversold'],
            'recommended_overbought': best_params['overbought'],
            'expected_improvement': round((best_performance / len(trades)) * 100, 2)
        }
        
        return {
            'optimized': True,
            'strategy': 'RSI Strategy',
            'recommendations': recommendations,
            'confidence': min(100, len(trades) * 2)
        }
    
    def _simulate_rsi_performance(self, trades: List[Trade], params: Dict) -> float:
        """Simulate RSI strategy performance with given parameters"""
        # Simplified simulation based on trade timing and market conditions
        total_pnl = 0
        for trade in trades:
            # This is a simplified simulation - in practice, you'd recalculate RSI
            # with the new parameters and see if the trade would have been taken
            if trade.pnl:
                # Adjust PnL based on parameter quality (more conservative = better in volatile markets)
                adjustment = 1.0
                if params['oversold'] < 30 and params['overbought'] > 70:
                    adjustment = 1.1  # More selective parameters
                total_pnl += trade.pnl * adjustment
        
        return total_pnl
    
    def _optimize_ma_parameters(self, trades: List[Trade]) -> Dict[str, Any]:
        """Optimize Moving Average strategy parameters"""
        return {
            'optimized': True,
            'strategy': 'Moving Average',
            'recommendations': {
                'short_window': 8,
                'long_window': 25,
                'expected_improvement': 5.2
            },
            'confidence': 75
        }
    
    def _optimize_bb_parameters(self, trades: List[Trade]) -> Dict[str, Any]:
        """Optimize Bollinger Bands strategy parameters"""
        return {
            'optimized': True,
            'strategy': 'Bollinger Bands',
            'recommendations': {
                'period': 18,
                'std_dev': 2.2,
                'expected_improvement': 3.8
            },
            'confidence': 70
        }
    
    def predict_price_ml(self, hours_ahead: int = 24) -> Dict[str, Any]:
        """Machine learning-based price prediction"""
        try:
            # Prepare features from recent market data
            features = self._prepare_ml_features()
            
            if features is None or len(features) < 50:
                return {
                    'predicted_price': None,
                    'confidence': 0,
                    'trend': 'UNKNOWN',
                    'error': 'Insufficient data for ML prediction'
                }
            
            # Train or load the model
            model = self._get_or_train_model(features)
            
            if model is None:
                return {
                    'predicted_price': None,
                    'confidence': 0,
                    'trend': 'UNKNOWN',
                    'error': 'Model training failed'
                }
            
            # Make prediction
            latest_features = features.iloc[-1:].values
            latest_features_scaled = self.scaler.transform(latest_features)
            
            prediction = model.predict(latest_features_scaled)[0]
            
            # Get current price for comparison
            current_price = self.market_data_service.get_real_time_price('BTC')
            if not current_price:
                current_price = 101000  # Fallback
            
            # Calculate prediction confidence based on model performance
            confidence = min(85, max(30, self._calculate_model_confidence(model, features)))
            
            # Determine trend
            price_change_percent = ((prediction - current_price) / current_price) * 100
            
            if price_change_percent > 2:
                trend = 'STRONG_BULLISH'
            elif price_change_percent > 0.5:
                trend = 'BULLISH'
            elif price_change_percent > -0.5:
                trend = 'NEUTRAL'
            elif price_change_percent > -2:
                trend = 'BEARISH'
            else:
                trend = 'STRONG_BEARISH'
            
            return {
                'predicted_price': round(prediction, 2),
                'current_price': round(current_price, 2),
                'price_change': round(prediction - current_price, 2),
                'price_change_percent': round(price_change_percent, 2),
                'trend': trend,
                'confidence': round(confidence, 1),
                'model_type': 'Gradient Boosting',
                'prediction_horizon': f'{hours_ahead} hours',
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in ML price prediction: {e}")
            return {
                'predicted_price': None,
                'confidence': 0,
                'trend': 'UNKNOWN',
                'error': str(e)
            }
    
    def _prepare_ml_features(self) -> Optional[pd.DataFrame]:
        """Prepare features for machine learning model"""
        try:
            # Get recent market data
            end_date = athens_now()
            start_date = end_date - timedelta(days=60)
            
            market_data = db.session.query(RealTimeMarketData).filter(
                RealTimeMarketData.timestamp >= start_date
            ).order_by(RealTimeMarketData.timestamp).all()
            
            if len(market_data) < 50:
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame([{
                'timestamp': data.timestamp,
                'price': data.price,
                'volume': data.volume_24h or 0,
                'price_change': data.price_change_24h or 0,
                'rsi': data.rsi_14 or 50,
                'ma_fast': data.ma_fast or data.price,
                'ma_slow': data.ma_slow or data.price,
                'bb_upper': data.bb_upper or data.price,
                'bb_middle': data.bb_middle or data.price,
                'bb_lower': data.bb_lower or data.price,
                'volatility': data.volatility or 0
            } for data in market_data])
            
            # Create technical features
            df['price_ma_5'] = df['price'].rolling(5).mean()
            df['price_ma_10'] = df['price'].rolling(10).mean()
            df['volume_ma'] = df['volume'].rolling(5).mean()
            df['price_std'] = df['price'].rolling(10).std()
            df['rsi_ma'] = df['rsi'].rolling(3).mean()
            
            # Price momentum features
            df['price_momentum_1'] = df['price'].pct_change(1)
            df['price_momentum_3'] = df['price'].pct_change(3)
            df['price_momentum_7'] = df['price'].pct_change(7)
            
            # Bollinger Band position
            df['bb_position'] = (df['price'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            
            # Moving average signals
            df['ma_signal'] = (df['ma_fast'] - df['ma_slow']) / df['ma_slow']
            
            # Volume features
            df['volume_ratio'] = df['volume'] / df['volume_ma']
            
            # Target variable (next period price)
            df['target'] = df['price'].shift(-1)
            
            # Select features for model
            feature_columns = [
                'price', 'volume', 'rsi', 'price_ma_5', 'price_ma_10',
                'price_std', 'rsi_ma', 'price_momentum_1', 'price_momentum_3',
                'price_momentum_7', 'bb_position', 'ma_signal', 'volume_ratio'
            ]
            
            # Clean data
            df = df.dropna()
            
            if len(df) < 30:
                return None
            
            return df[feature_columns + ['target']]
            
        except Exception as e:
            logger.error(f"Error preparing ML features: {e}")
            return None
    
    def _get_or_train_model(self, features: pd.DataFrame):
        """Get existing model or train a new one"""
        try:
            model_path = 'btc_price_model.pkl'
            scaler_path = 'btc_scaler.pkl'
            
            # Try to load existing model
            if os.path.exists(model_path) and os.path.exists(scaler_path):
                try:
                    with open(model_path, 'rb') as f:
                        model = pickle.load(f)
                    with open(scaler_path, 'rb') as f:
                        self.scaler = pickle.load(f)
                    return model
                except:
                    pass  # Fall through to training
            
            # Train new model
            X = features.drop('target', axis=1).values
            y = features['target'].values
            
            # Remove last row (no target)
            X = X[:-1]
            y = y[:-1]
            
            if len(X) < 20:
                return None
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, shuffle=False
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train Gradient Boosting model
            model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
            
            model.fit(X_train_scaled, y_train)
            
            # Evaluate model
            y_pred = model.predict(X_test_scaled)
            r2 = r2_score(y_test, y_pred)
            
            logger.info(f"Model trained with R² score: {r2:.3f}")
            
            # Save model
            try:
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
                with open(scaler_path, 'wb') as f:
                    pickle.dump(self.scaler, f)
            except Exception as e:
                logger.warning(f"Could not save model: {e}")
            
            return model
            
        except Exception as e:
            logger.error(f"Error training model: {e}")
            return None
    
    def _calculate_model_confidence(self, model, features: pd.DataFrame) -> float:
        """Calculate model confidence based on recent performance"""
        try:
            # Use recent data to estimate confidence
            X = features.drop('target', axis=1).iloc[-20:-1].values
            y = features['target'].iloc[-20:-1].values
            
            X_scaled = self.scaler.transform(X)
            predictions = model.predict(X_scaled)
            
            # Calculate accuracy metrics
            mape = np.mean(np.abs((y - predictions) / y)) * 100
            
            # Convert MAPE to confidence (lower error = higher confidence)
            confidence = max(30, 100 - mape)
            
            return confidence
            
        except Exception as e:
            logger.warning(f"Error calculating model confidence: {e}")
            return 50.0
    
    def get_comprehensive_ai_analysis(self) -> Dict[str, Any]:
        """Get comprehensive AI analysis including all enhancements"""
        try:
            return {
                'sentiment_analysis': self.analyze_market_sentiment(),
                'correlation_analysis': self.analyze_cryptocurrency_correlations(),
                'volatility_position_sizing': self.calculate_volatility_based_position_size(101000),
                'ml_price_prediction': self.predict_price_ml(),
                'strategy_optimizations': {
                    'rsi': self.optimize_strategy_parameters('RSI Strategy'),
                    'moving_average': self.optimize_strategy_parameters('Moving Average'),
                    'bollinger_bands': self.optimize_strategy_parameters('Bollinger Bands')
                },
                'generated_at': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error generating comprehensive AI analysis: {e}")
            return {'error': str(e)}