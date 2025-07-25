"""
Database Performance Optimization for Bitcoin Trading Bot
Implements connection pooling, query optimization, and caching
"""

import logging
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.pool import QueuePool
from app import db
from models import Trade, Portfolio, Account, MarketData, RealTimeMarketData, TradingStrategy

logger = logging.getLogger(__name__)

class DatabaseOptimizer:
    """Optimizes database operations for better performance"""
    
    def __init__(self):
        self.query_cache = {}
        self.cache_ttl = 30  # 30 seconds default cache
        self.setup_connection_pooling()
        self.setup_query_optimization()
        
    def setup_connection_pooling(self):
        """Configure optimized connection pooling"""
        try:
            # Configure connection pool settings
            if hasattr(db.engine, 'pool'):
                db.engine.pool._recycle = 3600  # Recycle connections every hour
                db.engine.pool._pre_ping = True  # Validate connections before use
                
            # Add connection pool monitoring
            @event.listens_for(db.engine, "connect")
            def set_sqlite_pragma(dbapi_connection, connection_record):
                if 'sqlite' in str(db.engine.url):
                    cursor = dbapi_connection.cursor()
                    cursor.execute("PRAGMA journal_mode=WAL")
                    cursor.execute("PRAGMA synchronous=NORMAL")
                    cursor.execute("PRAGMA cache_size=10000")
                    cursor.execute("PRAGMA temp_store=MEMORY")
                    cursor.close()
                    
            logger.info("Database connection pooling configured")
            
        except Exception as e:
            logger.error(f"Error configuring connection pooling: {e}")
    
    def setup_query_optimization(self):
        """Setup query optimization strategies"""
        try:
            # Enable query logging for development
            if logger.isEnabledFor(logging.DEBUG):
                logging.getLogger('sqlalchemy.engine').setLevel(logging.INFO)
                
            # Setup query result caching
            self.cache_stats = {
                'hits': 0,
                'misses': 0,
                'evictions': 0
            }
            
            logger.info("Query optimization configured")
            
        except Exception as e:
            logger.error(f"Error configuring query optimization: {e}")
    
    @contextmanager
    def optimized_session(self):
        """Context manager for optimized database sessions"""
        session = db.session
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            session.close()
    
    def get_cached_query(self, cache_key: str, query_func, ttl: int = None) -> Any:
        """Get cached query result or execute and cache"""
        if ttl is None:
            ttl = self.cache_ttl
            
        # Check cache
        if cache_key in self.query_cache:
            cached_data, cached_time = self.query_cache[cache_key]
            if datetime.now() - cached_time < timedelta(seconds=ttl):
                self.cache_stats['hits'] += 1
                return cached_data
            else:
                # Cache expired
                del self.query_cache[cache_key]
                self.cache_stats['evictions'] += 1
        
        # Execute query and cache result
        self.cache_stats['misses'] += 1
        result = query_func()
        self.query_cache[cache_key] = (result, datetime.now())
        
        return result
    
    def get_optimized_trades(self, limit: int = 50, symbol: str = None) -> List[Trade]:
        """Get trades with optimized query"""
        cache_key = f"trades_{limit}_{symbol}_{datetime.now().minute}"
        
        def query_func():
            query = db.session.query(Trade)
            if symbol:
                query = query.filter(Trade.symbol == symbol)
            return query.order_by(Trade.timestamp.desc()).limit(limit).all()
        
        return self.get_cached_query(cache_key, query_func, ttl=60)
    
    def get_optimized_portfolio(self) -> List[Portfolio]:
        """Get portfolio with optimized query"""
        cache_key = f"portfolio_{datetime.now().minute}"
        
        def query_func():
            return db.session.query(Portfolio).filter(Portfolio.quantity != 0).all()
        
        return self.get_cached_query(cache_key, query_func, ttl=30)
    
    def get_optimized_account(self) -> Optional[Account]:
        """Get account with optimized query"""
        cache_key = f"account_{datetime.now().minute}"
        
        def query_func():
            return db.session.query(Account).first()
        
        return self.get_cached_query(cache_key, query_func, ttl=30)
    
    def get_optimized_market_data(self, symbol: str, hours: int = 24) -> List[RealTimeMarketData]:
        """Get market data with optimized query"""
        cache_key = f"market_data_{symbol}_{hours}_{datetime.now().hour}"
        
        def query_func():
            cutoff_time = datetime.now() - timedelta(hours=hours)
            return db.session.query(RealTimeMarketData).filter(
                RealTimeMarketData.symbol == symbol,
                RealTimeMarketData.timestamp >= cutoff_time
            ).order_by(RealTimeMarketData.timestamp.desc()).all()
        
        return self.get_cached_query(cache_key, query_func, ttl=300)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics with optimized queries"""
        cache_key = f"performance_metrics_{datetime.now().hour}"
        
        def query_func():
            # Get all completed trades efficiently
            completed_trades = db.session.query(Trade).filter(
                Trade.pnl.isnot(None)
            ).all()
            
            if not completed_trades:
                return {
                    'total_trades': 0,
                    'total_pnl': 0,
                    'win_rate': 0,
                    'avg_trade': 0,
                    'best_trade': 0,
                    'worst_trade': 0
                }
            
            # Calculate metrics
            total_trades = len(completed_trades)
            total_pnl = sum(trade.pnl for trade in completed_trades)
            winning_trades = len([t for t in completed_trades if t.pnl > 0])
            win_rate = (winning_trades / total_trades) * 100
            avg_trade = total_pnl / total_trades
            best_trade = max(trade.pnl for trade in completed_trades)
            worst_trade = min(trade.pnl for trade in completed_trades)
            
            return {
                'total_trades': total_trades,
                'total_pnl': round(total_pnl, 2),
                'win_rate': round(win_rate, 2),
                'avg_trade': round(avg_trade, 2),
                'best_trade': round(best_trade, 2),
                'worst_trade': round(worst_trade, 2)
            }
        
        return self.get_cached_query(cache_key, query_func, ttl=300)
    
    def get_strategy_performance(self) -> Dict[str, Any]:
        """Get strategy performance with optimized queries"""
        cache_key = f"strategy_performance_{datetime.now().hour}"
        
        def query_func():
            strategies = db.session.query(TradingStrategy).all()
            strategy_performance = {}
            
            for strategy in strategies:
                trades = db.session.query(Trade).filter(
                    Trade.strategy == strategy.name,
                    Trade.pnl.isnot(None)
                ).all()
                
                if trades:
                    total_pnl = sum(trade.pnl for trade in trades)
                    winning_trades = len([t for t in trades if t.pnl > 0])
                    win_rate = (winning_trades / len(trades)) * 100
                    
                    strategy_performance[strategy.name] = {
                        'total_trades': len(trades),
                        'total_pnl': round(total_pnl, 2),
                        'win_rate': round(win_rate, 2),
                        'avg_trade': round(total_pnl / len(trades), 2),
                        'enabled': strategy.enabled
                    }
            
            return strategy_performance
        
        return self.get_cached_query(cache_key, query_func, ttl=300)
    
    def optimize_database_maintenance(self):
        """Perform database maintenance tasks"""
        try:
            with self.optimized_session() as session:
                # Clean up old market data (keep last 30 days)
                cutoff_date = datetime.now() - timedelta(days=30)
                old_market_data = session.query(RealTimeMarketData).filter(
                    RealTimeMarketData.timestamp < cutoff_date
                ).count()
                
                if old_market_data > 0:
                    session.query(RealTimeMarketData).filter(
                        RealTimeMarketData.timestamp < cutoff_date
                    ).delete()
                    logger.info(f"Cleaned up {old_market_data} old market data records")
                
                # Clean up old trade records (keep last 1000)
                trade_count = session.query(Trade).count()
                if trade_count > 1000:
                    old_trades = session.query(Trade).order_by(
                        Trade.timestamp.desc()
                    ).offset(1000).subquery()
                    
                    deleted_count = session.query(Trade).filter(
                        Trade.id.in_(session.query(old_trades.c.id))
                    ).delete(synchronize_session=False)
                    
                    logger.info(f"Archived {deleted_count} old trade records")
                
                # Update database statistics
                if 'sqlite' in str(db.engine.url):
                    session.execute("ANALYZE")
                elif 'postgresql' in str(db.engine.url):
                    session.execute("VACUUM ANALYZE")
                
                logger.info("Database maintenance completed")
                
        except Exception as e:
            logger.error(f"Error during database maintenance: {e}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        total_requests = self.cache_stats['hits'] + self.cache_stats['misses']
        hit_rate = (self.cache_stats['hits'] / total_requests * 100) if total_requests > 0 else 0
        
        return {
            'cache_size': len(self.query_cache),
            'hit_rate': round(hit_rate, 2),
            'hits': self.cache_stats['hits'],
            'misses': self.cache_stats['misses'],
            'evictions': self.cache_stats['evictions']
        }
    
    def clear_cache(self, pattern: str = None):
        """Clear cache entries matching pattern"""
        if pattern:
            keys_to_remove = [k for k in self.query_cache.keys() if pattern in k]
            for key in keys_to_remove:
                del self.query_cache[key]
            logger.info(f"Cleared {len(keys_to_remove)} cache entries matching '{pattern}'")
        else:
            cache_size = len(self.query_cache)
            self.query_cache.clear()
            logger.info(f"Cleared all {cache_size} cache entries")
    
    def batch_insert_market_data(self, data_list: List[Dict[str, Any]]):
        """Efficiently insert multiple market data records"""
        try:
            with self.optimized_session() as session:
                market_data_objects = []
                for data in data_list:
                    market_data = RealTimeMarketData(**data)
                    market_data_objects.append(market_data)
                
                session.bulk_save_objects(market_data_objects)
                logger.info(f"Batch inserted {len(market_data_objects)} market data records")
                
        except Exception as e:
            logger.error(f"Error batch inserting market data: {e}")
    
    def get_database_health(self) -> Dict[str, Any]:
        """Get database health metrics"""
        try:
            with self.optimized_session() as session:
                # Count records in each table
                trade_count = session.query(Trade).count()
                portfolio_count = session.query(Portfolio).count()
                market_data_count = session.query(RealTimeMarketData).count()
                
                # Get recent activity
                recent_trades = session.query(Trade).filter(
                    Trade.timestamp >= datetime.now() - timedelta(hours=24)
                ).count()
                
                recent_market_data = session.query(RealTimeMarketData).filter(
                    RealTimeMarketData.timestamp >= datetime.now() - timedelta(hours=24)
                ).count()
                
                return {
                    'status': 'healthy',
                    'trade_count': trade_count,
                    'portfolio_count': portfolio_count,
                    'market_data_count': market_data_count,
                    'recent_trades_24h': recent_trades,
                    'recent_market_data_24h': recent_market_data,
                    'cache_stats': self.get_cache_stats()
                }
                
        except Exception as e:
            logger.error(f"Error getting database health: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }

# Global database optimizer instance
db_optimizer = DatabaseOptimizer()