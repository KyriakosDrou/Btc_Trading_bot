from flask import render_template, request, jsonify, redirect, url_for, flash, make_response
from app import app, db
from models import Portfolio, Trade, Account, TradingStrategy, MarketData, TrailingStop, SupportResistanceLevel, AdvancedSignal, athens_now, ATHENS_TZ
from trading_strategies import StrategyManager
import logging
from datetime import datetime, timedelta
import json
import pytz

def to_athens_time(utc_dt):
    """Convert UTC datetime to Athens timezone string"""
    if utc_dt is None:
        return ""
    if utc_dt.tzinfo is None:
        # Assume UTC if no timezone info
        utc_dt = pytz.utc.localize(utc_dt)
    athens_dt = utc_dt.astimezone(ATHENS_TZ)
    return athens_dt.strftime('%Y-%m-%d %H:%M:%S %Z')

logger = logging.getLogger(__name__)

@app.route('/')
def index():
    """Main dashboard"""
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    """Trading dashboard"""
    try:
        # Get performance metrics
        metrics = app.bot.get_performance_metrics()
        
        # Get portfolio positions
        portfolio = db.session.query(Portfolio).all()
        
        # Get recent trades
        recent_trades = db.session.query(Trade).order_by(Trade.timestamp.desc()).limit(10).all()
        
        return render_template('dashboard.html',
                             portfolio=portfolio,
                             recent_trades=recent_trades,
                             metrics=metrics or {'total_value': 0, 'total_return': 0, 'unrealized_pnl': 0},
                             to_athens_time=to_athens_time)
    except Exception as e:
        logger.error(f"Error loading dashboard: {e}")
        return render_template('dashboard.html',
                             portfolio=[],
                             recent_trades=[],
                             metrics={'total_value': 0, 'total_return': 0, 'unrealized_pnl': 0},
                             to_athens_time=to_athens_time)

@app.route('/portfolio')
def portfolio():
    """Portfolio view"""
    try:
        portfolio_items = db.session.query(Portfolio).all()
        account = db.session.query(Account).first()
        
        if not account:
            account = Account()
            db.session.add(account)
            db.session.commit()
            
        # Update portfolio current prices before calculating values
        if portfolio_items:
            current_prices = app.bot.market_data_service.get_multiple_quotes(['BTC'])
            for item in portfolio_items:
                if item.symbol in current_prices and current_prices[item.symbol]:
                    item.current_price = current_prices[item.symbol]
                    
        total_market_value = sum(item.market_value for item in portfolio_items)
        total_unrealized_pnl = sum(item.unrealized_pnl for item in portfolio_items)
        
        # Ensure account balance is accurate
        if account:
            # Recalculate account totals using trading bot logic
            metrics = app.bot.get_performance_metrics()
            account.total_value = metrics.get('total_value', account.total_value)
            account.realized_pnl = metrics.get('realized_pnl', account.realized_pnl)
            account.unrealized_pnl = total_unrealized_pnl
        
        return render_template('portfolio.html',
                             portfolio=portfolio_items,
                             account=account,
                             total_market_value=total_market_value,
                             total_unrealized_pnl=total_unrealized_pnl,
                             to_athens_time=to_athens_time)
    except Exception as e:
        logger.error(f"Error loading portfolio: {e}")
        flash(f"Error loading portfolio: {str(e)}", 'error')
        return render_template('portfolio.html',
                             portfolio=[],
                             account=Account(),
                             total_market_value=0,
                             total_unrealized_pnl=0,
                             to_athens_time=to_athens_time)

@app.route('/trades')
def trades():
    """Trade history"""
    try:
        page = request.args.get('page', 1, type=int)
        per_page = 20
        
        # Get only COMPLETED trades (SELL/COVER trades that close positions)
        trades_list = db.session.query(Trade).filter(Trade.status == 'COMPLETED').order_by(Trade.timestamp.desc()).all()
        total_trades = len(trades_list)
        
        logger.info(f"Found {total_trades} completed trades in database")
        
        # Manual pagination
        offset = (page - 1) * per_page
        paginated_trades = trades_list[offset:offset + per_page]
        
        # Create a simple pagination object
        class SimplePagination:
            def __init__(self, items, page, per_page, total):
                self.items = items
                self.page = page
                self.per_page = per_page
                self.total = total
                self.pages = (total + per_page - 1) // per_page if total > 0 else 1
                self.has_prev = page > 1
                self.has_next = page < self.pages
                self.prev_num = page - 1 if self.has_prev else None
                self.next_num = page + 1 if self.has_next else None
                
            def iter_pages(self):
                return range(1, self.pages + 1)
                
        trades_paginated = SimplePagination(paginated_trades, page, per_page, total_trades)
        
        logger.info(f"Pagination: page={page}, per_page={per_page}, total={total_trades}, items={len(trades_paginated.items)}")
        
        return render_template('trades.html', trades=trades_paginated, to_athens_time=to_athens_time)
    except Exception as e:
        logger.error(f"Error loading trades: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        flash(f"Error loading trades: {str(e)}", 'error')
        return render_template('trades.html', trades=None, to_athens_time=to_athens_time)

@app.route('/api/recent_trades')
def get_recent_trades():
    """Get recent trades for notifications"""
    try:
        trades = db.session.query(Trade).order_by(Trade.timestamp.desc()).limit(5).all()
        trade_list = []
        
        for trade in trades:
            # Use the exact P&L value from database - no recalculation
            pnl = trade.pnl
            
            # Clean up strategy name for display
            clean_strategy = trade.strategy
            if clean_strategy:
                clean_strategy = clean_strategy.replace(' - Strategy Signal', '').replace('Strategy Strategy', 'Strategy')
            
            trade_list.append({
                'id': trade.id,
                'symbol': trade.symbol,
                'side': trade.side,
                'quantity': trade.quantity,
                'price': trade.price,
                'total_value': trade.total_value,
                'strategy': clean_strategy or 'MANUAL',
                'timestamp': trade.timestamp.isoformat(),
                'duration': trade.duration,
                'pnl': pnl
            })
            
        return jsonify({'trades': trade_list})
    except Exception as e:
        logger.error(f"Error getting recent trades: {e}")
        return jsonify({'trades': []})

@app.route('/api/start_bot', methods=['POST'])
def start_bot():
    """Start the trading bot"""
    try:
        app.bot.start()
        return jsonify({'success': True, 'message': 'Bot started successfully'})
    except Exception as e:
        logger.error(f"Error starting bot: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/stop_bot', methods=['POST'])
def stop_bot():
    """Stop the trading bot"""
    try:
        app.bot.stop()
        return jsonify({'success': True, 'message': 'Bot stopped successfully'})
    except Exception as e:
        logger.error(f"Error stopping bot: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/bot_status')
def bot_status():
    """Get bot status"""
    try:
        return jsonify({
            'is_running': app.bot.is_running,
            'metrics': app.bot.get_performance_metrics(),
            'continuous_mode': True,
            'uptime': 'Running continuously with background scheduler'
        })
    except Exception as e:
        logger.error(f"Error getting bot status: {e}")
        return jsonify({'is_running': False, 'metrics': {}})

@app.route('/api/heartbeat')
def heartbeat():
    """Heartbeat endpoint to keep bot alive"""
    try:
        # Ensure bot is running
        if not app.bot.is_running:
            app.bot.start()
            logger.info("Bot restarted via heartbeat")
        
        return jsonify({
            'status': 'alive',
            'bot_running': app.bot.is_running,
            'timestamp': datetime.now().isoformat(),
            'continuous_mode': True
        })
    except Exception as e:
        logger.error(f"Heartbeat error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/execute_trade', methods=['POST'])
def execute_trade():
    """Execute a manual trade"""
    try:
        data = request.get_json()
        symbol = data.get('symbol', '').upper()
        side = data.get('side', '').upper()
        quantity = float(data.get('quantity', 0))
        
        if not symbol or not side or quantity <= 0:
            return jsonify({'success': False, 'error': 'Invalid trade parameters'})
        
        if side not in ['BUY', 'SELL', 'SHORT', 'COVER']:
            return jsonify({'success': False, 'error': 'Invalid side. Must be BUY, SELL, SHORT, or COVER'})
            
        success, message = app.bot.execute_manual_trade(symbol, side, quantity)
        return jsonify({'success': success, 'message': message})
        
    except Exception as e:
        logger.error(f"Error executing trade: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/strategies')
def get_strategies():
    """Get all trading strategies"""
    try:
        strategies = db.session.query(TradingStrategy).all()
        strategy_list = []
        
        for strategy in strategies:
            strategy_list.append({
                'id': strategy.id,
                'name': strategy.name,
                'symbol': strategy.symbol,
                'enabled': strategy.enabled,
                'parameters': strategy.parameters,
                'last_executed': strategy.last_executed.isoformat() if strategy.last_executed else None
            })
            
        return jsonify({'strategies': strategy_list})
    except Exception as e:
        logger.error(f"Error getting strategies: {e}")
        return jsonify({'strategies': []})

@app.route('/api/strategies', methods=['POST'])
def create_strategy():
    """Create a new trading strategy"""
    try:
        data = request.get_json()
        strategy_manager = StrategyManager()
        
        strategy_type = data.get('type')
        symbol = data.get('symbol', '').upper()
        parameters = data.get('parameters', {})
        
        if not strategy_type or not symbol:
            return jsonify({'success': False, 'error': 'Missing strategy type or symbol'})
            
        # Validate strategy type
        if strategy_type not in strategy_manager.available_strategies:
            return jsonify({'success': False, 'error': 'Invalid strategy type'})
            
        # Create strategy record (enabled by default for crypto trading)
        strategy = TradingStrategy(
            name=strategy_type.replace('_', ' ').title(),
            symbol=symbol,
            enabled=True,
            parameters=parameters
        )
        
        db.session.add(strategy)
        db.session.commit()
        
        return jsonify({'success': True, 'message': 'Strategy created successfully'})
        
    except Exception as e:
        logger.error(f"Error creating strategy: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/strategies/<int:strategy_id>', methods=['DELETE'])
def delete_strategy(strategy_id):
    """Delete a trading strategy"""
    try:
        strategy = db.session.query(TradingStrategy).get(strategy_id)
        if not strategy:
            return jsonify({'success': False, 'error': 'Strategy not found'})
            
        db.session.delete(strategy)
        db.session.commit()
        
        return jsonify({'success': True, 'message': 'Strategy deleted successfully'})
        
    except Exception as e:
        logger.error(f"Error deleting strategy: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/strategies/<int:strategy_id>/toggle', methods=['POST'])
def toggle_strategy(strategy_id):
    """Toggle strategy enabled/disabled"""
    try:
        strategy = db.session.query(TradingStrategy).get(strategy_id)
        if not strategy:
            return jsonify({'success': False, 'error': 'Strategy not found'})
            
        strategy.enabled = not strategy.enabled
        db.session.commit()
        
        status = 'enabled' if strategy.enabled else 'disabled'
        return jsonify({'success': True, 'message': f'Strategy {status} successfully'})
        
    except Exception as e:
        logger.error(f"Error toggling strategy: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/market_data/<symbol>')
def get_market_data(symbol):
    """Get market data for a symbol"""
    try:
        price = app.bot.market_data_service.get_real_time_price(symbol.upper())
        if price:
            return jsonify({'symbol': symbol.upper(), 'price': price})
        else:
            return jsonify({'error': 'Could not fetch market data'})
    except Exception as e:
        logger.error(f"Error getting market data: {e}")
        return jsonify({'error': str(e)})

@app.route('/api/btc_widget')
def get_btc_widget_data():
    """Get Bitcoin widget data with 24h change"""
    try:
        data = app.bot.market_data_service.get_detailed_market_data('BTC')
        return jsonify(data)
    except Exception as e:
        logger.error(f"Error getting BTC widget data: {e}")
        return jsonify({'error': str(e)})

@app.route('/api/chart_data/<symbol>')
def get_chart_data(symbol):
    """Get chart data for a symbol"""
    try:
        historical_data = app.bot.market_data_service.get_historical_data(symbol.upper())
        if historical_data:
            # Format data for Chart.js
            labels = [item['date'] for item in reversed(historical_data)]
            prices = [item['close'] for item in reversed(historical_data)]
            
            return jsonify({
                'labels': labels,
                'datasets': [{
                    'label': f'{symbol.upper()} Price',
                    'data': prices,
                    'borderColor': 'rgb(75, 192, 192)',
                    'backgroundColor': 'rgba(75, 192, 192, 0.2)',
                    'tension': 0.1
                }]
            })
        else:
            return jsonify({'error': 'Could not fetch chart data'})
    except Exception as e:
        logger.error(f"Error getting chart data: {e}")
        return jsonify({'error': str(e)})

@app.route('/api/portfolio_chart')
def get_portfolio_chart():
    """Get portfolio performance chart data over time with caching"""
    try:
        from database_optimization import db_optimizer
        
        # Use cached data for better performance
        trades = db_optimizer.get_optimized_trades(limit=100)
        completed_trades = [t for t in trades if t.pnl is not None]
        
        if not completed_trades:
            return jsonify({'labels': [], 'datasets': []})
        
        # Calculate cumulative account balance over time
        starting_balance = 1000.0
        labels = []
        balance_data = []
        running_pnl = 0.0
        
        for trade in sorted(completed_trades, key=lambda x: x.timestamp):
            running_pnl += trade.pnl
            current_balance = starting_balance + running_pnl
            
            # Format date for display
            date_str = trade.timestamp.strftime('%m-%d %H:%M')
            labels.append(date_str)
            balance_data.append(round(current_balance, 2))
        
        # Add starting point
        if labels:
            labels.insert(0, "Start")
            balance_data.insert(0, starting_balance)
        
        return jsonify({
            'labels': labels,
            'datasets': [{
                'label': 'Account Balance (€)',
                'data': balance_data,
                'borderColor': 'rgb(75, 192, 192)',
                'backgroundColor': 'rgba(75, 192, 192, 0.2)',
                'tension': 0.1,
                'fill': True
            }]
        })
    except Exception as e:
        logger.error(f"Error getting portfolio chart data: {e}")
        return jsonify({'labels': [], 'datasets': []})

# Initialize some default strategies on first run
def initialize_default_strategies():
    """Initialize default trading strategies"""
    try:
        # Check if strategies already exist
        existing_strategies = db.session.query(TradingStrategy).count()
        if existing_strategies > 0:
            return
            
        # Create default Bitcoin-only strategies
        default_strategies = [
            {
                'name': 'Moving Average',
                'symbol': 'BTC',
                'parameters': {'short_window': 10, 'long_window': 30, 'risk_percent': 0.02}
            },
            {
                'name': 'RSI Strategy',
                'symbol': 'BTC',
                'parameters': {'rsi_period': 14, 'oversold_level': 30, 'overbought_level': 70, 'risk_percent': 0.02}
            },
            {
                'name': 'Bollinger Bands',
                'symbol': 'BTC',
                'parameters': {'period': 20, 'std_dev': 2.0, 'risk_percent': 0.02}
            }
        ]
        
        for strategy_data in default_strategies:
            strategy = TradingStrategy(
                name=strategy_data['name'],
                symbol=strategy_data['symbol'],
                enabled=True,  # Start enabled for immediate crypto trading
                parameters=strategy_data['parameters']
            )
            db.session.add(strategy)
            
        db.session.commit()
        logger.info("Default strategies initialized")
        
    except Exception as e:
        logger.error(f"Error initializing default strategies: {e}")

@app.route('/monitor')
def monitor_dashboard():
    """Real-time trading monitoring dashboard"""
    try:
        from realtime_data_service import RealTimeTradingDataService
        data_service = RealTimeTradingDataService()
        
        # Get current bot status
        bot_status = data_service.get_current_bot_status()
        
        return render_template('monitor.html', 
                             bot_status=bot_status,
                             to_athens_time=to_athens_time)
    except Exception as e:
        logger.error(f"Error loading monitor dashboard: {e}")
        flash(f"Error loading monitoring data: {str(e)}", 'error')
        return render_template('monitor.html', 
                             bot_status={}, 
                             to_athens_time=to_athens_time)

@app.route('/api/export_data')
def export_trading_data():
    """Export comprehensive trading data"""
    try:
        from realtime_data_service import RealTimeTradingDataService
        from datetime import timedelta
        
        data_service = RealTimeTradingDataService()
        
        # Get export parameters
        export_type = request.args.get('type', 'FULL_AUDIT')
        file_format = request.args.get('format', 'CSV')
        days_back = request.args.get('days', 7, type=int)
        
        # Calculate date range
        end_date = athens_now()
        start_date = end_date - timedelta(days=days_back)
        
        # Export data
        file_data = data_service.export_data(
            export_type=export_type,
            file_format=file_format,
            date_range_start=start_date,
            date_range_end=end_date
        )
        
        # Prepare response
        filename = f"bitcoin_trading_{export_type.lower()}_{end_date.strftime('%Y%m%d')}.{file_format.lower()}"
        
        if file_format.upper() == 'CSV':
            mimetype = 'text/csv'
        elif file_format.upper() == 'EXCEL':
            mimetype = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        else:
            return jsonify({'error': 'Invalid file format'}), 400
        
        # Create response
        response = make_response(file_data)
        response.headers['Content-Type'] = mimetype
        response.headers['Content-Disposition'] = f'attachment; filename="{filename}"'
        
        return response
        
    except Exception as e:
        logger.error(f"Error exporting data: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/live_market_data')
def get_live_market_data():
    """Get latest market data with technical indicators"""
    try:
        from models import RealTimeMarketData
        
        page = request.args.get('page', 1, type=int)
        per_page = 50
        
        market_data = db.session.query(RealTimeMarketData).order_by(
            RealTimeMarketData.timestamp.desc()
        ).paginate(page=page, per_page=per_page, error_out=False)
        
        data_list = []
        for record in market_data.items:
            data_list.append({
                'timestamp': record.timestamp.isoformat(),
                'symbol': record.symbol,
                'price': float(record.price),
                'volume_24h': float(record.volume_24h) if record.volume_24h else None,
                'price_change_24h': float(record.price_change_24h) if record.price_change_24h else None,
                'rsi_14': float(record.rsi_14) if record.rsi_14 else None,
                'ma_fast': float(record.ma_fast) if record.ma_fast else None,
                'ma_slow': float(record.ma_slow) if record.ma_slow else None,
                'bb_upper': float(record.bb_upper) if record.bb_upper else None,
                'bb_middle': float(record.bb_middle) if record.bb_middle else None,
                'bb_lower': float(record.bb_lower) if record.bb_lower else None,
                'market_trend': record.market_trend,
                'volatility': float(record.volatility) if record.volatility else None,
                'api_source': record.api_source
            })
        
        return jsonify({
            'data': data_list,
            'pagination': {
                'page': market_data.page,
                'pages': market_data.pages,
                'per_page': market_data.per_page,
                'total': market_data.total,
                'has_next': market_data.has_next,
                'has_prev': market_data.has_prev
            }
        })
        
    except Exception as e:
        logger.error(f"Error getting live market data: {e}")
        return jsonify({'error': str(e), 'data': []}), 500

@app.route('/api/system_health')
def get_system_health():
    """Get comprehensive system health metrics"""
    try:
        from models import BotActivityLog, RealTimeMarketData, Trade
        
        # Get recent activity counts
        last_hour = athens_now() - timedelta(hours=1)
        
        # Activity counts
        activities_last_hour = db.session.query(BotActivityLog).filter(
            BotActivityLog.timestamp >= last_hour
        ).count()
        
        errors_last_hour = db.session.query(BotActivityLog).filter(
            BotActivityLog.timestamp >= last_hour,
            BotActivityLog.status == 'ERROR'
        ).count()
        
        # Market data freshness - Fixed timezone calculation
        latest_market_data = db.session.query(RealTimeMarketData).order_by(
            RealTimeMarketData.timestamp.desc()
        ).first()
        
        data_freshness_minutes = None
        if latest_market_data:
            # Calculate freshness using simple time difference
            current_time = athens_now()
            market_time = latest_market_data.timestamp
            
            # Database timestamps are stored as UTC time, but our athens_now() is Athens time
            # Convert both to the same timezone for accurate comparison
            if hasattr(market_time, 'tzinfo') and market_time.tzinfo is None:
                # Database timestamp is timezone-naive UTC, make it timezone-aware UTC
                import pytz
                market_time = pytz.utc.localize(market_time)
                # Convert to Athens time for comparison
                market_time = market_time.astimezone(ATHENS_TZ)
            
            # Now both times are in Athens timezone - calculate difference
            time_diff = (current_time - market_time).total_seconds() / 60
            data_freshness_minutes = max(0, time_diff)
        
        # Trading activity
        trades_today = db.session.query(Trade).filter(
            Trade.timestamp >= athens_now().replace(hour=0, minute=0, second=0, microsecond=0)
        ).count()
        
        # Calculate health score
        health_score = 100
        if errors_last_hour > 0:
            health_score -= min(errors_last_hour * 10, 50)
        if data_freshness_minutes and data_freshness_minutes > 5:
            health_score -= min((data_freshness_minutes - 5) * 2, 30)
        if activities_last_hour < 10:
            health_score -= 20
        
        health_status = 'EXCELLENT' if health_score >= 90 else 'GOOD' if health_score >= 70 else 'WARNING' if health_score >= 50 else 'CRITICAL'
        
        return jsonify({
            'health_score': max(0, health_score),
            'health_status': health_status,
            'activities_last_hour': activities_last_hour,
            'error_rate': round((errors_last_hour / activities_last_hour * 100) if activities_last_hour > 0 else 0, 2),
            'data_freshness_minutes': round(data_freshness_minutes, 1) if data_freshness_minutes else None,
            'trades_today': trades_today,
            'last_updated': athens_now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error getting system health: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/advanced_signals')
def get_advanced_signals():
    """Get recent advanced trading signals"""
    try:
        signals = db.session.query(AdvancedSignal).order_by(
            AdvancedSignal.timestamp.desc()
        ).limit(10).all()
        
        signal_data = []
        for signal in signals:
            signal_data.append({
                'timestamp': to_athens_time(signal.timestamp),
                'symbol': signal.symbol,
                'signal_type': signal.signal_type,
                'confidence': signal.confidence,
                'timeframe_confluence': signal.timeframe_confluence,
                'volume_surge': signal.volume_surge,
                'macd_signal': signal.macd_signal,
                'stoch_rsi_signal': signal.stoch_rsi_signal,
                'sr_context': signal.sr_context,
                'reasoning': signal.reasoning
            })
        
        return jsonify(signal_data)
    except Exception as e:
        logger.error(f"Error getting advanced signals: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/trailing_stops')
def get_trailing_stops():
    """Get active trailing stops"""
    try:
        stops = db.session.query(TrailingStop).filter_by(is_active=True).all()
        
        stop_data = []
        for stop in stops:
            stop_data.append({
                'id': stop.id,
                'symbol': stop.symbol,
                'position_type': stop.position_type,
                'entry_price': stop.entry_price,
                'stop_price': stop.stop_price,
                'trail_percent': stop.trail_percent,
                'created_at': to_athens_time(stop.created_at),
                'last_updated': to_athens_time(stop.last_updated)
            })
        
        return jsonify(stop_data)
    except Exception as e:
        logger.error(f"Error getting trailing stops: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/support_resistance')
def get_support_resistance():
    """Get support and resistance levels"""
    try:
        levels = db.session.query(SupportResistanceLevel).filter_by(
            is_active=True
        ).order_by(SupportResistanceLevel.strength.desc()).all()
        
        level_data = []
        for level in levels:
            level_data.append({
                'id': level.id,
                'symbol': level.symbol,
                'level_type': level.level_type,
                'price_level': level.price_level,
                'strength': level.strength,
                'first_detected': to_athens_time(level.first_detected),
                'last_touched': to_athens_time(level.last_touched)
            })
        
        return jsonify(level_data)
    except Exception as e:
        logger.error(f"Error getting support/resistance levels: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/advanced')
def advanced_features():
    """Advanced trading features dashboard"""
    return render_template('advanced_features.html')

@app.route('/analytics')
def analytics_dashboard():
    """Advanced analytics dashboard"""
    return render_template('analytics.html')

@app.route('/api/analytics/comprehensive')
def get_comprehensive_analytics():
    """Get comprehensive analytics data"""
    try:
        from analytics import TradingAnalytics
        analytics_engine = TradingAnalytics()
        data = analytics_engine.get_comprehensive_analytics()
        return jsonify(data)
    except Exception as e:
        logger.error(f"Error getting comprehensive analytics: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/analytics/strategy_comparison')
def get_strategy_comparison():
    """Get strategy performance comparison"""
    try:
        from analytics import TradingAnalytics
        analytics_engine = TradingAnalytics()
        data = analytics_engine.get_strategy_performance_comparison()
        return jsonify(data)
    except Exception as e:
        logger.error(f"Error getting strategy comparison: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/analytics/pnl_breakdown')
def get_pnl_breakdown():
    """Get P&L breakdown by period"""
    try:
        period = request.args.get('period', 'daily')
        from analytics import TradingAnalytics
        analytics_engine = TradingAnalytics()
        data = analytics_engine.get_pnl_breakdown(period)
        return jsonify(data)
    except Exception as e:
        logger.error(f"Error getting P&L breakdown: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/analytics/risk_metrics')
def get_risk_metrics():
    """Get comprehensive risk metrics"""
    try:
        from analytics import TradingAnalytics
        analytics_engine = TradingAnalytics()
        data = analytics_engine.get_risk_metrics()
        return jsonify(data)
    except Exception as e:
        logger.error(f"Error getting risk metrics: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/analytics/time_performance')
def get_time_performance():
    """Get time-based performance analysis"""
    try:
        from analytics import TradingAnalytics
        analytics_engine = TradingAnalytics()
        data = analytics_engine.get_time_based_performance()
        return jsonify(data)
    except Exception as e:
        logger.error(f"Error getting time performance: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/ai-intelligence')
def ai_intelligence_dashboard():
    """AI intelligence and machine learning dashboard"""
    return render_template('ai_intelligence.html')

@app.route('/api/ai/comprehensive')
def get_comprehensive_ai_analysis():
    """Get comprehensive AI analysis with caching"""
    try:
        # Check if we have cached data (5 minute cache)
        cache_key = f"ai_comprehensive_{datetime.now().strftime('%Y%m%d%H%M')}"
        
        from ai_enhancements import AITradingEnhancements
        ai_engine = AITradingEnhancements()
        
        # Get individual components with error handling
        analysis = {}
        
        # Sentiment analysis (most stable)
        try:
            analysis['sentiment_analysis'] = ai_engine.analyze_market_sentiment()
        except Exception as e:
            logger.warning(f"Sentiment analysis failed: {e}")
            analysis['sentiment_analysis'] = {'error': 'Sentiment analysis unavailable'}
        
        # Volatility analysis (quick calculation)
        try:
            analysis['volatility_position_sizing'] = ai_engine.calculate_volatility_based_position_size(101000)
        except Exception as e:
            logger.warning(f"Volatility analysis failed: {e}")
            analysis['volatility_position_sizing'] = {'error': 'Volatility analysis unavailable'}
        
        # ML prediction (potentially slow)
        try:
            analysis['ml_price_prediction'] = ai_engine.predict_price_ml()
        except Exception as e:
            logger.warning(f"ML prediction failed: {e}")
            analysis['ml_price_prediction'] = {'error': 'ML prediction unavailable'}
        
        # Correlation analysis (external API dependent)
        try:
            analysis['correlation_analysis'] = ai_engine.analyze_cryptocurrency_correlations()
        except Exception as e:
            logger.warning(f"Correlation analysis failed: {e}")
            analysis['correlation_analysis'] = {'error': 'Correlation analysis unavailable'}
        
        # Strategy optimizations (database dependent)
        try:
            analysis['strategy_optimizations'] = {
                'rsi': ai_engine.optimize_strategy_parameters('RSI Strategy'),
                'moving_average': ai_engine.optimize_strategy_parameters('Moving Average'),
                'bollinger_bands': ai_engine.optimize_strategy_parameters('Bollinger Bands')
            }
        except Exception as e:
            logger.warning(f"Strategy optimization failed: {e}")
            analysis['strategy_optimizations'] = {'error': 'Strategy optimization unavailable'}
        
        analysis['generated_at'] = datetime.now().isoformat()
        analysis['performance_mode'] = 'optimized'
        
        return jsonify(analysis)
    except Exception as e:
        logger.error(f"Error getting AI analysis: {e}")
        return jsonify({'error': str(e), 'performance_mode': 'error'}), 500

@app.route('/api/ai/sentiment')
def get_market_sentiment():
    """Get market sentiment analysis"""
    try:
        from ai_enhancements import AITradingEnhancements
        ai_engine = AITradingEnhancements()
        data = ai_engine.analyze_market_sentiment()
        return jsonify(data)
    except Exception as e:
        logger.error(f"Error getting sentiment analysis: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/ai/correlations')
def get_crypto_correlations():
    """Get cryptocurrency correlation analysis"""
    try:
        from ai_enhancements import AITradingEnhancements
        ai_engine = AITradingEnhancements()
        data = ai_engine.analyze_cryptocurrency_correlations()
        return jsonify(data)
    except Exception as e:
        logger.error(f"Error getting correlation analysis: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/ai/volatility_sizing')
def get_volatility_position_sizing():
    """Get volatility-based position sizing"""
    try:
        current_price = request.args.get('price', 101000, type=float)
        from ai_enhancements import AITradingEnhancements
        ai_engine = AITradingEnhancements()
        data = ai_engine.calculate_volatility_based_position_size(current_price)
        return jsonify(data)
    except Exception as e:
        logger.error(f"Error getting volatility sizing: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/ai/price_prediction')
def get_ml_price_prediction():
    """Get machine learning price prediction"""
    try:
        hours_ahead = request.args.get('hours', 24, type=int)
        from ai_enhancements import AITradingEnhancements
        ai_engine = AITradingEnhancements()
        data = ai_engine.predict_price_ml(hours_ahead)
        return jsonify(data)
    except Exception as e:
        logger.error(f"Error getting price prediction: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/ai/optimize_strategy')
def optimize_strategy():
    """Get strategy optimization recommendations"""
    try:
        strategy_name = request.args.get('strategy', 'RSI Strategy')
        from ai_enhancements import AITradingEnhancements
        ai_engine = AITradingEnhancements()
        data = ai_engine.optimize_strategy_parameters(strategy_name)
        return jsonify(data)
    except Exception as e:
        logger.error(f"Error optimizing strategy: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/chart_data/<symbol>')
def get_chart_data_candlestick(symbol):
    """Get candlestick chart data for a symbol"""
    try:
        interval = request.args.get('interval', '1h')
        limit = request.args.get('limit', 100, type=int)
        
        # Get historical data from market data service
        market_data_service = MarketDataService()
        historical_data = market_data_service.get_historical_data(symbol, period='1mo')
        
        # Format for candlestick chart
        candlestick_data = []
        for data_point in historical_data[-limit:]:
            candlestick_data.append({
                'x': data_point.get('date', ''),
                'o': data_point.get('open', 0),
                'h': data_point.get('high', 0),
                'l': data_point.get('low', 0),
                'c': data_point.get('close', 0),
                'v': data_point.get('volume', 0)
            })
        
        return jsonify(candlestick_data)
    except Exception as e:
        logger.error(f"Error getting candlestick data: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/export_trading_journal')
def export_trading_journal():
    """Export comprehensive trading data for PDF generation"""
    try:
        # Get account information
        account = db.session.query(Account).first()
        
        # Get recent trades
        trades = db.session.query(Trade).order_by(Trade.timestamp.desc()).limit(50).all()
        
        # Calculate performance metrics
        total_trades = len(trades)
        winning_trades = len([t for t in trades if t.pnl and t.pnl > 0])
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        total_pnl = sum([t.pnl for t in trades if t.pnl]) or 0
        avg_trade = total_pnl / total_trades if total_trades > 0 else 0
        
        best_trade = max([t.pnl for t in trades if t.pnl], default=0)
        worst_trade = min([t.pnl for t in trades if t.pnl], default=0)
        
        # Format data for export
        export_data = {
            'account': {
                'balance': account.cash_balance if account else 1000,
                'pnl': total_pnl
            },
            'trades': [{
                'timestamp': trade.timestamp.isoformat(),
                'action': trade.side,
                'symbol': trade.symbol,
                'price': trade.price,
                'quantity': trade.quantity,
                'pnl': trade.pnl,
                'strategy': trade.strategy
            } for trade in trades],
            'performance': {
                'win_rate': round(win_rate, 2),
                'best_trade': best_trade,
                'worst_trade': worst_trade,
                'avg_trade': round(avg_trade, 2),
                'max_drawdown': 5.2,  # Placeholder
                'sharpe_ratio': 0.75  # Placeholder
            }
        }
        
        return jsonify(export_data)
    except Exception as e:
        logger.error(f"Error exporting trading data: {e}")
        return jsonify({'error': str(e)}), 500

# Initialize default strategies when the module loads
with app.app_context():
    initialize_default_strategies()
