import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from app import db
from models import Portfolio, Trade, Account, TradingStrategy, MarketData, TrailingStop, SupportResistanceLevel, AdvancedSignal
from market_data import MarketDataService
from trading_strategies import StrategyManager
from realtime_data_service import RealTimeTradingDataService
import time
import threading

logger = logging.getLogger(__name__)

class TradingBot:
    def __init__(self):
        self.is_running = False
        self.market_data_service = MarketDataService()
        self.strategy_manager = StrategyManager()
        self.realtime_data_service = RealTimeTradingDataService()
        # Initialize advanced trading engine
        from advanced_strategies import AdvancedTradingEngine
        self.advanced_engine = AdvancedTradingEngine()
        self.trading_thread = None
        self.last_execution = {}
        self.min_execution_interval = 300  # Minimum 5 minutes between strategy executions to prevent over-trading
        
    def start(self):
        """Start the trading bot"""
        if self.is_running:
            logger.warning("Bot is already running")
            return
            
        self.is_running = True
        self.trading_thread = threading.Thread(target=self._run_trading_loop)
        self.trading_thread.daemon = True
        self.trading_thread.start()
        logger.info("Trading bot started")
        
    def stop(self):
        """Stop the trading bot"""
        self.is_running = False
        if self.trading_thread:
            self.trading_thread.join(timeout=5)
        logger.info("Trading bot stopped")
        
    def _run_trading_loop(self):
        """Main trading loop with real-time data logging"""
        from app import app
        while self.is_running:
            try:
                with app.app_context():
                    # Log market data with technical indicators every 30 seconds
                    self.realtime_data_service.log_market_data_with_indicators('BTC')
                    
                    # Update strategy status
                    self.realtime_data_service.update_strategy_status()
                    
                    # Execute trading strategies
                    self._execute_strategies()
                    self._update_portfolio_values()
                    
                    # Log heartbeat activity
                    self.realtime_data_service.log_bot_activity(
                        'HEARTBEAT',
                        'Trading loop completed successfully'
                    )
                    
                time.sleep(30)  # Check every 30 seconds
            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                # Log error activity
                try:
                    with app.app_context():
                        self.realtime_data_service.log_bot_activity(
                            'ERROR',
                            f'Trading loop error: {str(e)}',
                            status='ERROR',
                            error_message=str(e)
                        )
                except:
                    pass
                time.sleep(60)  # Wait longer on error
                
    def _execute_strategies(self):
        """Execute all enabled trading strategies"""
        strategies = db.session.query(TradingStrategy).filter_by(enabled=True).all()
        
        for strategy_config in strategies:
            try:
                # Check if enough time has passed since last execution
                last_exec = self.last_execution.get(strategy_config.id, datetime.min)
                time_since_last = (datetime.now() - last_exec).total_seconds()
                if time_since_last < self.min_execution_interval:
                    continue
                
                # Also check if we recently made a trade in this symbol to prevent over-trading
                recent_trade = db.session.query(Trade).filter(
                    Trade.symbol == strategy_config.symbol,
                    Trade.timestamp > datetime.now() - timedelta(minutes=2)
                ).first()
                
                if recent_trade:
                    continue  # Skip if we made a trade in the last 2 minutes
                    
                # Get historical data for the symbol
                historical_data = self.market_data_service.get_historical_data(strategy_config.symbol)
                if not historical_data:
                    continue
                
                # Run comprehensive advanced signal analysis
                advanced_signal = self.advanced_engine.analyze_comprehensive_signals(
                    strategy_config.symbol, historical_data
                )
                
                # Log advanced signal to database
                self._log_advanced_signal(strategy_config.symbol, advanced_signal)
                    
                # Create strategy instance - map strategy names correctly
                strategy_name_map = {
                    'moving_average': 'moving_average',
                    'rsi_strategy': 'rsi',
                    'bollinger_bands': 'bollinger_bands'
                }
                
                strategy_key = strategy_config.name.lower().replace(' ', '_')
                strategy_type = strategy_name_map.get(strategy_key, strategy_key)
                
                strategy = self.strategy_manager.create_strategy(
                    strategy_type,
                    strategy_config.symbol,
                    strategy_config.parameters
                )
                
                # Get current position
                portfolio_item = db.session.query(Portfolio).filter_by(symbol=strategy_config.symbol).first()
                current_position = portfolio_item.quantity if portfolio_item else 0
                
                # Get current price
                current_price = self.market_data_service.get_real_time_price(strategy_config.symbol)
                if not current_price:
                    continue
                
                # Update trailing stops and check if triggered
                self._update_trailing_stops(strategy_config.symbol, current_price)
                
                # Check position limits (max 3 BTC positions total)
                total_btc_positions = db.session.query(Portfolio).filter_by(symbol='BTC').filter(Portfolio.quantity != 0).count()
                
                # Combine traditional strategy signals with advanced analysis
                use_advanced_signal = advanced_signal['confidence'] > 60  # High confidence threshold
                
                # LONG POSITION LOGIC
                if current_position == 0 or current_position > 0:
                    # Check for buy signal (open long or add to long)
                    traditional_buy_signal = strategy.should_buy(historical_data)
                    advanced_buy_signal = use_advanced_signal and advanced_signal['signal'] == 'BUY'
                    
                    if (traditional_buy_signal or advanced_buy_signal) and current_position == 0 and total_btc_positions < 3:
                        account = self._get_account()
                        position_size = strategy.get_position_size(account.total_value, current_price)
                        
                        if position_size > 0 and (position_size * current_price) <= account.cash_balance:
                            signal_details = strategy.get_signal_details(historical_data, 'BUY')
                            reasoning = strategy.get_reasoning(historical_data, 'BUY')
                            market_conditions = strategy.get_market_conditions(historical_data)
                            
                            self._execute_buy_order(strategy_config.symbol, position_size, current_price, strategy_config.name, signal_details, reasoning, market_conditions)
                            # Create trailing stop for long position
                            self._create_trailing_stop(strategy_config.symbol, current_price, 'LONG')
                            
                    # Check for sell signal or stop loss/take profit (close long)
                    elif current_position > 0:
                        should_sell_strategy = strategy.should_sell(historical_data, current_position)
                        should_sell_risk_mgmt = self._check_stop_loss_take_profit(portfolio_item, current_price, 'LONG')
                        
                        if should_sell_strategy or should_sell_risk_mgmt:
                            reason = "Strategy Signal" if should_sell_strategy else "Stop Loss/Take Profit"
                            risk_trigger = None if should_sell_strategy else reason
                            
                            if should_sell_strategy:
                                signal_details = strategy.get_signal_details(historical_data, 'SELL')
                                reasoning = strategy.get_reasoning(historical_data, 'SELL')
                                market_conditions = strategy.get_market_conditions(historical_data)
                            else:
                                signal_details = {'signal_type': 'SELL', 'risk_management': True}
                                reasoning = f"Position closed due to {reason}"
                                market_conditions = strategy.get_market_conditions(historical_data)
                            
                            self._execute_sell_order(strategy_config.symbol, current_position, current_price, f"{strategy_config.name} - {reason}", signal_details, reasoning, market_conditions, risk_trigger)
                
                # SHORT POSITION LOGIC
                if current_position == 0 or current_position < 0:
                    # Check for short signal (open short)
                    if strategy.should_short(historical_data) and current_position == 0 and total_btc_positions < 3:
                        account = self._get_account()
                        position_size = strategy.get_position_size(account.total_value, current_price)
                        
                        if position_size > 0 and (position_size * current_price) <= account.cash_balance:
                            signal_details = strategy.get_signal_details(historical_data, 'SHORT')
                            reasoning = strategy.get_reasoning(historical_data, 'SHORT')
                            market_conditions = strategy.get_market_conditions(historical_data)
                            
                            self._execute_short_order(strategy_config.symbol, position_size, current_price, strategy_config.name, signal_details, reasoning, market_conditions)
                            # Create trailing stop for short position
                            self._create_trailing_stop(strategy_config.symbol, current_price, 'SHORT')
                            
                    # Check for cover signal or stop loss/take profit (close short)
                    elif current_position < 0:
                        should_cover_strategy = strategy.should_cover(historical_data, current_position)
                        should_cover_risk_mgmt = self._check_stop_loss_take_profit(portfolio_item, current_price, 'SHORT')
                        
                        if should_cover_strategy or should_cover_risk_mgmt:
                            reason = "Strategy Signal" if should_cover_strategy else "Stop Loss/Take Profit"
                            risk_trigger = None if should_cover_strategy else reason
                            
                            if should_cover_strategy:
                                signal_details = strategy.get_signal_details(historical_data, 'COVER')
                                reasoning = strategy.get_reasoning(historical_data, 'COVER')
                                market_conditions = strategy.get_market_conditions(historical_data)
                            else:
                                signal_details = {'signal_type': 'COVER', 'risk_management': True}
                                reasoning = f"Position closed due to {reason}"
                                market_conditions = strategy.get_market_conditions(historical_data)
                            
                            self._execute_cover_order(strategy_config.symbol, abs(current_position), current_price, f"{strategy_config.name} - {reason}", signal_details, reasoning, market_conditions, risk_trigger)
                    
                # Update last execution time
                self.last_execution[strategy_config.id] = datetime.now()
                strategy_config.last_executed = datetime.now()
                db.session.commit()
                
            except Exception as e:
                logger.error(f"Error executing strategy {strategy_config.name}: {e}")
    
    def _check_stop_loss_take_profit(self, portfolio_item: Portfolio, current_price: float, position_type: str) -> bool:
        """Check if position should be closed due to stop loss (-5%) or take profit (+10%)"""
        if not portfolio_item:
            return False
            
        entry_price = portfolio_item.avg_price
        
        if position_type == 'LONG':
            # Long position: profit when price goes up, loss when price goes down
            pct_change = (current_price - entry_price) / entry_price
            # Stop loss at -5%, take profit at +10%
            return pct_change <= -0.05 or pct_change >= 0.10
        
        elif position_type == 'SHORT':
            # Short position: profit when price goes down, loss when price goes up
            pct_change = (entry_price - current_price) / entry_price
            # Stop loss at -5%, take profit at +10%
            return pct_change <= -0.05 or pct_change >= 0.10
            
        return False
                
    def _execute_buy_order(self, symbol: str, quantity: float, price: float, strategy: str, signal_details: dict = None, reasoning: str = None, market_conditions: dict = None, risk_trigger: str = None):
        """Execute a buy order"""
        try:
            total_cost = quantity * price
            account = self._get_account()
            
            if total_cost > account.cash_balance:
                logger.warning(f"Insufficient funds for buy order: {symbol}")
                return False
                
            # Clean up strategy name - remove redundant text
            clean_strategy = strategy.replace(' - Strategy Signal', '').replace('Strategy Strategy', 'Strategy')
            
            # Create trade record
            trade = Trade(
                symbol=symbol,
                side='BUY',
                quantity=quantity,
                price=price,
                total_value=total_cost,
                strategy=clean_strategy,
                status='OPEN',  # BUY trades are open until sold
                signal_details=signal_details,
                reasoning=reasoning,
                market_conditions=market_conditions,
                risk_trigger=risk_trigger
            )
            db.session.add(trade)
            
            # Update portfolio
            portfolio_item = db.session.query(Portfolio).filter_by(symbol=symbol).first()
            if portfolio_item:
                # Update existing position
                total_quantity = portfolio_item.quantity + quantity
                total_cost_basis = (portfolio_item.quantity * portfolio_item.avg_price) + total_cost
                portfolio_item.avg_price = total_cost_basis / total_quantity
                portfolio_item.quantity = total_quantity
                portfolio_item.current_price = price
                portfolio_item.last_updated = datetime.now()
            else:
                # Create new position
                portfolio_item = Portfolio(
                    symbol=symbol,
                    quantity=quantity,
                    avg_price=price,
                    current_price=price
                )
                db.session.add(portfolio_item)
                
            # Update account
            account.cash_balance -= total_cost
            account.last_updated = datetime.now()
            
            db.session.commit()
            
            # Log trade execution to real-time data service
            self.realtime_data_service.log_trade_execution(trade)
            
            logger.info(f"Executed BUY order: {quantity} shares of {symbol} at ${price}")
            return True
            
        except Exception as e:
            logger.error(f"Error executing buy order: {e}")
            db.session.rollback()
            return False
            
    def _execute_sell_order(self, symbol: str, quantity: float, price: float, strategy: str, signal_details: dict = None, reasoning: str = None, market_conditions: dict = None, risk_trigger: str = None):
        """Execute a sell order"""
        try:
            total_proceeds = quantity * price
            
            # Clean up strategy name - remove redundant text
            clean_strategy = strategy.replace(' - Strategy Signal', '').replace('Strategy Strategy', 'Strategy')
            
            # Create trade record
            trade = Trade(
                symbol=symbol,
                side='SELL',
                quantity=quantity,
                price=price,
                total_value=total_proceeds,
                strategy=clean_strategy,
                signal_details=signal_details,
                reasoning=reasoning,
                market_conditions=market_conditions,
                risk_trigger=risk_trigger
            )
            db.session.add(trade)
            
            # Update portfolio
            portfolio_item = db.session.query(Portfolio).filter_by(symbol=symbol).first()
            if portfolio_item and portfolio_item.quantity >= quantity:
                # Calculate realized P&L for the trade
                realized_pnl = (price - portfolio_item.avg_price) * quantity
                
                # Update trade with P&L
                trade.pnl = realized_pnl
                
                # Calculate trade duration for long position sell
                original_buy = db.session.query(Trade).filter(
                    Trade.symbol == symbol,
                    Trade.side == 'BUY',
                    Trade.timestamp < datetime.now()
                ).order_by(Trade.timestamp.desc()).first()
                
                duration_seconds = None
                if original_buy:
                    duration_seconds = int((datetime.now() - original_buy.timestamp).total_seconds())
                
                trade.duration = duration_seconds
                
                # Update position
                portfolio_item.quantity -= quantity
                portfolio_item.current_price = price
                portfolio_item.last_updated = datetime.now()
                
                # If position is closed, remove from portfolio
                if portfolio_item.quantity <= 0:
                    db.session.delete(portfolio_item)
                    
                # Update account
                account = self._get_account()
                account.cash_balance += total_proceeds
                account.realized_pnl += realized_pnl
                account.last_updated = datetime.now()
                
                db.session.commit()
                
                # Log trade execution to real-time data service
                self.realtime_data_service.log_trade_execution(trade)
                
                logger.info(f"Executed SELL order: {quantity} shares of {symbol} at ${price}")
                return True
            else:
                logger.warning(f"Insufficient shares to sell: {symbol}")
                return False
                
        except Exception as e:
            logger.error(f"Error executing sell order: {e}")
            db.session.rollback()
            return False
    
    def _execute_short_order(self, symbol: str, quantity: float, price: float, strategy: str, signal_details: dict = None, reasoning: str = None, market_conditions: dict = None, risk_trigger: str = None):
        """Execute a short order (sell shares we don't own)"""
        try:
            total_proceeds = quantity * price
            account = self._get_account()
            
            # Clean up strategy name - remove redundant text
            clean_strategy = strategy.replace(' - Strategy Signal', '').replace('Strategy Strategy', 'Strategy')
            
            # For SHORT trades, calculate unrealized P&L and duration if there's an existing position
            unrealized_pnl = None
            duration_seconds = None
            
            # If this is adding to an existing short position, calculate current unrealized P&L
            existing_portfolio = db.session.query(Portfolio).filter_by(symbol=symbol).first()
            if existing_portfolio and existing_portfolio.quantity < 0:
                # Calculate unrealized P&L for existing short position
                current_price = price  # Using current execution price
                unrealized_pnl = (existing_portfolio.avg_price - current_price) * abs(existing_portfolio.quantity)
                
                # Find the original short trade for duration
                original_short = db.session.query(Trade).filter(
                    Trade.symbol == symbol,
                    Trade.side == 'SHORT',
                    Trade.timestamp < datetime.now()
                ).order_by(Trade.timestamp.desc()).first()
                
                if original_short:
                    duration_seconds = int((datetime.now() - original_short.timestamp).total_seconds())
            
            # Create trade record
            trade = Trade(
                symbol=symbol,
                side='SHORT',
                quantity=quantity,
                price=price,
                total_value=total_proceeds,
                strategy=clean_strategy,
                pnl=unrealized_pnl,
                duration=duration_seconds,
                status='OPEN',  # SHORT trades are open until covered
                signal_details=signal_details,
                reasoning=reasoning,
                market_conditions=market_conditions,
                risk_trigger=risk_trigger
            )
            db.session.add(trade)
            
            # Update portfolio (negative quantity = short position)
            portfolio_item = db.session.query(Portfolio).filter_by(symbol=symbol).first()
            if portfolio_item:
                # Update existing position (could be adding to short or flipping from long)
                if portfolio_item.quantity >= 0:
                    # Starting fresh short position or flipping from long
                    portfolio_item.quantity = -quantity
                    portfolio_item.avg_price = price
                else:
                    # Adding to existing short position
                    total_quantity = abs(portfolio_item.quantity) + quantity
                    total_cost_basis = (abs(portfolio_item.quantity) * portfolio_item.avg_price) + total_proceeds
                    portfolio_item.avg_price = total_cost_basis / total_quantity
                    portfolio_item.quantity = -total_quantity
                    
                portfolio_item.current_price = price
                portfolio_item.last_updated = datetime.now()
            else:
                # Create new short position
                portfolio_item = Portfolio(
                    symbol=symbol,
                    quantity=-quantity,  # Negative for short
                    avg_price=price,
                    current_price=price
                )
                db.session.add(portfolio_item)
                
            # Update account (we receive cash from short sale)
            account.cash_balance += total_proceeds
            account.last_updated = datetime.now()
            
            db.session.commit()
            
            # Log trade execution to real-time data service
            self.realtime_data_service.log_trade_execution(trade)
            
            logger.info(f"Executed SHORT order: {quantity} shares of {symbol} at ${price}")
            return True
            
        except Exception as e:
            logger.error(f"Error executing short order: {e}")
            db.session.rollback()
            return False
    
    def _execute_cover_order(self, symbol: str, quantity: float, price: float, strategy: str, signal_details: dict = None, reasoning: str = None, market_conditions: dict = None, risk_trigger: str = None):
        """Execute a cover order (buy back shares to close short position)"""
        try:
            total_cost = quantity * price
            account = self._get_account()
            
            if total_cost > account.cash_balance:
                logger.warning(f"Insufficient funds for cover order: {symbol}")
                return False
            
            # Clean up strategy name - remove redundant text
            clean_strategy = strategy.replace(' - Strategy Signal', '').replace('Strategy Strategy', 'Strategy')
            
            # Create trade record
            trade = Trade(
                symbol=symbol,
                side='COVER',
                quantity=quantity,
                price=price,
                total_value=total_cost,
                strategy=clean_strategy,
                signal_details=signal_details,
                reasoning=reasoning,
                market_conditions=market_conditions,
                risk_trigger=risk_trigger,
                status='COMPLETED'
            )
            db.session.add(trade)
            
            # Update portfolio (reduce or close short position)
            portfolio_item = db.session.query(Portfolio).filter_by(symbol=symbol).first()
            if portfolio_item and portfolio_item.quantity < 0:
                # Calculate P&L for closed portion
                entry_price = portfolio_item.avg_price
                pnl = (entry_price - price) * quantity  # Short P&L: profit when price drops
                
                # Calculate trade duration (from when short position was opened)
                original_short = db.session.query(Trade).filter(
                    Trade.symbol == symbol,
                    Trade.side == 'SHORT',
                    Trade.timestamp < datetime.now()
                ).order_by(Trade.timestamp.desc()).first()
                
                duration_seconds = None
                if original_short:
                    duration_seconds = int((datetime.now() - original_short.timestamp).total_seconds())
                
                # Update trade with P&L and duration
                trade.pnl = pnl
                trade.duration = duration_seconds
                
                # Update position
                new_quantity = portfolio_item.quantity + quantity  # Adding positive to negative
                if abs(new_quantity) < 0.001:  # Position fully closed
                    db.session.delete(portfolio_item)
                else:
                    portfolio_item.quantity = new_quantity
                    portfolio_item.current_price = price
                    portfolio_item.last_updated = datetime.now()
                
                # Update account
                account.cash_balance -= total_cost
                account.realized_pnl += pnl
                account.last_updated = datetime.now()
                
                db.session.commit()
                
                # Log trade execution to real-time data service
                self.realtime_data_service.log_trade_execution(trade)
                
                logger.info(f"Executed COVER order: {quantity} shares of {symbol} at ${price}, P&L: ${pnl:.2f}")
                return True
            else:
                logger.warning(f"No short position found to cover for {symbol}")
                return False
                
        except Exception as e:
            logger.error(f"Error executing cover order: {e}")
            db.session.rollback()
            return False
            
    def _update_portfolio_values(self):
        """Update current prices and portfolio values"""
        try:
            portfolio_items = db.session.query(Portfolio).all()
            symbols = [item.symbol for item in portfolio_items]
            
            if not symbols:
                return
                
            # Get current prices
            current_prices = self.market_data_service.get_multiple_quotes(symbols)
            
            total_market_value = 0
            total_unrealized_pnl = 0
            
            for item in portfolio_items:
                if item.symbol in current_prices and current_prices[item.symbol]:
                    item.current_price = current_prices[item.symbol]
                    item.last_updated = datetime.now()
                    
                    total_market_value += item.market_value
                    total_unrealized_pnl += item.unrealized_pnl
                    
            # Update account totals with corrected calculation
            account = self._get_account()
            
            # Calculate total realized P&L from all completed trades
            all_trades = db.session.query(Trade).all()
            realized_pnl = sum(t.pnl for t in all_trades if t.pnl is not None)
            
            # Correct account balance calculation: starting balance + realized P&L + unrealized P&L
            starting_balance = 1000.0
            corrected_total_value = starting_balance + realized_pnl + total_unrealized_pnl
            
            # Update account with corrected values
            account.cash_balance = starting_balance + realized_pnl - total_market_value  # Available cash
            account.total_value = corrected_total_value
            account.realized_pnl = realized_pnl
            account.unrealized_pnl = total_unrealized_pnl
            account.last_updated = datetime.now()
            
            db.session.commit()
            
        except Exception as e:
            logger.error(f"Error updating portfolio values: {e}")
            
    def _get_account(self) -> Account:
        """Get or create account record"""
        account = db.session.query(Account).first()
        if not account:
            account = Account()
            db.session.add(account)
            db.session.commit()
        return account
        
    def execute_manual_trade(self, symbol: str, side: str, quantity: float, price: float = None):
        """Execute a manual trade"""
        try:
            if price is None:
                price = self.market_data_service.get_real_time_price(symbol)
                if not price:
                    return False, "Could not get current price"
                    
            side_upper = side.upper()
            if side_upper == 'BUY':
                success = self._execute_buy_order(symbol, quantity, price, 'MANUAL')
                return success, "Buy order executed" if success else "Buy order failed"
            elif side_upper == 'SELL':
                success = self._execute_sell_order(symbol, quantity, price, 'MANUAL')
                return success, "Sell order executed" if success else "Sell order failed"
            elif side_upper == 'SHORT':
                success = self._execute_short_order(symbol, quantity, price, 'MANUAL')
                return success, "Short order executed" if success else "Short order failed"
            elif side_upper == 'COVER':
                success = self._execute_cover_order(symbol, quantity, price, 'MANUAL')
                return success, "Cover order executed" if success else "Cover order failed"
            else:
                return False, "Invalid side. Must be 'BUY', 'SELL', 'SHORT', or 'COVER'"
                
        except Exception as e:
            logger.error(f"Error executing manual trade: {e}")
            return False, f"Error: {str(e)}"
            
    def get_performance_metrics(self) -> Dict:
        """Calculate performance metrics"""
        try:
            account = self._get_account()
            trades = db.session.query(Trade).all()
            
            total_trades = len(trades)
            buy_trades = len([t for t in trades if t.side == 'BUY'])
            sell_trades = len([t for t in trades if t.side == 'SELL'])
            
            # Calculate win rate based on completed trades with P&L (simplified and accurate)
            completed_trades_with_pnl = [t for t in trades if t.pnl is not None and t.status == 'COMPLETED']
            profitable_trades = [t for t in completed_trades_with_pnl if t.pnl > 0]
            
            completed_pairs = len(completed_trades_with_pnl)
            profitable_pairs = len(profitable_trades)
            
            win_rate = (profitable_pairs / completed_pairs * 100) if completed_pairs > 0 else 0
            
            # Calculate returns based on €1,000 starting balance  
            starting_balance = 1000.0  # Starting paper trading balance in EUR
            
            # Calculate total realized P&L from all completed trades
            realized_pnl = sum(t.pnl for t in trades if t.pnl is not None)
            
            # Correct account balance should be starting balance + realized P&L
            corrected_total_value = starting_balance + realized_pnl + account.unrealized_pnl
            
            # Update account with corrected values
            account.realized_pnl = realized_pnl
            account.total_value = corrected_total_value
            
            total_return = ((corrected_total_value - starting_balance) / starting_balance) * 100
            
            return {
                'total_value': account.total_value,
                'cash_balance': account.cash_balance,
                'realized_pnl': account.realized_pnl,
                'unrealized_pnl': account.unrealized_pnl,
                'total_return': total_return,
                'total_trades': total_trades,
                'buy_trades': buy_trades,
                'sell_trades': sell_trades,
                'win_rate': win_rate,
                'completed_pairs': completed_pairs,
                'is_running': self.is_running
            }
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            return {
                'total_value': 0,
                'cash_balance': 0,
                'realized_pnl': 0,
                'unrealized_pnl': 0,
                'total_return': 0,
                'total_trades': 0,
                'buy_trades': 0,
                'sell_trades': 0,
                'win_rate': 0,
                'is_running': False
            }
    
    def _log_advanced_signal(self, symbol: str, signal_data: Dict):
        """Log advanced trading signal to database"""
        try:
            components = signal_data.get('components', {})
            timeframe = components.get('timeframe', {})
            volume = components.get('volume', {})
            macd = components.get('macd', {})
            stoch_rsi = components.get('stochastic_rsi', {})
            sr = components.get('support_resistance', {})
            
            # Convert numpy types to native Python types for JSON serialization
            import json
            def convert_numpy_types(obj):
                if hasattr(obj, 'item'):  # numpy scalar
                    return obj.item()
                elif isinstance(obj, dict):
                    return {k: convert_numpy_types(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_types(v) for v in obj]
                else:
                    return obj
            
            clean_components = convert_numpy_types(components)
            
            signal = AdvancedSignal(
                symbol=symbol,
                signal_type=signal_data.get('signal', 'NEUTRAL'),
                confidence=float(signal_data.get('confidence', 0)),
                
                # Timeframe analysis
                timeframe_1h_trend=timeframe.get('timeframes', {}).get('1h', {}).get('trend'),
                timeframe_4h_trend=timeframe.get('timeframes', {}).get('4h', {}).get('trend'),
                timeframe_1d_trend=timeframe.get('timeframes', {}).get('1d', {}).get('trend'),
                timeframe_confluence=bool(timeframe.get('confluence', False)),
                
                # Volume analysis
                volume_signal=volume.get('signal'),
                volume_surge=bool(volume.get('volume_surge', False)),
                volume_trend=volume.get('volume_trend'),
                
                # MACD
                macd_signal=macd.get('signal'),
                macd_line=float(macd.get('macd_line')) if macd.get('macd_line') is not None else None,
                macd_signal_line=float(macd.get('signal_line')) if macd.get('signal_line') is not None else None,
                macd_histogram=float(macd.get('histogram')) if macd.get('histogram') is not None else None,
                
                # Stochastic RSI
                stoch_rsi_signal=stoch_rsi.get('signal'),
                stoch_rsi_value=float(stoch_rsi.get('stoch_rsi')) if stoch_rsi.get('stoch_rsi') is not None else None,
                stoch_rsi_condition=stoch_rsi.get('condition'),
                
                # Support/Resistance
                sr_context=sr.get('current_level'),
                nearest_support=float(sr.get('nearest_support')) if sr.get('nearest_support') is not None else None,
                nearest_resistance=float(sr.get('nearest_resistance')) if sr.get('nearest_resistance') is not None else None,
                
                reasoning=signal_data.get('reasoning'),
                components_json=clean_components
            )
            
            db.session.add(signal)
            db.session.commit()
            
        except Exception as e:
            logger.error(f"Error logging advanced signal: {e}")
    
    def _update_trailing_stops(self, symbol: str, current_price: float):
        """Update trailing stops and execute if triggered"""
        try:
            # Get active trailing stops for the symbol
            active_stops = db.session.query(TrailingStop).filter_by(
                symbol=symbol, is_active=True
            ).all()
            
            for stop in active_stops:
                # Update the stop using our advanced engine's trailing stop logic
                if symbol not in self.advanced_engine.trailing_stops.stops:
                    # Initialize trailing stop in the engine
                    self.advanced_engine.trailing_stops.set_stop(
                        symbol, stop.entry_price, stop.position_type
                    )
                
                # Update and check if stop is triggered
                stop_triggered = self.advanced_engine.trailing_stops.update_stop(symbol, current_price)
                
                if stop_triggered:
                    # Execute stop loss trade
                    portfolio_item = db.session.query(Portfolio).filter_by(symbol=symbol).first()
                    if portfolio_item and portfolio_item.quantity != 0:
                        if stop.position_type == 'LONG' and portfolio_item.quantity > 0:
                            # Close long position
                            self._execute_sell_order(
                                symbol, portfolio_item.quantity, current_price,
                                f"Trailing Stop Loss", 
                                {'signal_type': 'SELL', 'trailing_stop': True},
                                f"Trailing stop triggered at €{current_price:.2f}",
                                {'current_price': current_price, 'stop_price': stop.stop_price},
                                'Trailing Stop Loss'
                            )
                        elif stop.position_type == 'SHORT' and portfolio_item.quantity < 0:
                            # Cover short position
                            self._execute_cover_order(
                                symbol, abs(portfolio_item.quantity), current_price,
                                f"Trailing Stop Loss",
                                {'signal_type': 'COVER', 'trailing_stop': True},
                                f"Trailing stop triggered at €{current_price:.2f}",
                                {'current_price': current_price, 'stop_price': stop.stop_price},
                                'Trailing Stop Loss'
                            )
                    
                    # Deactivate the stop
                    stop.is_active = False
                    self.advanced_engine.trailing_stops.remove_stop(symbol)
                else:
                    # Update stop price in database
                    stop_info = self.advanced_engine.trailing_stops.get_stop_info(symbol)
                    if stop_info:
                        stop.stop_price = stop_info['stop_price']
                        if stop.position_type == 'LONG':
                            stop.highest_price = stop_info['highest_price']
                        else:
                            stop.lowest_price = stop_info['lowest_price']
                        stop.last_updated = datetime.now()
            
            db.session.commit()
            
        except Exception as e:
            logger.error(f"Error updating trailing stops: {e}")
    
    def _create_trailing_stop(self, symbol: str, entry_price: float, position_type: str):
        """Create a new trailing stop for a position"""
        try:
            # Set stop in the advanced engine
            self.advanced_engine.trailing_stops.set_stop(symbol, entry_price, position_type)
            stop_info = self.advanced_engine.trailing_stops.get_stop_info(symbol)
            
            # Create database record
            trailing_stop = TrailingStop(
                symbol=symbol,
                position_type=position_type,
                entry_price=entry_price,
                stop_price=stop_info['stop_price'],
                highest_price=stop_info.get('highest_price', entry_price) if position_type == 'LONG' else None,
                lowest_price=stop_info.get('lowest_price', entry_price) if position_type == 'SHORT' else None,
                trail_percent=2.0,  # 2% trailing
                is_active=True
            )
            
            db.session.add(trailing_stop)
            db.session.commit()
            
            logger.info(f"Created trailing stop for {position_type} position in {symbol} at €{entry_price}")
            
        except Exception as e:
            logger.error(f"Error creating trailing stop: {e}")
