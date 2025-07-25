#!/usr/bin/env python3
"""
Comprehensive Trading Bot Validation and Auto-Fix Script
Automatically detects and fixes calculation errors throughout the codebase.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app import app, db
from models import Trade, Portfolio, Account, athens_now
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class TradingBotValidator:
    def __init__(self):
        self.issues_found = []
        self.fixes_applied = []
        
    def validate_and_fix_all(self):
        """Run all validation checks and apply fixes"""
        print("🔍 Starting comprehensive trading bot validation...")
        
        with app.app_context():
            self.check_missing_pnl()
            self.check_account_balance_consistency()
            self.check_portfolio_status_consistency()
            self.check_win_rate_accuracy()
            self.check_trade_status_logic()
            
        self.generate_report()
        
    def check_missing_pnl(self):
        """Check for trades missing P&L calculations"""
        print("Checking for missing P&L calculations...")
        
        # Find COMPLETED trades without P&L
        completed_without_pnl = db.session.query(Trade).filter(
            Trade.status == 'COMPLETED',
            Trade.pnl.is_(None)
        ).count()
        
        if completed_without_pnl > 0:
            self.issues_found.append(f"Found {completed_without_pnl} completed trades without P&L")
            
            # Fix: Set P&L to 0 for completed trades without P&L (shouldn't happen)
            updated = db.session.query(Trade).filter(
                Trade.status == 'COMPLETED',
                Trade.pnl.is_(None)
            ).update({'pnl': 0})
            
            if updated > 0:
                db.session.commit()
                self.fixes_applied.append(f"Set P&L to 0 for {updated} completed trades")
        
        # Set P&L to 0 for all OPEN trades (they don't have realized P&L yet)
        open_without_pnl = db.session.query(Trade).filter(
            Trade.status == 'OPEN',
            Trade.pnl.is_(None)
        ).count()
        
        if open_without_pnl > 0:
            updated = db.session.query(Trade).filter(
                Trade.status == 'OPEN',
                Trade.pnl.is_(None)
            ).update({'pnl': 0})
            
            db.session.commit()
            self.fixes_applied.append(f"Set P&L to 0 for {updated} open trades")
            
    def check_account_balance_consistency(self):
        """Check account balance calculation consistency"""
        print("Checking account balance consistency...")
        
        account = db.session.query(Account).first()
        if not account:
            self.issues_found.append("No account record found")
            account = Account()
            db.session.add(account)
            db.session.commit()
            self.fixes_applied.append("Created missing account record")
            
        # Calculate expected values
        all_trades = db.session.query(Trade).all()
        realized_pnl = sum(t.pnl for t in all_trades if t.pnl is not None)
        starting_balance = 1000.0
        expected_total = starting_balance + realized_pnl
        
        # Check if account values are incorrect
        tolerance = 0.01  # 1 cent tolerance
        if abs(account.realized_pnl - realized_pnl) > tolerance:
            self.issues_found.append(f"Account realized P&L mismatch: {account.realized_pnl} vs expected {realized_pnl}")
            account.realized_pnl = realized_pnl
            self.fixes_applied.append("Corrected account realized P&L")
            
        if abs(account.total_value - expected_total) > tolerance:
            self.issues_found.append(f"Account total value mismatch: {account.total_value} vs expected {expected_total}")
            account.total_value = expected_total
            self.fixes_applied.append("Corrected account total value")
            
        db.session.commit()
        
    def check_portfolio_status_consistency(self):
        """Check portfolio positions match OPEN trades"""
        print("Checking portfolio-trade consistency...")
        
        # Get all OPEN trades
        open_trades = db.session.query(Trade).filter(Trade.status == 'OPEN').all()
        portfolio_items = db.session.query(Portfolio).all()
        
        # Group open trades by symbol
        open_positions = {}
        for trade in open_trades:
            if trade.symbol not in open_positions:
                open_positions[trade.symbol] = {'quantity': 0, 'cost_basis': 0, 'trades': []}
            
            if trade.side == 'BUY':
                open_positions[trade.symbol]['quantity'] += trade.quantity
                open_positions[trade.symbol]['cost_basis'] += trade.total_value
            elif trade.side == 'SHORT':
                open_positions[trade.symbol]['quantity'] -= trade.quantity  # Negative for short
                open_positions[trade.symbol]['cost_basis'] += trade.total_value
                
            open_positions[trade.symbol]['trades'].append(trade)
        
        # Check if portfolio matches open positions
        portfolio_symbols = {item.symbol for item in portfolio_items}
        expected_symbols = set(open_positions.keys())
        
        if portfolio_symbols != expected_symbols:
            self.issues_found.append("Portfolio positions don't match open trades")
            
            # Rebuild portfolio from open trades
            db.session.query(Portfolio).delete()
            
            for symbol, position in open_positions.items():
                if position['quantity'] != 0:  # Only create portfolio entries for non-zero positions
                    avg_price = position['cost_basis'] / abs(position['quantity'])
                    
                    portfolio_item = Portfolio(
                        symbol=symbol,
                        quantity=position['quantity'],
                        avg_price=avg_price,
                        current_price=avg_price  # Will be updated by market data
                    )
                    db.session.add(portfolio_item)
                    
            db.session.commit()
            self.fixes_applied.append("Rebuilt portfolio from open trades")
            
    def check_win_rate_accuracy(self):
        """Check win rate calculation accuracy"""
        print("Checking win rate calculation...")
        
        # Get completed trades with P&L
        completed_trades = db.session.query(Trade).filter(
            Trade.status == 'COMPLETED',
            Trade.pnl.isnot(None)
        ).all()
        
        if completed_trades:
            profitable = sum(1 for t in completed_trades if t.pnl > 0)
            total = len(completed_trades)
            calculated_win_rate = (profitable / total) * 100
            
            print(f"Win rate: {calculated_win_rate:.1f}% ({profitable}/{total} profitable)")
            
    def check_trade_status_logic(self):
        """Check trade status logic consistency"""
        print("Checking trade status logic...")
        
        # BUY trades should be OPEN until sold
        # SHORT trades should be OPEN until covered
        # SELL and COVER trades should be COMPLETED
        
        incorrect_statuses = []
        
        # SELL trades should always be COMPLETED
        sell_trades_open = db.session.query(Trade).filter(
            Trade.side == 'SELL',
            Trade.status == 'OPEN'
        ).all()
        
        if sell_trades_open:
            for trade in sell_trades_open:
                trade.status = 'COMPLETED'
                incorrect_statuses.append(f"SELL trade {trade.id}")
                
        # COVER trades should always be COMPLETED
        cover_trades_open = db.session.query(Trade).filter(
            Trade.side == 'COVER',
            Trade.status == 'OPEN'
        ).all()
        
        if cover_trades_open:
            for trade in cover_trades_open:
                trade.status = 'COMPLETED'
                incorrect_statuses.append(f"COVER trade {trade.id}")
                
        if incorrect_statuses:
            db.session.commit()
            self.fixes_applied.append(f"Fixed status for trades: {', '.join(incorrect_statuses)}")
            
    def generate_report(self):
        """Generate validation report"""
        print("\n" + "="*50)
        print("TRADING BOT VALIDATION REPORT")
        print("="*50)
        
        if not self.issues_found:
            print("✅ No issues found - all calculations are consistent!")
        else:
            print(f"🔍 Found {len(self.issues_found)} issues:")
            for issue in self.issues_found:
                print(f"  ❌ {issue}")
                
        if self.fixes_applied:
            print(f"\n🔧 Applied {len(self.fixes_applied)} fixes:")
            for fix in self.fixes_applied:
                print(f"  ✅ {fix}")
        else:
            print("\n✅ No fixes needed")
            
        print("\n" + "="*50)

if __name__ == "__main__":
    validator = TradingBotValidator()
    validator.validate_and_fix_all()