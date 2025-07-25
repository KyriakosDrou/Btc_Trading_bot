#!/usr/bin/env python3
"""
Standalone script to run the trading bot continuously.
This ensures the bot runs even when the web server is idle.
"""

import sys
import os
import logging
import time
from threading import Thread

# Add the project directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def run_standalone_bot():
    """Run the trading bot in standalone mode"""
    try:
        # Initialize Flask app context
        from app import app, db
        from trading_bot import TradingBot
        
        with app.app_context():
            # Create database tables if they don't exist
            db.create_all()
            
            # Initialize and start the bot
            bot = TradingBot()
            bot.start()
            
            logger.info("Standalone trading bot started successfully")
            
            # Keep the bot running
            while True:
                try:
                    if not bot.is_running:
                        logger.warning("Bot stopped unexpectedly, restarting...")
                        bot.start()
                    
                    time.sleep(60)  # Check every minute
                    
                except KeyboardInterrupt:
                    logger.info("Stopping bot...")
                    bot.stop()
                    break
                except Exception as e:
                    logger.error(f"Bot monitoring error: {e}")
                    time.sleep(30)
                    
    except Exception as e:
        logger.error(f"Failed to start standalone bot: {e}")
        sys.exit(1)

if __name__ == "__main__":
    logger.info("Starting standalone Bitcoin trading bot")
    run_standalone_bot()