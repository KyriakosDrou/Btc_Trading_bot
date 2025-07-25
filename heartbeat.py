#!/usr/bin/env python3
"""
Heartbeat script to keep the trading bot alive continuously.
This script runs independently of the Flask web server.
"""

import requests
import time
import logging
import os
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def ping_bot():
    """Ping the bot to keep it alive"""
    try:
        # Get bot status
        response = requests.get('http://localhost:5000/api/bot_status', timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            is_running = data.get('is_running', False)
            
            if not is_running:
                logger.info("Bot is not running, attempting to start...")
                # Start the bot
                start_response = requests.post('http://localhost:5000/api/start_bot', timeout=10)
                if start_response.status_code == 200:
                    logger.info("Bot started successfully")
                else:
                    logger.error(f"Failed to start bot: {start_response.status_code}")
            else:
                logger.info("Bot is running normally")
                
        else:
            logger.error(f"Failed to check bot status: {response.status_code}")
            
    except requests.RequestException as e:
        logger.error(f"Request error: {e}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")

def main():
    """Main heartbeat loop"""
    logger.info("Starting Bitcoin trading bot heartbeat service")
    
    while True:
        try:
            ping_bot()
            time.sleep(120)  # Check every 2 minutes
            
        except KeyboardInterrupt:
            logger.info("Heartbeat service stopped by user")
            break
        except Exception as e:
            logger.error(f"Heartbeat error: {e}")
            time.sleep(60)  # Wait 1 minute on error

if __name__ == "__main__":
    main()