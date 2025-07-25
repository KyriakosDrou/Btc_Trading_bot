import os
import logging
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import DeclarativeBase
from werkzeug.middleware.proxy_fix import ProxyFix
from threading import Thread
import time
import logging
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
import atexit

# Configure logging
logging.basicConfig(level=logging.DEBUG)

class Base(DeclarativeBase):
    pass

db = SQLAlchemy(model_class=Base)

# Create the app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "trading-bot-secret-key")
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

# Configure the database
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL", "sqlite:///trading_bot.db")
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "pool_recycle": 300,
    "pool_pre_ping": True,
}

# Initialize the app with the extension
db.init_app(app)

# Initialize scheduler
scheduler = BackgroundScheduler()
scheduler.start()
atexit.register(lambda: scheduler.shutdown())

with app.app_context():
    # Import models and routes
    import models  # noqa: F401
    import routes  # noqa: F401
    
    # Create all tables
    db.create_all()
    
    # Initialize trading bot
    from trading_bot import TradingBot
    bot = TradingBot()
    
    # Add bot to app context
    app.bot = bot
    app.scheduler = scheduler
    
    # Create continuous background task for bot
    def keep_bot_alive():
        """Keep trading bot running continuously"""
        try:
            with app.app_context():
                # Auto-start bot if not running
                if not bot.is_running:
                    logging.info("Auto-starting trading bot for continuous operation")
                    bot.start()
                    
                # Heartbeat to keep bot active
                logging.debug("Trading bot heartbeat - ensuring continuous operation")
        except Exception as e:
            logging.error(f"Keep-alive error: {e}")
    
    # Schedule the keep-alive task to run every 60 seconds
    scheduler.add_job(
        func=keep_bot_alive,
        trigger=IntervalTrigger(seconds=60),
        id='keep_bot_alive',
        name='Keep Trading Bot Alive',
        replace_existing=True
    )
    
    # Start bot immediately
    bot.start()
    logging.info("Trading bot initialized with continuous background operation")
