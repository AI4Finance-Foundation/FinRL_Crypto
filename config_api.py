'''
Binance API configuration.
Reads API keys from environment variables for security.
'''

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

API_KEY_BINANCE = os.getenv('API_KEY_BINANCE')
API_SECRET_BINANCE = os.getenv('API_SECRET_BINANCE')