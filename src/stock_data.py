# src/stock_data.py
import yfinance as yf
import time
from cachetools import TTLCache
# Removed unused HTTPError import

# Define common exchange suffixes
EXCHANGE_SUFFIX = {
    "US": "",    # USA (Default)
    "IN": ".NS", # India (National Stock Exchange)
    "UK": ".L",  # UK (London Stock Exchange)
    "JP": ".T",  # Japan (Tokyo)
    "CN": ".SS", # China (Shanghai) - Note: Yahoo Finance coverage for CN might be limited/delayed
    "DE": ".DE", # Germany (XETRA)
    "CA": ".TO", # Canada (Toronto)
    # Add more as needed
}

class StockDataFetcher:
    def __init__(self):
        # Cache stock data (TTL: 30 minutes) - Reduced TTL slightly
        self.cache = TTLCache(maxsize=100, ttl=1800)

    def get_stock_data(self, stock_symbol, country='US', max_retries=3, retry_delay=15): # Reduced delay slightly
        """Fetches stock data, appending exchange suffix based on country."""
        suffix = EXCHANGE_SUFFIX.get(country.upper(), "") # Get suffix, default to ""
        ticker_symbol = f"{stock_symbol.upper()}{suffix}"
        cache_key = ticker_symbol # Use the full ticker symbol as cache key

        if cache_key in self.cache:
            print(f"Cache hit for {cache_key}")
            return self.cache[cache_key]
        else:
             print(f"Cache miss for {cache_key}. Fetching from yfinance...")


        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    wait_time = retry_delay * (2 ** attempt) # Exponential backoff
                    print(f"Rate limit likely hit. Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)

                stock = yf.Ticker(ticker_symbol)
                stock_info = stock.info

                # Check if data was actually retrieved
                if not stock_info or stock_info.get('quoteType') == 'MUTUALFUND': # yfinance often returns minimal dict for errors
                     # Attempt history as a fallback check for existence
                     hist = stock.history(period="1d")
                     if hist.empty:
                         raise ValueError(f"No data found for ticker symbol: {ticker_symbol}. It might be delisted or incorrect.")

                # Prioritize 'currentPrice', fallback to 'regularMarketPrice' or 'previousClose'
                current_price = stock_info.get('currentPrice') or \
                                stock_info.get('regularMarketPrice') or \
                                stock_info.get('previousClose')


                if current_price is None:
                    # If still no price, raise error
                    raise ValueError(f"Could not get current price for {ticker_symbol}")

                data = {
                    'success': True,
                    'data': {
                        'ticker': ticker_symbol,
                        'current_price': float(current_price),
                        'currency': stock_info.get('currency', 'USD'),
                        'name': stock_info.get('shortName', ticker_symbol), # Add company name
                        'exchange': stock_info.get('exchange', 'N/A') # Add exchange info
                    }
                }
                self.cache[cache_key] = data # Cache the successful result
                return data

            except Exception as e:
                print(f"Error fetching stock data for {ticker_symbol} (attempt {attempt + 1}/{max_retries}): {str(e)}")
                if attempt == max_retries - 1:
                    return {
                        'success': False,
                        'error': f"Failed to fetch stock data for {ticker_symbol} after {max_retries} attempts: {str(e)}"
                    }
                # Continue to next retry attempt implicitly