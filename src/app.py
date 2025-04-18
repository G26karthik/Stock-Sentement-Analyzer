# src/app.py
from flask import Flask, render_template, request, jsonify, url_for
from models.sentiment_analyzer import EnhancedSentimentAnalyzer
from visualization.plotter import SentimentPlotter
import os
import pandas as pd
from dotenv import load_dotenv
import traceback
import time

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Ensure static directory exists
if not os.path.exists('static'):
    os.makedirs('static')

# Initialize analyzers
sentiment_analyzer = EnhancedSentimentAnalyzer()
plotter = SentimentPlotter()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    plot_path_url = None
    try:
        # Get stock symbol from form (country removed)
        stock_symbol = request.form.get('stock_symbol', '').strip().upper()

        if not stock_symbol:
             return jsonify({'error': 'Stock symbol cannot be empty.'}), 400

        # Get Reddit sentiment using the enhanced analyzer
        sentiment_data = sentiment_analyzer.analyze(stock_symbol)

        # Generate sentiment trend plot if sentiment analysis was successful
        if sentiment_data.get('success') and isinstance(sentiment_data.get('daily_sentiment'), pd.Series) and not sentiment_data['daily_sentiment'].empty:
            daily_sentiment_series = sentiment_data['daily_sentiment']
            stock_plot_data = None # No stock data to plot

            try:
                sentiment_trend_fig = plotter.plot_sentiment_trend(
                    daily_sentiment_series,
                    stock_plot_data, # Pass None for stock data
                    title=f"Sentiment Trend for {stock_symbol}"
                )
                timestamp = int(time.time())
                relative_plot_path = f'static/{stock_symbol}_trend_{timestamp}.png'
                plotter.save_plot(sentiment_trend_fig, relative_plot_path)
                plot_path_url = url_for('static', filename=os.path.basename(relative_plot_path))
                print(f"Plot saved to {relative_plot_path}, URL: {plot_path_url}")

                # Convert Series to JSON serializable format *after* plotting
                sentiment_data['daily_sentiment'] = [[ts.strftime('%Y-%m-%d'), val] for ts, val in daily_sentiment_series.items()]

            except Exception as plot_err:
                 print(f"Error generating or saving plot: {plot_err}")
                 traceback.print_exc() # Print detailed traceback for plotting error
                 plot_path_url = None # Ensure plot path is None if plotting fails
                 # Keep daily_sentiment as Series if plotting fails, convert later if needed or handle error
                 if isinstance(sentiment_data.get('daily_sentiment'), pd.Series):
                     # Attempt conversion anyway for consistency, or handle differently
                     try:
                         daily_sentiment_series = sentiment_data['daily_sentiment']
                         sentiment_data['daily_sentiment'] = [[ts.strftime('%Y-%m-%d'), val] for ts, val in daily_sentiment_series.items()]
                     except Exception as conv_err:
                          print(f"Error converting daily sentiment after plot error: {conv_err}")
                          sentiment_data['daily_sentiment'] = [] # Fallback to empty list


        elif sentiment_data.get('success') and not isinstance(sentiment_data.get('daily_sentiment'), pd.Series):
             # Handle case where daily_sentiment might not be a Series (e.g., already converted or error)
             print("Warning: daily_sentiment was not a Pandas Series when expected for plotting.")
             # Ensure it's in a serializable format if it exists
             if 'daily_sentiment' not in sentiment_data or sentiment_data['daily_sentiment'] is None:
                  sentiment_data['daily_sentiment'] = []

        # Combine the data for JSON response (removed stock_data and country)
        result = {
            'stock_symbol': stock_symbol,
            'sentiment': sentiment_data,
            'trend_plot_url': plot_path_url
        }

        return jsonify(result)

    except Exception as e:
        print(f"Error in /analyze endpoint: {str(e)}")
        traceback.print_exc() # Print detailed traceback for server error
        return jsonify({'error': f'An internal server error occurred: {str(e)}'}), 500

if __name__ == '__main__':
    # Make sure host is set to '0.0.0.0' to be accessible externally if needed,
    # otherwise default '127.0.0.1' is fine for local use.
    app.run(debug=True, host='0.0.0.0')