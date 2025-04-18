# src/models/sentiment_analyzer.py
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from nltk.sentiment import SentimentIntensityAnalyzer # Using VADER for quality score
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
import nltk
import praw # Added praw import
import os # Added os import
from dotenv import load_dotenv # Added dotenv import
from datetime import datetime # Added datetime import

# Ensure NLTK data is available (add downloads if not already present elsewhere)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError: # Changed from nltk.downloader.DownloadError
    print("NLTK 'punkt' not found. Downloading...")
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError: # Changed from nltk.downloader.DownloadError
    print("NLTK 'stopwords' not found. Downloading...")
    nltk.download('stopwords')
try:
    nltk.data.find('sentiment/vader_lexicon')
except LookupError: # Changed from nltk.downloader.DownloadError
    print("NLTK 'vader_lexicon' not found. Downloading...")
    nltk.download('vader_lexicon')


load_dotenv() # Load environment variables

class EnhancedSentimentAnalyzer:
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer() # VADER for quality scoring
        # Note: TF-IDF and RandomForestClassifier are initialized but not used in the current flow
        # self.vectorizer = TfidfVectorizer(max_features=1000)
        # self.classifier = RandomForestClassifier(n_estimators=100)
        self.stop_words = set(stopwords.words('english'))
        self.quality_threshold = 0.3 # Adjusted threshold slightly
        # Initialize Reddit API client
        self.reddit = praw.Reddit(
            client_id=os.getenv('REDDIT_CLIENT_ID'),
            client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
            user_agent=os.getenv('REDDIT_USER_AGENT')
        )

    def preprocess_text(self, text):
        """Preprocess text for analysis"""
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text) # Keep only letters and whitespace
        tokens = word_tokenize(text)
        tokens = [word for word in tokens if word not in self.stop_words]
        return ' '.join(tokens)

    def calculate_quality_score(self, post_text, post_score):
        """Calculate quality score based on text and Reddit score."""
        score = 0
        if not isinstance(post_text, str): post_text = ""
        if not isinstance(post_score, (int, float)): post_score = 0

        # Length bonus
        if len(post_text) > 150: # Increased length requirement
            score += 0.3
        elif len(post_text) > 50:
             score += 0.1

        # Reddit score bonus
        if post_score > 20: # Increased score requirement
            score += 0.4
        elif post_score > 5:
            score += 0.2

        # Sentiment intensity bonus (using VADER)
        sentiment_scores = self.sia.polarity_scores(post_text)
        score += abs(sentiment_scores['compound']) * 0.3 # Reduced weight slightly

        return min(score, 1.0) # Cap score at 1.0

    def filter_low_quality_posts(self, posts_df):
        """Filter out low quality posts based on calculated score."""
        if posts_df.empty:
            return posts_df
        # Apply quality score calculation row-wise
        posts_df['quality_score'] = posts_df.apply(
            lambda row: self.calculate_quality_score(row.get('text', ''), row.get('score', 0)),
            axis=1
        )
        filtered_df = posts_df[posts_df['quality_score'] >= self.quality_threshold].copy() # Use copy to avoid SettingWithCopyWarning
        print(f"Original posts: {len(posts_df)}, Filtered posts (quality >= {self.quality_threshold}): {len(filtered_df)}")
        return filtered_df

    def analyze_sentiment_vader(self, text):
        """Basic sentiment analysis using VADER."""
        if not isinstance(text, str):
            return 0.0
        return self.sia.polarity_scores(text)['compound']

    def get_reddit_posts(self, stock_symbol, limit=150): # Increased limit slightly
        """Fetches posts from Reddit."""
        posts = []
        # Expanded subreddit list
        subreddits = 'stocks+investing+wallstreetbets+StockMarket+IndiaInvestments+IndianStockMarket'
        search_query = f'{stock_symbol}' # Simplified query, rely on subreddits
        print(f"Searching Reddit ({subreddits}) for '{search_query}'...")
        try:
            for post in self.reddit.subreddit(subreddits).search(
                search_query, limit=limit, time_filter='month', sort='relevance' # Added sort
            ):
                posts.append({
                    'title': post.title,
                    'text': post.selftext,
                    'score': post.score,
                    'created_utc': datetime.fromtimestamp(post.created_utc), # Store as datetime object
                    'url': f'https://reddit.com{post.permalink}',
                    'subreddit': post.subreddit.display_name
                })
        except Exception as e:
            print(f"Error fetching Reddit posts: {str(e)}")
        print(f"Fetched {len(posts)} posts initially.")
        return pd.DataFrame(posts)

    def analyze(self, stock_symbol):
        """Main analysis method: fetch, filter, analyze sentiment, aggregate."""
        posts_df = self.get_reddit_posts(stock_symbol)

        if posts_df.empty:
            return {
                'success': False,
                'error': f'No Reddit posts found for {stock_symbol}',
                'daily_sentiment': pd.Series(dtype=float) # Return empty series
            }

        # Filter low-quality posts
        filtered_posts_df = self.filter_low_quality_posts(posts_df)

        if filtered_posts_df.empty:
            return {
                'success': False,
                'error': f'No high-quality Reddit posts found for {stock_symbol} after filtering',
                 'daily_sentiment': pd.Series(dtype=float) # Return empty series
            }

        # Analyze sentiment of filtered posts using VADER
        # Combine title and text for sentiment analysis
        filtered_posts_df['full_text'] = filtered_posts_df['title'].fillna('') + " " + filtered_posts_df['text'].fillna('')
        filtered_posts_df['sentiment'] = filtered_posts_df['full_text'].apply(self.analyze_sentiment_vader)

        # Calculate daily sentiment from filtered posts
        try:
            # Ensure index is datetime
            if not pd.api.types.is_datetime64_any_dtype(filtered_posts_df['created_utc']):
                 filtered_posts_df['date'] = pd.to_datetime(filtered_posts_df['created_utc'])
            else:
                 filtered_posts_df['date'] = filtered_posts_df['created_utc']

            filtered_posts_df.set_index('date', inplace=True)
            daily_sentiment = filtered_posts_df['sentiment'].resample('D').mean().fillna(0)
        except Exception as e:
            print(f"Error calculating daily sentiment: {e}")
            daily_sentiment = pd.Series(dtype=float) # Return empty series on error

        # Aggregate results
        avg_sentiment = filtered_posts_df['sentiment'].mean()
        filtered_posts_df['sentiment_category'] = filtered_posts_df['sentiment'].apply(
            lambda x: 'positive' if x > 0.05 else ('negative' if x < -0.05 else 'neutral') # Standard VADER thresholds
        )
        sentiment_counts = filtered_posts_df['sentiment_category'].value_counts().to_dict()

        # Get top posts based on quality score *then* Reddit score from the filtered list
        top_posts = filtered_posts_df.nlargest(5, ['quality_score', 'score']).to_dict('records')
        # Convert datetime objects in top_posts to strings for JSON serialization
        for post in top_posts:
            if 'created_utc' in post and isinstance(post['created_utc'], datetime):
                post['created_utc'] = post['created_utc'].strftime('%Y-%m-%d %H:%M:%S')
            # Remove non-serializable columns if they exist from resampling/indexing
            post.pop('date', None)
            post.pop('full_text', None)


        return {
            'success': True,
            'average_sentiment': float(avg_sentiment) if pd.notna(avg_sentiment) else 0.0,
            'post_count': len(filtered_posts_df), # Count of high-quality posts
            'sentiment_distribution': sentiment_counts,
            'top_posts': top_posts,
            'daily_sentiment': daily_sentiment # Keep as Series for plotting in app.py
        }

    # Removed analyze_trend as daily_sentiment is calculated in analyze()
    # Removed unused TF-IDF/RandomForest methods for now