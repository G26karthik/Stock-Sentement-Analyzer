# Reddit Stock Sentiment Analyzer

A Flask web application that analyzes recent Reddit discussions for a given stock symbol to gauge public sentiment. It fetches posts, performs sentiment analysis using NLP, filters for quality discussions, and visualizes the sentiment trend.

## Features

*   **Reddit Data Fetching:** Retrieves recent posts mentioning a stock symbol from relevant subreddits (e.g., r/stocks, r/wallstreetbets, r/investing).
*   **Sentiment Analysis:** Uses NLTK's VADER to calculate sentiment scores for fetched posts.
*   **Quality Filtering:** Implements a scoring mechanism to filter out low-quality or irrelevant posts based on length, Reddit score, and sentiment intensity.
*   **Trend Visualization:** Generates and displays a plot showing the daily average sentiment trend using Matplotlib.
*   **Web Interface:** Provides a clean, responsive, dark-mode UI built with Flask, HTML, CSS, and JavaScript for easy interaction.

## Tech Stack

*   **Backend:** Python, Flask
*   **Data Fetching:** Praw (for Reddit API)
*   **NLP:** NLTK (VADER for sentiment)
*   **Data Handling:** Pandas
*   **Visualization:** Matplotlib
*   **Frontend:** HTML, CSS, JavaScript

## Setup

1.  **Clone the repository (if you haven't already):**
    ```bash
    git clone <your-repository-url>
    cd stock # Navigate into the project directory
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # Activate it:
    # Windows:
    .\venv\Scripts\Activate.ps1
    # macOS/Linux:
    # source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Create a `.env` file** in the project root (`stock/`) with your Reddit API credentials:
    ```dotenv
    REDDIT_CLIENT_ID=your_client_id
    REDDIT_CLIENT_SECRET=your_client_secret
    REDDIT_USER_AGENT=your_user_agent_string (e.g., MyStockAnalyzer/0.1 by u/YourUsername)
    ```
    *(You can get credentials by creating a 'script' app on Reddit: [https://www.reddit.com/prefs/apps](https://www.reddit.com/prefs/apps))*

## Usage

1.  **Navigate to the `src` directory:**
    ```bash
    cd src
    ```

2.  **Run the Flask application:**
    ```bash
    flask run --host=0.0.0.0
    ```
    *(The `--host=0.0.0.0` makes it accessible on your local network; use `127.0.0.1` or omit for local access only)*

3.  **Open your web browser** and go to `http://127.0.0.1:5000` (or your machine's IP address if using `0.0.0.0`).

4.  **Enter a stock symbol** (e.g., AAPL, TSLA, GME) and click "Analyze".

5.  View the sentiment analysis results, trend plot, and top relevant posts.

## Project Structure
