#importing the necessary libraries 
from flask import Flask, request, jsonify, render_template
import joblib
import re
import tweepy
import os

#creating the flask app
app = Flask(__name__)

#loading the saved model and the tf-idf vectorizer
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Twitter API credentials - Replace with your own keys
# Get these from https://developer.twitter.com/en/portal/dashboard
TWITTER_API_KEY = os.environ.get("TWITTER_API_KEY", "YOUR_API_KEY")
TWITTER_API_SECRET = os.environ.get("TWITTER_API_SECRET", "YOUR_API_SECRET")
TWITTER_ACCESS_TOKEN = os.environ.get("TWITTER_ACCESS_TOKEN", "YOUR_ACCESS_TOKEN")
TWITTER_ACCESS_SECRET = os.environ.get("TWITTER_ACCESS_SECRET", "YOUR_ACCESS_SECRET")
TWITTER_BEARER_TOKEN = os.environ.get("TWITTER_BEARER_TOKEN", "YOUR_BEARER_TOKEN")

# Initialize Twitter client (API v2)
def get_twitter_client():
    try:
        client = tweepy.Client(
            bearer_token=TWITTER_BEARER_TOKEN,
            consumer_key=TWITTER_API_KEY,
            consumer_secret=TWITTER_API_SECRET,
            access_token=TWITTER_ACCESS_TOKEN,
            access_token_secret=TWITTER_ACCESS_SECRET,
            wait_on_rate_limit=True
        )
        return client
    except Exception as e:
        print(f"Error initializing Twitter client: {e}")
        return None

#function to clean text (same as training)
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def analyze_sentiment(text):
    """Analyze sentiment of a single text"""
    cleaned_text = clean_text(text)
    text_vector = vectorizer.transform([cleaned_text])
    prediction = model.predict(text_vector)[0]
    confidence = max(model.predict_proba(text_vector)[0])
    return {
        "sentiment": prediction,
        "confidence": round(confidence * 100, 2)
    }


#home route - serves the frontend
@app.route("/")
def home():
    return render_template("index.html")


#prediction route
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text", "")
    
    if not text.strip():
        return jsonify({"error": "Please enter some text"}), 400
    
    # clean the text
    cleaned_text = clean_text(text)
    
    # convert text to tfidf
    text_vector = vectorizer.transform([cleaned_text])

    # predict sentiment
    prediction = model.predict(text_vector)[0]
    
    # get confidence score
    confidence = max(model.predict_proba(text_vector)[0])

    return jsonify({
        "sentiment": prediction,
        "confidence": round(confidence * 100, 2),
        "original_text": text,
        "cleaned_text": cleaned_text
    })
    #search tweets route - fetch real-time tweets
@app.route("/search_tweets", methods=["POST"])
def search_tweets():
    data = request.get_json()
    query = data.get("query", "")
    count = min(data.get("count", 10), 100)  # Max 100 tweets
    
    if not query.strip():
        return jsonify({"error": "Please enter a search query"}), 400
    
    client = get_twitter_client()
    if not client:
        return jsonify({"error": "Twitter API not configured. Please add your API credentials."}), 500
    
    try:
        # Search recent tweets (last 7 days) using Twitter API v2
        # Exclude retweets and replies for cleaner results
        search_query = f"{query} -is:retweet -is:reply lang:en"
        
        tweets = client.search_recent_tweets(
            query=search_query,
            max_results=count,
            tweet_fields=["created_at", "author_id", "public_metrics"],
            user_fields=["username", "name"],
            expansions=["author_id"]
        )
        if not tweets.data:
            return jsonify({"tweets": [], "summary": {"total": 0, "positive": 0, "negative": 0}})
        
        # Create user lookup dictionary
        users = {user.id: user for user in (tweets.includes.get("users", []) if tweets.includes else [])}
        
        results = []
        positive_count = 0
        negative_count = 0
        
        for tweet in tweets.data:
            sentiment_result = analyze_sentiment(tweet.text)
            
            if sentiment_result["sentiment"] == "positive":
                positive_count += 1
            else:
                negative_count += 1
            
            user = users.get(tweet.author_id, None)
            
            results.append({
                "id": str(tweet.id),
                "text": tweet.text,
                "username": user.username if user else "unknown",
                "name": user.name if user else "Unknown",
                "created_at": tweet.created_at.isoformat() if tweet.created_at else None,
                "likes": tweet.public_metrics.get("like_count", 0) if tweet.public_metrics else 0,
                "retweets": tweet.public_metrics.get("retweet_count", 0) if tweet.public_metrics else 0,
                "sentiment": sentiment_result["sentiment"],
                "confidence": sentiment_result["confidence"]
            })
        total = len(results)
        summary = {
            "total": total,
            "positive": positive_count,
            "negative": negative_count,
            "positive_percent": round((positive_count / total) * 100, 1) if total > 0 else 0,
            "negative_percent": round((negative_count / total) * 100, 1) if total > 0 else 0
        }
        
        return jsonify({"tweets": results, "summary": summary, "query": query})
