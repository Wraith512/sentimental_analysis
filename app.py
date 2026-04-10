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
