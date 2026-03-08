#importing the necessary libraries 
from flask import Flask, request, jsonify
import joblib

#creating the flask app
app = Flask(__name__)

#loading the saved model and the tf-idf vectorizr
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")


#home route
@app.route("/")
def home():
    return "Twitter Sentiment Analysis API is running"


