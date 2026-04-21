# Sentiment Analysis for Twitter using tweepy

This project focuses on building a sentiment analysis system using machine learning. It analyzes textual data such as reviews or comments and classifies them as positive or negative. The project includes text preprocessing, feature extraction using TF-IDF, model training and evaluation to measure performance and im going to be using the sentiment140 dataset from kaggle with 1.6 million tweets.

# Tech stack we are going to be using 
Machine Learning: Python, scikit-learn, TF-IDF, Logistic Regression

Backend: Flask, Flask-CORS, Tweepy, joblib

Frontend: React, JavaScript, Chart.js, Axios/Fetch

# Project Workflow

1. User enters a keyword or hashtag on the website  
2. Frontend sends the request to the backend API  
3. Backend fetches related tweets using the Twitter API (tweepy)
4. Tweets are cleaned and converted into TF-IDF vectors  
5. Logistic Regression model predicts sentiment  
6. Results are sent back to the frontend  
7. Website displays sentiment distribution and sample tweets

# Why this project

This project demonstrates how machine learning models can be integrated into real-world web applications to analyze social media data and extract meaningful insights.

# Applications

1. **Brand and Product Monitoring** – Track public sentiment about companies and products.  
2. **Market and Trend Analysis** – Analyze reactions to trending topics and events.  
3. **Public Opinion Tracking** – Understand how people feel about social or political issues.  
4. **Decision Support** – Help organizations make data-driven decisions using social media sentiment.

# Future Works:

Multi-Class Sentiment Detection — Extend the current binary classification (positive/negative) to support multiple emotion categories such as neutral, angry, and fearful for more detailed sentiment insights.
Deep Learning Integration — Replace the Logistic Regression model with advanced transformer-based models such as BERT or RoBERTa to significantly improve prediction accuracy and contextual understanding.
Real-Time Tweet Streaming — Implement Tweepy's live streaming API to continuously monitor and analyze tweets as they are posted, with dynamically updating charts on the dashboard.
Sentiment Trend Analysis Over Time — Track how public sentiment around a keyword shifts over days or weeks and visualize it as a time-series graph to identify patterns and events that influence opinion.
Geographical Sentiment Mapping — Integrate location data from tweets to display sentiment distribution on an interactive world map, revealing regional differences in public opinion on any given topic.



