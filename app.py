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


#prediction route
@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()
    text = data["text"]

    # convert text to tfidf
    text_vector = vectorizer.transform([text])

    # predict sentiment
    prediction = model.predict(text_vector)[0]

    return jsonify({
        "sentiment": prediction
    })


#running the server 
if __name__ == "__main__":
    app.run(debug=True)


