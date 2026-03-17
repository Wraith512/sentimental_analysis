# SENTIMENTAL ANALYSIS (using tweepy)


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.model_selection  import train_test_split
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import re

df = pd.read_csv(
    r"C:\Users\ASUS\Desktop\archive\training.1600000.processed.noemoticon.csv",
    header=None
)

df.columns = ["sentiment", "id", "date", "query", "user", "text"]


df.head()


df.shape()


df = df[["sentiment", "text"]]

df.head()

df["sentiment"] = df["sentiment"].replace({0: "negative", 4: "positive"})

print(df.isnull().sum())
print("Duplicates:", df.duplicated().sum())

df = df.dropna()
df = df.drop_duplicates()


def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["clean_text"] = df["text"].apply(clean_text)

df.head()

X = df["clean_text"]
y = df["sentiment"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1,2),
    sublinear_tf = True,
    min_df = 3,
    strip_accents ='unicode',
    analyzer ='word')

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


model = LogisticRegression(max_iter=200)

model.fit(X_train_tfidf, y_train)

y_pred = model.predict(X_test_tfidf)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))


joblib.dump(model, "sentiment_analysis_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")


model = joblib.load("sentiment_analysis_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")








