# %% [markdown]
# SENTIMENTAL ANALYSIS (using tweepy)
# 

# %%
#this is the code for sentimental analysis for twitter 
#using twitter api for realtime analysis (tweepy)

# %%
#importing the necesssary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.model_selection  import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import re

##basic eda


# %%
#importing the dataset 
df = pd.read_csv(
    r"C:\Users\ASUS\Desktop\archive\training.1600000.processed.noemoticon.csv",
    header=None
)

df.columns = ["sentiment", "id", "date", "query", "user", "text"]

# %%
df.head()

# %%
df.shape()

# %%
df = df[["sentiment", "text"]]

df.head()

#converting the sentiment nummbers to labels from 0-negative and 4-postive 
df["sentiment"] = df["sentiment"].replace({0: "negative", 4: "positive"})

#checking missing and duplicate values
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

#defining the features and the labels 
X = df["clean_text"]
y = df["sentiment"]

#splitting the data into train and test 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#tf-idf vectorizer to convert the words to numbers without loosing the semantic valie 
vectorizer = TfidfVectorizer(max_features=5000)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

#training the model 
model = LogisticRegression(max_iter=200)

model.fit(X_train_tfidf, y_train)

#printing the model accuracy
y_pred = model.predict(X_test_tfidf)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

#saving the model
joblib.dump(model, "sentiment_analysis_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

#loading the model for further use 
model = joblib.load("sentiment_analysis_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")







