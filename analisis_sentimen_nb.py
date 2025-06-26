
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from wordcloud import WordCloud

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

# Load dataset
df = pd.read_csv("Tweets.csv")
df = df[['airline_sentiment', 'text']].dropna()
df = df[df['airline_sentiment'].isin(['positive', 'neutral', 'negative'])]

# Preprocessing teks
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+|#[A-Za-z0-9]+", "", text)
    text = re.sub(r"[^a-zA-Z]", " ", text)
    tokens = text.lower().split()
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

df['clean_text'] = df['text'].apply(clean_text)

# TF-IDF dan split data
tfidf = TfidfVectorizer(max_features=3000)
X = tfidf.fit_transform(df['clean_text'])
y = df['airline_sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model Naive Bayes
model = MultinomialNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluasi model
print("Classification Report:\n")
print(classification_report(y_test, y_pred))

# Confusion matrix
plt.figure(figsize=(8,6))
cm = confusion_matrix(y_test, y_pred, labels=["positive", "neutral", "negative"])
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=["positive", "neutral", "negative"],
            yticklabels=["positive", "neutral", "negative"])
plt.title("Confusion Matrix - Twitter Sentiment Naive Bayes")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Wordcloud untuk tiap sentimen
for sentiment in ['positive', 'neutral', 'negative']:
    text = " ".join(df[df['airline_sentiment'] == sentiment]['clean_text'])
    wc = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(8,4))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'WordCloud for {sentiment} tweets')
    plt.show()
