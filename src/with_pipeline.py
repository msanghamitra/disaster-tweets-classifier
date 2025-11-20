# Step 1: Necessary Libraries
import re
import string

import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Download once (comment out after first successful run)
nltk.download('punkt')
nltk.download('stopwords')


# -------------------------------------------------------------------
# Text cleaning function
# -------------------------------------------------------------------
def clean_text(text: str) -> str:
    """
    Basic tweet cleaning:
    - lowercasing
    - remove URLs, mentions, hashtag symbol
    - remove punctuation and digits
    - collapse multiple spaces
    """
    text = str(text)
    text = text.lower()

    # remove text in brackets
    text = re.sub(r'\[.*?\]', '', text)
    # remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    # remove mentions
    text = re.sub(r'@\w+', '', text)
    # remove the '#' symbol (keep the word)
    text = re.sub(r'#', '', text)
    # remove punctuation
    text = re.sub(r'[%s]' % re.escape(string.punctuation), ' ', text)
    # remove digits
    text = re.sub(r'\d+', '', text)
    # collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


# -------------------------------------------------------------------
# Stopword Builder
# -------------------------------------------------------------------
def build_stopwords(custom_words=None):
    """
    Combine NLTK stopwords with any additional user-provided stopwords.
    """
    sw = set(stopwords.words("english"))
    
    if custom_words:
        sw.update(w.lower() for w in custom_words)
    
    return list(sw)


# -------------------------------------------------------------------
# Build pipeline
# -------------------------------------------------------------------
custom_stopwords = ["rt", "amp", "via"]    # Add any words you want
stoplist = build_stopwords(custom_stopwords)

disaster_pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        max_features=1500,
        min_df=2,
        max_df=0.5,
        ngram_range=(1, 2),
        preprocessor=clean_text,
        stop_words=stoplist
    )),
    ("clf", LogisticRegression(max_iter=1000))
])


# -------------------------------------------------------------------
# Load data and split into train / test
# -------------------------------------------------------------------
data_path = "/Users/sanghamitramatta/Documents/All Women Bootcamp/Tweet prediction/Data/tweet_disasters.csv"   # <--- your input file
df = pd.read_csv(data_path)

X = df["text"]
y = df["target"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# -------------------------------------------------------------------
# Train the model
# -------------------------------------------------------------------
disaster_pipeline.fit(X_train, y_train)


# -------------------------------------------------------------------
# Evaluate on test data
# -------------------------------------------------------------------
y_test_pred = disaster_pipeline.predict(X_test)

print("=== Test Set Evaluation ===")
print(classification_report(y_test, y_test_pred, digits=4))


# -------------------------------------------------------------------
# Function to predict on a new dataset
# -------------------------------------------------------------------
def predict_disaster_tweets(model, df, text_col="text"):
    """
    Use the trained pipeline to predict on a new dataset.
    Returns a DataFrame with the original text and the predicted label.
    """
    if text_col not in df.columns:
        raise ValueError(f"Column '{text_col}' not found in dataframe")

    X_new = df[text_col]
    preds = model.predict(X_new)

    result = df[[text_col]].copy()
    result["predicted_target"] = preds  # 1 = disaster, 0 = not disaster

    return result


# -------------------------------------------------------------------
# Example: predict on a separate prediction file (optional)
# -------------------------------------------------------------------
pred_path = "/Users/sanghamitramatta/Documents/All Women Bootcamp/Tweet prediction/Data/tweet_disasters_pred.csv"
pred_df = pd.read_csv(pred_path)
pred_results = predict_disaster_tweets(disaster_pipeline, pred_df, text_col="text")
print(pred_results.head())
