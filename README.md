 

# Disaster Tweet Classification

This project predicts whether a tweet describes a real disaster or not using a NLP pipeline (TF–IDF + Logistic Regression).


Dataset
The datasets contains a set of tweets which have been divided into a training and a pred set. 
The training set contains a target column identifying whether the tweet pertains to a real disaster or not.
Our job is to create a ML model to predict whether the pred set tweets belong to a disaster or not, in the form of 1 or 0.
This is a classic case of a Binary Classification problem.


## Workflow 

RAW DATA → EDA (manual analysis) → Build pipeline → Train → Save model → Predict

---

## Project Structure

```text
tweet-prediction/
├── README.md
├── requirements.txt
│
├── data/
│   ├── raw/
│   │   ├── train.csv
│   │   └── pred.csv
│   └── processed/
│       └── predictions.csv
│
├── notebooks/
│   └── 01_eda_disaster_tweets.ipynb
│
└── src/
    └── with_pipeline.py
```
## Project Details

Exploratory data analysis is in: notebooks/01_eda_disaster_tweets.ipynb

The notebook includes:
basic dataset overview
class balance analysis
Word cloud to understand if stopword collection need to be update, if yes then get the list of words
initial text cleaning experiments

Future Improvements
Hyperparameter tuning (GridSearch / RandomizedSearch)
Additional models (e.g. Linear SVM)
More robust text preprocessing (emojis, URLs, hashtags)
Model evaluation on a held-out validation set