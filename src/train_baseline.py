# src/train_baseline.py  (CLINC OOS intent + tweet_eval sentiment)
import joblib, numpy as np, pandas as pd
from datasets import load_dataset
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from pathlib import Path

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

def load_intent_clinc():
    # CLINC OOS "plus" split; aman untuk datasets 3.x
    ds = load_dataset("clinc_oos", "plus")
    train = pd.DataFrame(ds["train"])
    test  = pd.DataFrame(ds["test"])
    # kolom intent bisa ClassLabel(int) atau string; normalisasi ke string
    if np.issubdtype(train["intent"].dtype, np.integer):
        names = ds["train"].features["intent"].names
        train["label_name"] = train["intent"].map(lambda i: names[i])
        test["label_name"]  = test["intent"].map(lambda i: names[i])
    else:
        train["label_name"] = train["intent"].astype(str)
        test["label_name"]  = test["intent"].astype(str)
    train.rename(columns={"text":"text"}, inplace=True)
    test.rename(columns={"text":"text"}, inplace=True)
    return train, test

def load_tweet_sentiment():
    ds = load_dataset("tweet_eval", "sentiment")
    label_map = {0:"neg",1:"neu",2:"pos"}
    train = pd.DataFrame(ds["train"])
    val   = pd.DataFrame(ds["validation"])
    test  = pd.DataFrame(ds["test"])
    for df in (train,val,test):
        df["label_name"] = df["label"].map(label_map)
        df.rename(columns={"text":"text"}, inplace=True)
    train_all = pd.concat([train, val], ignore_index=True)
    return train_all, test

def train_intent():
    train, test = load_intent_clinc()
    Xtr, ytr = train["text"].values, train["label_name"].values
    Xte, yte = test["text"].values,  test["label_name"].values

    le = LabelEncoder().fit(ytr)
    ytr_enc = le.transform(ytr); yte_enc = le.transform(yte)

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(min_df=2, ngram_range=(1,2))),
        ("clf", LinearSVC())
    ])
    pipe.fit(Xtr, ytr_enc)
    pred = pipe.predict(Xte)
    print("\n=== INTENT (CLINC OOS) ===")
    print(classification_report(yte_enc, pred, target_names=le.classes_, digits=3))

    joblib.dump(pipe, MODELS_DIR/"intent_tfidf.joblib")
    joblib.dump(le,   MODELS_DIR/"intent_label_encoder.joblib")

def train_sentiment():
    train, test = load_tweet_sentiment()
    Xtr, ytr = train["text"].values, train["label_name"].values
    Xte, yte = test["text"].values,  test["label_name"].values

    le = LabelEncoder().fit(ytr)
    ytr_enc = le.transform(ytr); yte_enc = le.transform(yte)

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(min_df=2, ngram_range=(1,2))),
        ("clf", LogisticRegression(max_iter=200))
    ])
    pipe.fit(Xtr, ytr_enc)
    pred = pipe.predict(Xte)
    print("\n=== SENTIMENT (tweet_eval) ===")
    print(classification_report(yte_enc, pred, target_names=le.classes_, digits=3))

    joblib.dump(pipe, MODELS_DIR/"sentiment_tfidf.joblib")
    joblib.dump(le,   MODELS_DIR/"sentiment_label_encoder.joblib")

if __name__ == "__main__":
    train_intent()
    train_sentiment()
    print("\nModels saved to ./models")
