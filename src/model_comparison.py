"""
Quick experiment script to compare different feature types and models
for binary sentiment classification on Flipkart reviews.

Binary mapping:
    Ratings 1–2 -> 0 (negative)
    Ratings 4–5 -> 1 (positive)
    Rating 3    -> dropped
"""

from __future__ import annotations

import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from src.feature_extraction import (
    extract_bow,
    extract_tfidf,
    extract_word2vec,
)
from src.text_preprocessing import preprocess_corpus, tokenize, remove_stopwords


def load_data(data_dir: str = "Data", filename: str = "data_cleaned.csv") -> pd.DataFrame:
    path = os.path.join(data_dir, filename)
    return pd.read_csv(path)


def prepare_binary(df: pd.DataFrame, text_column: str = "Review text") -> Tuple[pd.Series, pd.Series]:
    df = df.copy()
    df = df[df["Ratings"] != 3]
    df["sentiment"] = df["Ratings"].apply(lambda r: 1 if r >= 4 else 0)

    df["cleaned_review"] = preprocess_corpus(df[text_column])
    X = df["cleaned_review"]
    y = df["sentiment"]
    return X, y


def evaluate_model(clf, X_train, X_test, y_train, y_test, name: str) -> Dict[str, float]:
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    return {"model": name, "accuracy": acc, "f1": f1}


def main() -> None:
    df = load_data()
    X_text, y = prepare_binary(df, text_column="Review text")

    X_train_text, X_test_text, y_train, y_test = train_test_split(
        X_text,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    results = []

    # 1) BoW features
    X_train_bow, bow_vec = extract_bow(X_train_text, max_features=5000, ngram_range=(1, 1))
    X_test_bow = bow_vec.transform(X_test_text)

    results.append(
        evaluate_model(
            MultinomialNB(),
            X_train_bow,
            X_test_bow,
            y_train,
            y_test,
            "BoW + MultinomialNB",
        )
    )
    results.append(
        evaluate_model(
            LogisticRegression(max_iter=2000, class_weight="balanced"),
            X_train_bow,
            X_test_bow,
            y_train,
            y_test,
            "BoW + LogisticRegression",
        )
    )

    # 2) TF‑IDF features
    X_train_tfidf, tfidf_vec = extract_tfidf(X_train_text, max_features=8000, ngram_range=(1, 2))
    X_test_tfidf = tfidf_vec.transform(X_test_text)

    results.append(
        evaluate_model(
            MultinomialNB(),
            X_train_tfidf,
            X_test_tfidf,
            y_train,
            y_test,
            "TF‑IDF + MultinomialNB",
        )
    )
    results.append(
        evaluate_model(
            LogisticRegression(max_iter=2000, class_weight="balanced"),
            X_train_tfidf,
            X_test_tfidf,
            y_train,
            y_test,
            "TF‑IDF + LogisticRegression",
        )
    )
    results.append(
        evaluate_model(
            LinearSVC(class_weight="balanced"),
            X_train_tfidf,
            X_test_tfidf,
            y_train,
            y_test,
            "TF‑IDF + LinearSVC",
        )
    )

    # 3) Word2Vec features (simple, train on this dataset)
    from gensim.models import Word2Vec

    tokenized_train = [remove_stopwords(tokenize(t)) for t in X_train_text]
    tokenized_test = [remove_stopwords(tokenize(t)) for t in X_test_text]

    w2v_model = Word2Vec(
        sentences=tokenized_train,
        vector_size=100,
        window=5,
        min_count=2,
        workers=4,
    )
    X_train_w2v = extract_word2vec(tokenized_train, w2v_model.wv)
    X_test_w2v = extract_word2vec(tokenized_test, w2v_model.wv)

    results.append(
        evaluate_model(
            LogisticRegression(max_iter=2000, class_weight="balanced"),
            X_train_w2v,
            X_test_w2v,
            y_train,
            y_test,
            "Word2Vec + LogisticRegression",
        )
    )

    # Present results as a table
    res_df = pd.DataFrame(results).sort_values(by="f1", ascending=False)
    print("\n=== Model Comparison (sorted by F1) ===")
    print(res_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()

