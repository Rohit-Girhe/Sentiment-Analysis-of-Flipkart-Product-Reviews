"""
Model training script for binary sentiment classification on Flipkart reviews.

Mapping:
    Ratings 1–2 -> 0 (negative)
    Ratings 4–5 -> 1 (positive)
    Rating 3    -> dropped

This script:
    - Loads cleaned CSV from Data/
    - Applies text preprocessing
    - Extracts TF‑IDF features
    - Trains a LinearSVC classifier
    - Evaluates on a hold‑out set
    - Saves the vectorizer and model into artifacts/
"""

from __future__ import annotations

import os
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

from src.text_preprocessing import preprocess_corpus
from src.feature_extraction import extract_tfidf


def load_data(data_dir: str = "Data", filename: str = "data_cleaned.csv") -> pd.DataFrame:
    """Load the cleaned CSV."""
    path = os.path.join(data_dir, filename)
    df = pd.read_csv(path)
    return df


def prepare_binary_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create binary sentiment labels from Ratings.

    - Drop rows with Ratings == 3
    - 1, 2 -> 0 (negative)
    - 4, 5 -> 1 (positive)
    """
    df = df.copy()
    df = df[df["Ratings"] != 3]
    df["sentiment"] = df["Ratings"].apply(lambda r: 1 if r >= 4 else 0)
    return df


def train_binary_model(
    df: pd.DataFrame,
    text_column: str = "Review text",
):
    """Full training pipeline for binary sentiment model."""
    # Text preprocessing
    df["cleaned_review"] = preprocess_corpus(df[text_column])

    # Label mapping
    df = prepare_binary_labels(df)

    X = df["cleaned_review"]
    y = df["sentiment"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # TF‑IDF features
    X_train_tfidf, tfidf_vectorizer = extract_tfidf(
        X_train,
        max_features=8000,
        ngram_range=(1, 2),
    )
    X_test_tfidf = tfidf_vectorizer.transform(X_test)

    # Classifier: TF‑IDF + LinearSVC
    model = LinearSVC(class_weight="balanced")
    model.fit(X_train_tfidf, y_train)

    # Evaluation
    y_pred = model.predict(X_test_tfidf)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification report:\n", classification_report(y_test, y_pred))
    print("\nConfusion matrix:\n", confusion_matrix(y_test, y_pred))

    return model, tfidf_vectorizer


def save_artifacts(
    model,
    vectorizer,
    artifacts_dir: str = "artifacts",
    model_name: str = "sentiment_model.pkl",
    vectorizer_name: str = "tfidf_vectorizer.pkl",
) -> None:
    """Save trained model and vectorizer."""
    os.makedirs(artifacts_dir, exist_ok=True)
    joblib.dump(model, os.path.join(artifacts_dir, model_name))
    joblib.dump(vectorizer, os.path.join(artifacts_dir, vectorizer_name))
    print(f"Saved model to {os.path.join(artifacts_dir, model_name)}")
    print(f"Saved vectorizer to {os.path.join(artifacts_dir, vectorizer_name)}")


def main() -> None:
    df = load_data()
    model, vectorizer = train_binary_model(df, text_column="Review text")
    save_artifacts(model, vectorizer)


if __name__ == "__main__":
    main()

