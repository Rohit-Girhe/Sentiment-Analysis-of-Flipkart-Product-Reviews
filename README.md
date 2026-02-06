# Sentiment Analysis of Flipkart Product Reviews

End‑to‑end sentiment analysis project on Flipkart product reviews (focused on
**Yonex Mavis 350 Nylon Shuttle**), including data preprocessing, feature
engineering, model training/comparison, and a Flask web app for live
predictions.

---

## 1. Project Overview

- **Goal**: Predict whether a Flipkart product review expresses **positive** or
  **negative** sentiment.
- **Target**: Binary sentiment from star ratings  
  - Ratings **1–2 → 0 (Negative)**  
  - Ratings **4–5 → 1 (Positive)**  
  - Rating **3 → dropped** (neutral/ambiguous)
- **Best model** (from experiments):
  - **TF‑IDF + LinearSVC**
  - Accuracy ≈ **92%**, F1 ≈ **0.96** on the binary task.

The project also includes manual, domain‑specific tests using sentences about
the *Yonex Mavis 350* shuttle to qualitatively validate the model.

---

## 2. Project Structure

- `Data/`
  - `data_cleaned.csv` – cleaned Flipkart review data.
- `src/`
  - `text_preprocessing.py` – text cleaning, tokenization, stopword handling,
    lemmatization, corpus preprocessing (keeps negation words like *not*).
  - `feature_extraction.py` – BoW, TF‑IDF, Word2Vec, BERT feature helpers.
  - `model_training.py` – main binary sentiment training script (TF‑IDF +
    LinearSVC), saves artifacts.
  - `model_comparison.py` – compares multiple models/feature types and prints a
    metrics table.
  - `manual_test_sentences.py` – runs hand‑crafted Yonex Mavis 350 sentences
    through the trained model and prints labels + scores.
- `artifacts/`
  - `sentiment_model.pkl` – final trained LinearSVC model.
  - `tfidf_vectorizer.pkl` – fitted TF‑IDF vectorizer.
- `app/`
  - `app.py` – Flask web app that loads artifacts and serves predictions.
  - `templates/index.html` – main UI.
  - `Static/style.css` – styling for the web app.
- `Notebook/`
  - Jupyter notebooks for data exploration, cleaning, and early experiments.
- `requirements.txt`
  - Python dependencies.

---

## 3. Environment Setup

```bash
git clone <this-repo>
cd Sentiment-Analysis-of-Flipkart-Product-Reviews

python -m venv .venv
source .venv/Scripts/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

Make sure NLTK resources are available (run once in a Python shell or
notebook):

```python
import nltk
nltk.download("stopwords")
nltk.download("wordnet")
```

---

## 4. Training the Final Model

The main training script:

```bash
python -m src.model_training
```

This will:

1. Load `Data/data_cleaned.csv`.
2. Preprocess the `Review text` column with `preprocess_corpus`.
3. Map ratings to binary sentiment (1–2 negative, 4–5 positive, drop 3).
4. Split into train/test.
5. Extract **TF‑IDF** features (unigrams + bigrams).
6. Train a **LinearSVC** classifier with class balancing.
7. Print accuracy, classification report, and confusion matrix.
8. Save:
   - `artifacts/sentiment_model.pkl`
   - `artifacts/tfidf_vectorizer.pkl`

---

## 5. Model Experiments & Comparison

To compare different feature types and models:

```bash
python -m src.model_comparison
```

This script evaluates, among others:

- BoW + MultinomialNB / LogisticRegression
- TF‑IDF + MultinomialNB / LogisticRegression / LinearSVC
- Word2Vec + LogisticRegression

It prints a table of **accuracy** and **F1** scores sorted by F1.  
`TF‑IDF + LinearSVC` emerged as the best configuration.

---

## 6. Manual Domain‑Specific Testing

To inspect how the model behaves on realistic Yonex Mavis 350 sentences:

```bash
python -m src.manual_test_sentences
```

This prints a table with:

- `text` – original Mavis 350 sentence.
- `cleaned` – preprocessed version.
- `label` – Positive / Negative.
- `score_pos` – confidence‑like score (0–1) for the positive class.

This is useful for report screenshots and qualitative analysis.

---

## 7. Running the Web App

Start the Flask app:

```bash
python app/app.py
```

Then open in a browser:

- `http://localhost:5000`

Features:

- Text area to paste a Flipkart review (e.g., for **Yonex Mavis 350**).
- On submit:
  - Text is cleaned using the same `clean_text` pipeline.
  - Transformed with the saved TF‑IDF vectorizer.
  - Sent to the LinearSVC model for prediction.
- The UI displays:
  - Predicted label (**Positive** / **Negative**).
  - Confidence (sigmoid‑transformed decision score).

---

## 8. Notes & Possible Extensions

- Add an explicit **“Mixed/Neutral”** category for borderline scores.
- Use **BERT embeddings** (see `feature_extraction.py`) for more advanced
  models.
- Extend the app to support:
  - Batch prediction for multiple reviews.
  - Visualizations (e.g., distribution of sentiments, word clouds).

This README summarizes how to set up, train, evaluate, and run the Flipkart
Sentiment Analysis project end‑to‑end.
