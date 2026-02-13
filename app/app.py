from __future__ import annotations

import os
import sys

from flask import Flask, render_template, request
import joblib

# Ensure project root is on sys.path so we can import src.* when running app/app.py
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.text_preprocessing import clean_text


app = Flask(__name__, template_folder="templates", static_folder="Static")


# Load artifacts once at startup
ARTIFACTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "artifacts")
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "sentiment_model.pkl")
VECTORIZER_PATH = os.path.join(ARTIFACTS_DIR, "tfidf_vectorizer.pkl")

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)


def predict_sentiment(text: str) -> dict:
    """
    Run the full inference pipeline on a single review.

    Returns a dict with:
      - label: 'Positive' or 'Negative'
      - score: confidence score derived from the decision function
    """
    import math

    cleaned = clean_text(text)
    X_vec = vectorizer.transform([cleaned])

    # LinearSVC does not provide predict_proba, so we use decision_function
    decision = float(model.decision_function(X_vec)[0])
    # Map margin to a pseudo‑probability via sigmoid for display purposes
    prob_pos = 1.0 / (1.0 + math.exp(-decision))

    label = "Positive" if decision >= 0 else "Negative"
    return {"label": label, "score": prob_pos}


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    user_text = ""

    if request.method == "POST":
        user_text = request.form.get("review_text", "").strip()
        if user_text:
            prediction = predict_sentiment(user_text)

    return render_template("index.html", prediction=prediction, user_text=user_text)


if __name__ == "__main__":
    # Run the app: python app/app.py
    # Debug mode is disabled by default for security; control it via FLASK_DEBUG if needed.
    debug_enabled = os.environ.get("FLASK_DEBUG") == "1"
    app.run(host="0.0.0.0", port=5000, debug=debug_enabled)

