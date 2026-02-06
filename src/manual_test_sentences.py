"""
Run a set of hand-crafted test sentences through the trained model
to inspect predictions on edge cases (negations, mixed sentiment, etc.).

Usage:
    python -m src.manual_test_sentences
"""

from __future__ import annotations

import os
import sys

import joblib
import pandas as pd

# Ensure project root is on sys.path (so this works even if run from subdirs)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.text_preprocessing import clean_text  # noqa: E402


ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "artifacts")
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "sentiment_model.pkl")
VECTORIZER_PATH = os.path.join(ARTIFACTS_DIR, "tfidf_vectorizer.pkl")


def load_artifacts():
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer


TEST_SENTENCES = [
    # Strong positive (Yonex Mavis 350)
    "Yonex Mavis 350 shuttles are very consistent and perfect for regular practice.",
    "Flight of the Mavis 350 is almost like feather shuttles, really impressed.",
    "Durability of these Mavis 350 nylon shuttles is excellent, they last many games.",
    "The shuttle speed is accurate and rallies feel very smooth with Mavis 350.",
    "Original Yonex Mavis 350, great control and stable trajectory.",
    # Strong negative
    "Mavis 350 shuttles lost shape after just two games, very disappointed.",
    "These Yonex Mavis 350 are clearly fake, flight and speed are terrible.",
    "Shuttle cork came off the base, worst Mavis 350 I have ever used.",
    "Too slow and wobbly in the air, not suitable for serious matches.",
    "Packaging was damaged and half the Mavis 350 shuttles were unusable.",
    # Negation / subtle
    "Mavis 350 is not as durable as I expected for the price.",
    "The shuttle is not bad for casual play but not good enough for tournaments.",
    "Trajectory is not very stable when playing powerful smashes.",
    "These are not original Yonex Mavis 350, quality feels cheap.",
    "Game experience is not great with these shuttles on outdoor courts.",
    # Mixed / contrast
    "Control with Mavis 350 is good but the shuttles wear out too quickly.",
    "Flight is consistent but the speed of this batch of Mavis 350 is slower than rated.",
    "Price is reasonable, however the durability of these Yonex Mavis 350 shuttles is disappointing.",
    "Smashes feel powerful with Mavis 350, yet the cork cracks after a few games.",
    "Good for beginners but experienced players may not like this Mavis 350 shuttle.",
    # Neutral / mild
    "Mavis 350 is okay for practice sessions, nothing special.",
    "Average performance Yonex shuttle, works fine for school level games.",
    "These shuttles are decent for indoor courts but just average outdoors.",
    "Quality of Mavis 350 is acceptable considering the budget.",
    "Neither very good nor very bad, Mavis 350 is just a normal practice shuttle.",
]


def main() -> None:
    model, vectorizer = load_artifacts()

    rows = []
    for text in TEST_SENTENCES:
        cleaned = clean_text(text)
        X_vec = vectorizer.transform([cleaned])
        # LinearSVC -> decision_function
        margin = float(model.decision_function(X_vec)[0])

        # Map margin to pseudo probability via sigmoid
        import math

        prob_pos = 1.0 / (1.0 + math.exp(-margin))
        label = "Positive" if margin >= 0 else "Negative"

        rows.append(
            {
                "text": text,
                "cleaned": cleaned,
                "label": label,
                "score_pos": prob_pos,
            }
        )

    df = pd.DataFrame(rows)
    # Sort by score to see strong vs weak predictions
    df_sorted = df.sort_values(by="score_pos", ascending=False)
    pd.set_option("display.max_colwidth", 120)
    print(df_sorted.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()

