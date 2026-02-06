"""
Utility functions for text preprocessing used in the Flipkart
product review sentiment analysis project.

These helpers are designed to be:
- Reusable from notebooks and scripts
- Simple and sklearn‑friendly
"""

from __future__ import annotations

import re
import string
from typing import Iterable, List

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer


# Ensure required NLTK data is available.
# If it's already downloaded (as in your notebook), this is a no‑op.
try:
    _ = stopwords.words("english")
except LookupError:  # pragma: no cover
    nltk.download("stopwords")

try:
    _ = WordNetLemmatizer()
    nltk.data.find("corpora/wordnet")
except LookupError:  # pragma: no cover
    nltk.download("wordnet")


_BASE_STOPWORDS = set(stopwords.words("english"))

# Keep important negation words so the model can learn patterns like "not good"
_NEGATION_WORDS = {"no", "not", "nor", "never", "n't"}
_STOPWORDS = _BASE_STOPWORDS.difference(_NEGATION_WORDS)
_STEMMER = PorterStemmer()
_LEMMATIZER = WordNetLemmatizer()
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def basic_clean(text: str) -> str:
    """
    Perform basic text normalization:
    - Lowercasing
    - Remove URLs, HTML tags, digits
    - Remove extra whitespace
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\.\S+", " ", text)

    # Remove HTML tags
    text = re.sub(r"<.*?>", " ", text)

    # Remove digits
    text = re.sub(r"\d+", " ", text)

    # Remove punctuation
    text = text.translate(_PUNCT_TABLE)

    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


def tokenize(text: str) -> List[str]:
    """Split cleaned text into tokens."""
    return text.split()


def remove_stopwords(tokens: Iterable[str]) -> List[str]:
    """Remove common English stopwords."""
    return [t for t in tokens if t not in _STOPWORDS and len(t) > 1]


def stem_tokens(tokens: Iterable[str]) -> List[str]:
    """Apply Porter stemming."""
    return [_STEMMER.stem(t) for t in tokens]


def lemmatize_tokens(tokens: Iterable[str]) -> List[str]:
    """Apply WordNet lemmatization."""
    return [_LEMMATIZER.lemmatize(t) for t in tokens]


def clean_text(
    text: str,
    use_stemming: bool = False,
    use_lemmatization: bool = True,
) -> str:
    """
    Full preprocessing pipeline for a single text string.

    Steps:
    1. Basic cleaning (lowercase, remove urls/html/digits/punctuation)
    2. Tokenization
    3. Stopword removal
    4. Stemming or Lemmatization (optional, controlled by flags)
    """
    text = basic_clean(text)
    tokens = tokenize(text)
    tokens = remove_stopwords(tokens)

    if use_stemming and not use_lemmatization:
        tokens = stem_tokens(tokens)
    elif use_lemmatization:
        tokens = lemmatize_tokens(tokens)

    return " ".join(tokens)


def preprocess_corpus(
    texts: Iterable[str],
    use_stemming: bool = False,
    use_lemmatization: bool = True,
) -> List[str]:
    """
    Apply `clean_text` to an iterable of texts.

    Example (pandas):
        df["cleaned_review"] = preprocess_corpus(df["Review"])
    """
    return [
        clean_text(t, use_stemming=use_stemming, use_lemmatization=use_lemmatization)
        for t in texts
    ]


__all__ = [
    "basic_clean",
    "tokenize",
    "remove_stopwords",
    "stem_tokens",
    "lemmatize_tokens",
    "clean_text",
    "preprocess_corpus",
]

