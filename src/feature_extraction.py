"""
Feature extraction utilities for sentiment analysis of Flipkart product reviews.

Provides:
- Bag of Words (BoW)
- TF‑IDF
- Word2Vec (average word embeddings)
- BERT embeddings (sentence-level)
"""

from __future__ import annotations

from typing import Iterable, List, Tuple

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def extract_bow(
    texts: Iterable[str],
    max_features: int | None = 5000,
    ngram_range: Tuple[int, int] = (1, 1),
):
    """
    Bag-of-Words representation using sklearn's CountVectorizer.

    Returns
    -------
    X_bow : sparse matrix
        BoW feature matrix.
    vectorizer : CountVectorizer
        Fitted vectorizer (save this for inference).
    """
    vectorizer = CountVectorizer(max_features=max_features, ngram_range=ngram_range)
    X_bow = vectorizer.fit_transform(texts)
    return X_bow, vectorizer


def extract_tfidf(
    texts: Iterable[str],
    max_features: int | None = 8000,
    ngram_range: Tuple[int, int] = (1, 2),
):
    """
    TF‑IDF representation using sklearn's TfidfVectorizer.

    Returns
    -------
    X_tfidf : sparse matrix
        TF‑IDF feature matrix.
    vectorizer : TfidfVectorizer
        Fitted vectorizer (save this for inference).
    """
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
    X_tfidf = vectorizer.fit_transform(texts)
    return X_tfidf, vectorizer


def extract_word2vec(
    tokenized_texts: Iterable[List[str]],
    model,
    vector_size: int | None = None,
):
    """
    Compute document vectors by averaging Word2Vec word embeddings.

    Parameters
    ----------
    tokenized_texts : iterable of list[str]
        Pre-tokenized texts (e.g. from your preprocessing step).
    model : gensim Word2Vec or KeyedVectors
        Trained Word2Vec-like model.
    vector_size : int, optional
        Embedding dimension. If None, inferred from model.

    Returns
    -------
    X_w2v : np.ndarray, shape (n_samples, vector_size)
    """
    if vector_size is None:
        # Works for both Word2Vec and KeyedVectors in recent gensim
        vector_size = model.vector_size

    def doc_vector(tokens: List[str]) -> np.ndarray:
        vectors = []
        for t in tokens:
            if t in model.key_to_index:  # for gensim >=4
                vectors.append(model[t])
        if not vectors:
            return np.zeros(vector_size, dtype="float32")
        return np.mean(vectors, axis=0)

    X_w2v = np.vstack([doc_vector(tokens) for tokens in tokenized_texts])
    return X_w2v


def extract_bert_embeddings(
    texts: List[str],
    model_name: str = "bert-base-uncased",
    batch_size: int = 16,
    device: str | None = None,
):
    """
    Get sentence-level BERT embeddings using Hugging Face transformers.

    Uses the [CLS] token representation by default.

    Returns
    -------
    X_bert : np.ndarray, shape (n_samples, hidden_size)
    model, tokenizer : loaded transformer model and tokenizer
        Useful if you want to reuse them without re-loading.
    """
    from transformers import AutoModel, AutoTokenizer  # lazy import
    import torch

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    all_embeddings: List[np.ndarray] = []

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=256,
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}

            outputs = model(**encoded)
            # CLS token embedding: outputs.last_hidden_state[:, 0, :]
            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            all_embeddings.append(cls_embeddings)

    X_bert = np.vstack(all_embeddings)
    return X_bert, model, tokenizer


__all__ = [
    "extract_bow",
    "extract_tfidf",
    "extract_word2vec",
    "extract_bert_embeddings",
]

