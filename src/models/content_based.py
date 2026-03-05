"""
src/models/content_based.py
----------------------------
Content-Based Filtering using TF-IDF on movie genres/tags and
cosine similarity between item vectors and user profile vectors.

Flow
----
1. Build item feature matrix: TF-IDF on the 'genres' (and optionally 'tags') column
2. Build user profile: weighted average of item vectors for items the user rated
3. Score unseen items: cosine similarity(user_profile, item_vectors)
4. Return top-N ranked items
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


class ContentBasedFilter:
    """TF-IDF + Cosine Similarity content-based recommender."""

    def __init__(self, config: dict) -> None:
        cf = config["model"]
        self.max_features: int = cf.get("tfidf_max_features", 5000)
        ngram_range_cfg = cf.get("tfidf_ngram_range", [1, 2])
        self.ngram_range: Tuple[int, int] = tuple(ngram_range_cfg)  # type: ignore[assignment]

        self._vectorizer: Optional[TfidfVectorizer] = None
        self._item_matrix: Optional[np.ndarray] = None   # (n_items, n_features) dense
        self._item2idx: Dict[int, int] = {}
        self._idx2item: Dict[int, int] = {}
        self._item_features: Optional[pd.Series] = None  # raw text per movieId

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, movies: pd.DataFrame, ratings: Optional[pd.DataFrame] = None) -> "ContentBasedFilter":
        """Fit TF-IDF on item features.

        Parameters
        ----------
        movies  : pd.DataFrame with columns (movieId, title, genres)
        ratings : Optional — if provided, only movies that appear in ratings
                              are kept (to save memory on large catalogues).
        """
        logger.info("ContentBasedFilter: building item feature matrix …")

        df = movies.copy()
        if ratings is not None:
            valid_ids = ratings["movieId"].unique()
            df = df[df["movieId"].isin(valid_ids)]

        df = df.drop_duplicates(subset="movieId").reset_index(drop=True)

        # Build the text feature: "genres" pipe → space + title words
        df["_text"] = df["genres"].str.replace("|", " ", regex=False)
        if "tags" in df.columns:
            df["_text"] += " " + df["tags"].fillna("")

        self._item_features = df.set_index("movieId")["_text"]
        self._item2idx = {mid: i for i, mid in enumerate(df["movieId"])}
        self._idx2item = {i: mid for mid, i in self._item2idx.items()}

        self._vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            stop_words="english",
            sublinear_tf=True,
        )
        tfidf_sparse = self._vectorizer.fit_transform(df["_text"])
        # Store as dense for cosine similarity convenience (sparse kept for large datasets)
        self._item_matrix = tfidf_sparse.toarray().astype(np.float32)

        logger.info(
            "ContentBasedFilter: ready. Items=%d, Features=%d",
            len(self._item2idx),
            self._item_matrix.shape[1],
        )
        return self

    def predict(self, user_id: int, item_id: int, ratings_df: pd.DataFrame) -> float:
        """Predict a pseudo-score for a user–item pair based on content similarity."""
        user_profile = self._build_user_profile(user_id, ratings_df)
        if user_profile is None:
            return 0.0
        i_idx = self._item2idx.get(item_id)
        if i_idx is None:
            return 0.0
        item_vec = self._item_matrix[i_idx].reshape(1, -1)
        sim = cosine_similarity(user_profile, item_vec)[0, 0]
        return float(sim)

    def recommend(
        self,
        user_id: int,
        ratings_df: pd.DataFrame,
        n: int = 10,
    ) -> List[Tuple[int, float]]:
        """Return top-N (movieId, score) not already seen by user."""
        user_profile = self._build_user_profile(user_id, ratings_df)
        if user_profile is None:
            logger.debug(
                "ContentBased: user %d has no ratings — cannot build profile", user_id
            )
            return []

        seen_ids = set(
            ratings_df[ratings_df["userId"] == user_id]["movieId"].tolist()
        )

        sim_scores = cosine_similarity(user_profile, self._item_matrix)[0]
        results = []
        for i_idx, score in enumerate(sim_scores):
            movie_id = self._idx2item[i_idx]
            if movie_id not in seen_ids:
                results.append((movie_id, float(score)))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:n]

    def similar_items(self, item_id: int, n: int = 10) -> List[Tuple[int, float]]:
        """Return the n most similar items to a given item by TF-IDF cosine similarity."""
        i_idx = self._item2idx.get(item_id)
        if i_idx is None:
            return []
        item_vec = self._item_matrix[i_idx].reshape(1, -1)
        sim_scores = cosine_similarity(item_vec, self._item_matrix)[0]
        sim_scores[i_idx] = -1  # exclude self
        top_k = np.argsort(sim_scores)[::-1][:n]
        return [(self._idx2item[j], float(sim_scores[j])) for j in top_k if sim_scores[j] > 0]

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> "ContentBasedFilter":
        return joblib.load(path)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_user_profile(
        self, user_id: int, ratings_df: pd.DataFrame
    ) -> Optional[np.ndarray]:
        """Weighted average of item TF-IDF vectors based on user ratings."""
        user_ratings = ratings_df[ratings_df["userId"] == user_id]
        if user_ratings.empty:
            return None

        weighted_sum = np.zeros(self._item_matrix.shape[1], dtype=np.float64)
        total_weight = 0.0

        for row in user_ratings.itertuples(index=False):
            i_idx = self._item2idx.get(row.movieId)
            if i_idx is None:
                continue
            weight = float(row.rating)
            weighted_sum += weight * self._item_matrix[i_idx]
            total_weight += weight

        if total_weight == 0:
            return None

        profile = (weighted_sum / total_weight).reshape(1, -1).astype(np.float32)
        return profile
