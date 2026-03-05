"""
src/models/collaborative.py
----------------------------
Collaborative Filtering models:

1. UserBasedCF  — cosine similarity between user vectors
2. ItemBasedCF  — cosine similarity between item vectors
3. SVDModel     — Matrix Factorisation via scikit-surprise SVD

All three expose a unified interface:
    .fit(train_df)
    .predict(user_id, item_id) -> float
    .recommend(user_id, n) -> List[Tuple[int, float]]  (movieId, score)
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Try importing scikit-surprise (optional heavy dependency)
# ---------------------------------------------------------------------------
try:
    from surprise import SVD, Dataset, Reader
    from surprise.model_selection import GridSearchCV as SurpriseGridCV
    _SURPRISE_AVAILABLE = True
except ImportError:  # pragma: no cover
    _SURPRISE_AVAILABLE = False
    logger.warning(
        "scikit-surprise not installed. SVDModel will not be available. "
        "Install with: pip install scikit-surprise"
    )


# ===========================================================================
# User-Based Collaborative Filtering
# ===========================================================================

class UserBasedCF:
    """User-based collaborative filter using cosine similarity."""

    def __init__(self, config: dict) -> None:
        self.k: int = config["model"].get("knn_k", 20)
        self.min_k: int = config["model"].get("knn_min_k", 1)

        # Fitted artefacts
        self._user_item: Optional[np.ndarray] = None   # dense (n_users, n_items)
        self._sim: Optional[np.ndarray] = None          # (n_users, n_users)
        self._user2idx: Dict[int, int] = {}
        self._item2idx: Dict[int, int] = {}
        self._idx2item: Dict[int, int] = {}
        self._global_mean: float = 0.0

    # ------------------------------------------------------------------

    def fit(self, train_df: pd.DataFrame) -> "UserBasedCF":
        """Learn user-user similarity from training ratings."""
        logger.info("UserBasedCF: building user–item matrix …")
        self._user2idx = {u: i for i, u in enumerate(sorted(train_df["userId"].unique()))}
        self._item2idx = {m: i for i, m in enumerate(sorted(train_df["movieId"].unique()))}
        self._idx2item = {i: m for m, i in self._item2idx.items()}
        self._global_mean = train_df["rating"].mean()

        n_users = len(self._user2idx)
        n_items = len(self._item2idx)
        matrix = np.zeros((n_users, n_items), dtype=np.float32)
        for row in train_df.itertuples(index=False):
            u_idx = self._user2idx.get(row.userId, -1)
            i_idx = self._item2idx.get(row.movieId, -1)
            if u_idx >= 0 and i_idx >= 0:
                matrix[u_idx, i_idx] = row.rating

        self._user_item = matrix
        logger.info("UserBasedCF: computing %d×%d similarity matrix …", n_users, n_users)
        self._sim = cosine_similarity(matrix)
        np.fill_diagonal(self._sim, 0)   # exclude self-similarity
        logger.info("UserBasedCF: ready.")
        return self

    def predict(self, user_id: int, item_id: int) -> float:
        """Predict a single rating."""
        u_idx = self._user2idx.get(user_id)
        i_idx = self._item2idx.get(item_id)
        if u_idx is None or i_idx is None:
            return self._global_mean

        sim_row = self._sim[u_idx]  # similarities to all other users
        item_col = self._user_item[:, i_idx]  # who rated this item

        rated_mask = item_col > 0
        if not rated_mask.any():
            return self._global_mean

        # Top-K similar users who rated this item
        sim_scores = sim_row * rated_mask.astype(float)
        top_k_idx = np.argsort(sim_scores)[::-1][: self.k]
        top_k_sim = sim_scores[top_k_idx]
        top_k_ratings = item_col[top_k_idx]

        valid = top_k_sim > 0
        if valid.sum() < self.min_k:
            return self._global_mean

        numerator = np.dot(top_k_sim[valid], top_k_ratings[valid])
        denominator = top_k_sim[valid].sum()
        return float(numerator / denominator) if denominator else self._global_mean

    def recommend(self, user_id: int, n: int = 10) -> List[Tuple[int, float]]:
        """Return top-N (movieId, predicted_score) not already seen by user."""
        u_idx = self._user2idx.get(user_id)
        if u_idx is None:
            return []

        seen_mask = self._user_item[u_idx] > 0
        scores = []
        for i_idx in range(self._user_item.shape[1]):
            if seen_mask[i_idx]:
                continue
            score = self.predict(user_id, self._idx2item[i_idx])
            scores.append((self._idx2item[i_idx], score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:n]

    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> "UserBasedCF":
        return joblib.load(path)


# ===========================================================================
# Item-Based Collaborative Filtering
# ===========================================================================

class ItemBasedCF:
    """Item-based collaborative filter using item–item cosine similarity."""

    def __init__(self, config: dict) -> None:
        self.k: int = config["model"].get("knn_k", 20)
        self._user_item: Optional[np.ndarray] = None
        self._item_sim: Optional[np.ndarray] = None
        self._user2idx: Dict[int, int] = {}
        self._item2idx: Dict[int, int] = {}
        self._idx2item: Dict[int, int] = {}
        self._global_mean: float = 0.0

    def fit(self, train_df: pd.DataFrame) -> "ItemBasedCF":
        logger.info("ItemBasedCF: building item–item similarity …")
        self._user2idx = {u: i for i, u in enumerate(sorted(train_df["userId"].unique()))}
        self._item2idx = {m: i for i, m in enumerate(sorted(train_df["movieId"].unique()))}
        self._idx2item = {i: m for m, i in self._item2idx.items()}
        self._global_mean = train_df["rating"].mean()

        n_users = len(self._user2idx)
        n_items = len(self._item2idx)
        matrix = np.zeros((n_users, n_items), dtype=np.float32)
        for row in train_df.itertuples(index=False):
            u_idx = self._user2idx.get(row.userId, -1)
            i_idx = self._item2idx.get(row.movieId, -1)
            if u_idx >= 0 and i_idx >= 0:
                matrix[u_idx, i_idx] = row.rating

        self._user_item = matrix
        self._item_sim = cosine_similarity(matrix.T)   # (n_items, n_items)
        np.fill_diagonal(self._item_sim, 0)
        logger.info("ItemBasedCF: ready.")
        return self

    def predict(self, user_id: int, item_id: int) -> float:
        u_idx = self._user2idx.get(user_id)
        i_idx = self._item2idx.get(item_id)
        if u_idx is None or i_idx is None:
            return self._global_mean

        user_ratings = self._user_item[u_idx]           # ratings this user gave
        sim_row = self._item_sim[i_idx]                 # similarity to all items
        rated_mask = user_ratings > 0

        top_k_idx = np.argsort(sim_row * rated_mask)[: :-1][: self.k]
        top_k_sim = sim_row[top_k_idx]
        top_k_ratings = user_ratings[top_k_idx]

        valid = top_k_sim > 0
        if not valid.any():
            return self._global_mean

        numerator = np.dot(top_k_sim[valid], top_k_ratings[valid])
        denominator = top_k_sim[valid].sum()
        return float(numerator / denominator) if denominator else self._global_mean

    def recommend(self, user_id: int, n: int = 10) -> List[Tuple[int, float]]:
        u_idx = self._user2idx.get(user_id)
        if u_idx is None:
            return []
        seen_mask = self._user_item[u_idx] > 0
        scores = []
        for i_idx in range(self._user_item.shape[1]):
            if seen_mask[i_idx]:
                continue
            score = self.predict(user_id, self._idx2item[i_idx])
            scores.append((self._idx2item[i_idx], score))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:n]

    def similar_items(self, item_id: int, n: int = 10) -> List[Tuple[int, float]]:
        """Return the n most similar items to item_id."""
        i_idx = self._item2idx.get(item_id)
        if i_idx is None:
            return []
        sim_row = self._item_sim[i_idx]
        top_k = np.argsort(sim_row)[::-1][:n]
        return [(self._idx2item[j], float(sim_row[j])) for j in top_k if sim_row[j] > 0]

    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> "ItemBasedCF":
        return joblib.load(path)


# ===========================================================================
# SVD Matrix Factorisation (scikit-surprise)
# ===========================================================================

class SVDModel:
    """Matrix Factorisation using scikit-surprise SVD."""

    def __init__(self, config: dict) -> None:
        if not _SURPRISE_AVAILABLE:
            raise ImportError("Install scikit-surprise: pip install scikit-surprise")
        self.n_factors: int = config["model"].get("svd_factors", 50)
        self.n_epochs: int = config["model"].get("svd_epochs", 20)
        self.lr_all: float = config["model"].get("svd_lr_all", 0.005)
        self.reg_all: float = config["model"].get("svd_reg_all", 0.02)
        self._algo: Optional[SVD] = None
        self._trainset = None

    def fit(self, train_df: pd.DataFrame) -> "SVDModel":
        """Fit SVD on training ratings."""
        logger.info(
            "SVDModel: training SVD (factors=%d, epochs=%d) …",
            self.n_factors, self.n_epochs,
        )
        reader = Reader(rating_scale=(0.5, 5.0))
        data = Dataset.load_from_df(train_df[["userId", "movieId", "rating"]], reader)
        self._trainset = data.build_full_trainset()
        self._algo = SVD(
            n_factors=self.n_factors,
            n_epochs=self.n_epochs,
            lr_all=self.lr_all,
            reg_all=self.reg_all,
            random_state=42,
        )
        self._algo.fit(self._trainset)
        logger.info("SVDModel: ready.")
        return self

    def predict(self, user_id: int, item_id: int) -> float:
        pred = self._algo.predict(str(user_id), str(item_id))
        return float(pred.est)

    def recommend(self, user_id: int, n: int = 10, all_item_ids: Optional[List[int]] = None) -> List[Tuple[int, float]]:
        """Recommend top-N unseen items for user_id."""
        if self._trainset is None:
            return []
        # Items already rated by user in training
        try:
            inner_uid = self._trainset.to_inner_uid(str(user_id))
            seen = {
                self._trainset.to_raw_iid(iid)
                for iid, _ in self._trainset.ur[inner_uid]
            }
        except ValueError:
            seen = set()

        candidate_ids = all_item_ids or [
            self._trainset.to_raw_iid(iid)
            for iid in self._trainset.all_items()
        ]
        scores = [
            (int(iid), self.predict(user_id, int(iid)))
            for iid in candidate_ids
            if str(iid) not in seen
        ]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:n]

    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> "SVDModel":
        return joblib.load(path)
