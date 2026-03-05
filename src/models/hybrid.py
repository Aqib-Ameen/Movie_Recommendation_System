"""
src/models/hybrid.py
--------------------
Hybrid Recommender that blends Collaborative Filtering and Content-Based
Filtering with a configurable alpha weight:

    final_score = α × cf_score + (1 − α) × cb_score

Cold-start logic
----------------
Users with fewer than `cold_start_threshold` ratings fall back to
pure Content-Based recommendations (α = 0 for those users).
"""
from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from src.models.collaborative import SVDModel, UserBasedCF, ItemBasedCF
from src.models.content_based import ContentBasedFilter

logger = logging.getLogger(__name__)


class HybridRecommender:
    """Weighted hybrid of CF and Content-Based recommenders."""

    def __init__(
        self,
        config: dict,
        cf_model,   # SVDModel | UserBasedCF | ItemBasedCF
        cb_model: ContentBasedFilter,
        all_ratings: Optional[pd.DataFrame] = None,
    ) -> None:
        self.alpha: float = config["model"].get("hybrid_alpha", 0.6)
        self.cold_start_threshold: int = config["model"].get("cold_start_threshold", 5)
        self.top_n: int = config["api"].get("top_n", 10)

        self.cf = cf_model
        self.cb = cb_model
        self.all_ratings = all_ratings   # needed for CB user-profile building

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def recommend(
        self,
        user_id: int,
        n: Optional[int] = None,
        ratings_df: Optional[pd.DataFrame] = None,
    ) -> List[Tuple[int, float]]:
        """Return top-N (movieId, score) recommendations for user_id.

        Parameters
        ----------
        user_id     : the user to generate recommendations for
        n           : number of items to return (defaults to config top_n)
        ratings_df  : full ratings DataFrame used to build CB profiles;
                      if None, falls back to self.all_ratings
        """
        n = n or self.top_n
        df = ratings_df if ratings_df is not None else self.all_ratings

        effective_alpha = self._effective_alpha(user_id, df)
        logger.debug(
            "HybridRecommender: user=%d, α=%.2f (threshold=%d)",
            user_id, effective_alpha, self.cold_start_threshold,
        )

        # --- Get a broad candidate pool from both models ---
        cf_recs = self._cf_recommend(user_id, n * 3)
        cb_recs = self._cb_recommend(user_id, df, n * 3)

        # --- Normalise scores to [0, 1] ---
        cf_scores = self._normalize(dict(cf_recs))
        cb_scores = self._normalize(dict(cb_recs))

        # --- Combine ---
        all_items = set(cf_scores) | set(cb_scores)
        blended = {}
        for item_id in all_items:
            cf_s = cf_scores.get(item_id, 0.0)
            cb_s = cb_scores.get(item_id, 0.0)
            blended[item_id] = effective_alpha * cf_s + (1 - effective_alpha) * cb_s

        sorted_recs = sorted(blended.items(), key=lambda x: x[1], reverse=True)
        return sorted_recs[:n]

    def predict(
        self,
        user_id: int,
        item_id: int,
        ratings_df: Optional[pd.DataFrame] = None,
    ) -> float:
        """Predict a hybrid score for a single (user, item) pair."""
        df = ratings_df if ratings_df is not None else self.all_ratings
        effective_alpha = self._effective_alpha(user_id, df)

        cf_score = self._safe_cf_predict(user_id, item_id)
        cb_score = self.cb.predict(user_id, item_id, df) if df is not None else 0.0
        return effective_alpha * cf_score + (1 - effective_alpha) * cb_score

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _effective_alpha(self, user_id: int, df: Optional[pd.DataFrame]) -> float:
        """Return 0.0 (pure CB) for cold-start users, otherwise self.alpha."""
        if df is None:
            return self.alpha
        n_ratings = len(df[df["userId"] == user_id])
        if n_ratings < self.cold_start_threshold:
            logger.debug(
                "User %d is cold-start (%d ratings < threshold %d) — using CB only",
                user_id, n_ratings, self.cold_start_threshold,
            )
            return 0.0
        return self.alpha

    def _cf_recommend(self, user_id: int, n: int) -> List[Tuple[int, float]]:
        try:
            if isinstance(self.cf, SVDModel):
                return self.cf.recommend(user_id, n)
            else:
                return self.cf.recommend(user_id, n)
        except Exception as exc:
            logger.warning("CF recommend failed for user %d: %s", user_id, exc)
            return []

    def _cb_recommend(
        self, user_id: int, df: Optional[pd.DataFrame], n: int
    ) -> List[Tuple[int, float]]:
        if df is None:
            return []
        try:
            return self.cb.recommend(user_id, df, n)
        except Exception as exc:
            logger.warning("CB recommend failed for user %d: %s", user_id, exc)
            return []

    def _safe_cf_predict(self, user_id: int, item_id: int) -> float:
        try:
            return self.cf.predict(user_id, item_id)
        except Exception:
            return 0.0

    @staticmethod
    def _normalize(scores: dict) -> dict:
        """Min-max normalize a dict of {item_id: score} to [0, 1]."""
        if not scores:
            return {}
        vals = np.array(list(scores.values()), dtype=float)
        v_min, v_max = vals.min(), vals.max()
        if v_max == v_min:
            return {k: 1.0 for k in scores}
        return {k: float((v - v_min) / (v_max - v_min)) for k, v in scores.items()}
