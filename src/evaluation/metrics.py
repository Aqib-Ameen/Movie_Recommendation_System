"""
src/evaluation/metrics.py
--------------------------
Evaluation metrics for the recommendation system.

Rating-Prediction Metrics
--------------------------
- RMSE  — Root Mean Squared Error
- MAE   — Mean Absolute Error

Ranking Metrics (computed at cut-off K)
-----------------------------------------
- Precision@K  — fraction of top-K that are relevant
- Recall@K     — fraction of relevant items in top-K
- F1@K         — harmonic mean of Precision and Recall
- NDCG@K       — Normalized Discounted Cumulative Gain
- MRR          — Mean Reciprocal Rank

Catalogue Metric
----------------
- Coverage  — fraction of all items that the model ever recommends
"""
from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Rating-prediction metrics
# ---------------------------------------------------------------------------

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(y_true - y_pred)))


def evaluate_rating_predictions(
    model,
    test_df: pd.DataFrame,
) -> Dict[str, float]:
    """Compute RMSE and MAE over every (user, item) pair in test_df.

    Parameters
    ----------
    model   : any object with a .predict(user_id, item_id) -> float method
    test_df : pd.DataFrame with columns userId, movieId, rating

    Returns
    -------
    {"rmse": float, "mae": float}
    """
    y_true, y_pred = [], []
    for row in test_df.itertuples(index=False):
        try:
            pred = model.predict(row.userId, row.movieId)
        except Exception:
            pred = 0.0
        y_true.append(row.rating)
        y_pred.append(pred)

    y_true_arr = np.array(y_true, dtype=float)
    y_pred_arr = np.array(y_pred, dtype=float)
    return {"rmse": rmse(y_true_arr, y_pred_arr), "mae": mae(y_true_arr, y_pred_arr)}


# ---------------------------------------------------------------------------
# Ranking metrics
# ---------------------------------------------------------------------------

def precision_at_k(recommended: List[int], relevant: Set[int], k: int) -> float:
    """Precision@K."""
    if k == 0:
        return 0.0
    top_k = recommended[:k]
    hits = sum(1 for item in top_k if item in relevant)
    return hits / k


def recall_at_k(recommended: List[int], relevant: Set[int], k: int) -> float:
    """Recall@K."""
    if not relevant:
        return 0.0
    top_k = recommended[:k]
    hits = sum(1 for item in top_k if item in relevant)
    return hits / len(relevant)


def f1_at_k(recommended: List[int], relevant: Set[int], k: int) -> float:
    """F1@K."""
    p = precision_at_k(recommended, relevant, k)
    r = recall_at_k(recommended, relevant, k)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def ndcg_at_k(recommended: List[int], relevant: Set[int], k: int) -> float:
    """Normalized Discounted Cumulative Gain @ K."""
    top_k = recommended[:k]
    dcg = sum(
        1.0 / math.log2(idx + 2)
        for idx, item in enumerate(top_k)
        if item in relevant
    )
    # Ideal DCG: all relevant items at the top positions
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(idx + 2) for idx in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def mean_reciprocal_rank(recommended: List[int], relevant: Set[int]) -> float:
    """Mean Reciprocal Rank (MRR) — first relevant item position."""
    for idx, item in enumerate(recommended):
        if item in relevant:
            return 1.0 / (idx + 1)
    return 0.0


# ---------------------------------------------------------------------------
# Full ranking evaluation over a test set
# ---------------------------------------------------------------------------

def evaluate_ranking(
    model,
    test_df: pd.DataFrame,
    train_df: pd.DataFrame,
    k_values: Optional[List[int]] = None,
    relevance_threshold: float = 3.5,
) -> Dict[str, float]:
    """Compute macro-averaged ranking metrics over all users in test_df.

    Parameters
    ----------
    model               : object with .recommend(user_id, n) -> List[(item_id, score)]
    test_df             : held-out ratings
    train_df            : training ratings (used to exclude seen items)
    k_values            : list of cut-off K values, e.g. [5, 10, 20]
    relevance_threshold : min rating to consider an item "relevant"

    Returns
    -------
    Dict of metric_name -> macro-average across users
    """
    if k_values is None:
        k_values = [5, 10, 20]

    max_k = max(k_values)
    results: Dict[str, List[float]] = {
        f"precision@{k}": [] for k in k_values
    }
    results.update({f"recall@{k}": [] for k in k_values})
    results.update({f"ndcg@{k}": [] for k in k_values})
    results.update({f"f1@{k}": [] for k in k_values})
    results["mrr"] = []

    test_users = test_df["userId"].unique()

    for user_id in test_users:
        # Ground-truth: relevant items in test set
        user_test = test_df[test_df["userId"] == user_id]
        relevant = set(
            user_test[user_test["rating"] >= relevance_threshold]["movieId"].tolist()
        )
        if not relevant:
            continue  # skip users with no relevant items in test

        # Get recommendations
        try:
            recs = model.recommend(user_id, max_k)
        except Exception as exc:
            logger.debug("recommend() failed for user %d: %s", user_id, exc)
            recs = []

        rec_item_ids = [item_id for item_id, _ in recs]

        for k in k_values:
            results[f"precision@{k}"].append(precision_at_k(rec_item_ids, relevant, k))
            results[f"recall@{k}"].append(recall_at_k(rec_item_ids, relevant, k))
            results[f"ndcg@{k}"].append(ndcg_at_k(rec_item_ids, relevant, k))
            results[f"f1@{k}"].append(f1_at_k(rec_item_ids, relevant, k))
        results["mrr"].append(mean_reciprocal_rank(rec_item_ids, relevant))

    return {metric: float(np.mean(vals)) if vals else 0.0 for metric, vals in results.items()}


# ---------------------------------------------------------------------------
# Catalogue coverage
# ---------------------------------------------------------------------------

def coverage(
    model,
    user_ids: List[int],
    all_item_ids: Set[int],
    n: int = 10,
) -> float:
    """Fraction of the full item catalogue that the model recommends.

    Parameters
    ----------
    model        : object with .recommend(user_id, n) -> List[(item_id, score)]
    user_ids     : list of user IDs to sample recommendations for
    all_item_ids : the full set of catalogue item IDs
    n            : recommendation list length

    Returns
    -------
    float in [0, 1]
    """
    recommended_items: Set[int] = set()
    for user_id in user_ids:
        try:
            recs = model.recommend(user_id, n)
            recommended_items.update(item_id for item_id, _ in recs)
        except Exception:
            pass
    return len(recommended_items) / len(all_item_ids) if all_item_ids else 0.0
