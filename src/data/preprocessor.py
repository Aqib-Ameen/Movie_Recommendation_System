"""
src/data/preprocessor.py
------------------------
Cleans and prepares data for model training.

Steps
-----
1. Drop duplicate (userId, movieId) pairs – keep last interaction
2. Optionally normalise ratings to [0, 1]
3. Encode userId and movieId to contiguous integer indices
4. Build a sparse user–item interaction matrix
5. Stratified train / test split (per user)
"""
from __future__ import annotations

import logging
import os
import pickle
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class Preprocessor:
    """Transform raw ratings/movies DataFrames into model-ready artefacts."""

    def __init__(self, config: dict) -> None:
        self.config = config
        self.test_size: float = config["data"].get("test_size", 0.2)
        self.random_state: int = config["data"].get("random_state", 42)
        self.processed_dir = Path(config["data"].get("processed_dir", "data/processed/"))

        # Learned encodings (fitted on training data)
        self.user2idx: Dict[int, int] = {}
        self.idx2user: Dict[int, int] = {}
        self.item2idx: Dict[int, int] = {}
        self.idx2item: Dict[int, int] = {}

        self._fitted = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit_transform(
        self, ratings: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Fit encodings and split ratings into train / test DataFrames.

        Returns
        -------
        train_df, test_df : pd.DataFrame
            Both contain columns: userId, movieId, rating,
            user_idx (int), item_idx (int)
        """
        logger.info("Preprocessing %d rating rows …", len(ratings))

        # 1. Deduplicate
        ratings = self._deduplicate(ratings)

        # 2. Build encodings
        self._build_encodings(ratings)

        # 3. Stratified split
        train_df, test_df = self._split(ratings)

        # 4. Add integer indices
        train_df = self._add_indices(train_df)
        test_df = self._add_indices(test_df)

        self._fitted = True
        logger.info(
            "Train: %d rows | Test: %d rows | %d users | %d items",
            len(train_df),
            len(test_df),
            len(self.user2idx),
            len(self.item2idx),
        )
        return train_df, test_df

    def build_interaction_matrix(self, df: pd.DataFrame) -> csr_matrix:
        """Build a sparse user–item matrix from a (possibly partial) DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain user_idx, item_idx, rating columns.

        Returns
        -------
        csr_matrix of shape (n_users, n_items)
        """
        self._check_fitted()
        n_users = len(self.user2idx)
        n_items = len(self.item2idx)

        rows = df["user_idx"].values
        cols = df["item_idx"].values
        data = df["rating"].values.astype(np.float32)
        return csr_matrix((data, (rows, cols)), shape=(n_users, n_items))

    def normalize_ratings(self, df: pd.DataFrame) -> pd.DataFrame:
        """Scale ratings to [0, 1] (min–max over the full observed range)."""
        df = df.copy()
        r_min, r_max = df["rating"].min(), df["rating"].max()
        if r_max > r_min:
            df["rating"] = (df["rating"] - r_min) / (r_max - r_min)
        return df

    def save(self) -> None:
        """Persist encodings to disk."""
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        path = self.processed_dir / "preprocessor.pkl"
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "user2idx": self.user2idx,
                    "idx2user": self.idx2user,
                    "item2idx": self.item2idx,
                    "idx2item": self.idx2item,
                },
                f,
            )
        logger.info("Preprocessor state saved to %s", path)

    @classmethod
    def load(cls, config: dict) -> "Preprocessor":
        """Restore a fitted Preprocessor from disk."""
        proc = cls(config)
        path = Path(config["data"].get("processed_dir", "data/processed/")) / "preprocessor.pkl"
        with open(path, "rb") as f:
            state = pickle.load(f)
        proc.user2idx = state["user2idx"]
        proc.idx2user = state["idx2user"]
        proc.item2idx = state["item2idx"]
        proc.idx2item = state["idx2item"]
        proc._fitted = True
        return proc

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _deduplicate(df: pd.DataFrame) -> pd.DataFrame:
        before = len(df)
        df = df.sort_values("timestamp") if "timestamp" in df.columns else df
        df = df.drop_duplicates(subset=["userId", "movieId"], keep="last")
        removed = before - len(df)
        if removed:
            logger.info("Removed %d duplicate (userId, movieId) pairs", removed)
        return df.reset_index(drop=True)

    def _build_encodings(self, df: pd.DataFrame) -> None:
        unique_users = sorted(df["userId"].unique())
        unique_items = sorted(df["movieId"].unique())
        self.user2idx = {u: i for i, u in enumerate(unique_users)}
        self.idx2user = {i: u for u, i in self.user2idx.items()}
        self.item2idx = {m: i for i, m in enumerate(unique_items)}
        self.idx2item = {i: m for m, i in self.item2idx.items()}

    def _split(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Per-user stratified split.

        For each user, test_size fraction of their ratings go to test.
        Users with only one rating are kept entirely in train.
        """
        train_parts, test_parts = [], []
        for _, user_df in df.groupby("userId"):
            if len(user_df) < 2:
                train_parts.append(user_df)
                continue
            tr, te = train_test_split(
                user_df,
                test_size=self.test_size,
                random_state=self.random_state,
            )
            train_parts.append(tr)
            test_parts.append(te)

        train = pd.concat(train_parts).reset_index(drop=True)
        test = pd.concat(test_parts).reset_index(drop=True) if test_parts else pd.DataFrame(columns=df.columns)
        return train, test

    def _add_indices(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # Items in test that weren't in train get idx = -1 (handled by models)
        df["user_idx"] = df["userId"].map(self.user2idx).fillna(-1).astype(int)
        df["item_idx"] = df["movieId"].map(self.item2idx).fillna(-1).astype(int)
        return df

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError(
                "Preprocessor has not been fitted yet. Call fit_transform() first."
            )
