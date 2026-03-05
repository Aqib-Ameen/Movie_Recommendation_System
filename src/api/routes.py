"""
src/api/routes.py
-----------------
FastAPI route definitions. All routes use the AppState singleton
(populated in main.py at startup) to access the trained models.
"""
from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from src.api.schemas import (
    HealthResponse,
    RateRequest,
    RateResponse,
    RecommendationResponse,
    RecommendedItem,
    SimilarItem,
    SimilarItemsResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter()


# ---------------------------------------------------------------------------
# Helpers — lazy import to avoid circular during testing
# ---------------------------------------------------------------------------

def _get_state():
    from src.api.main import state
    return state


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

@router.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    """Liveness / readiness probe."""
    s = _get_state()
    return HealthResponse(status="ok", model_loaded=s.hybrid is not None)


# ---------------------------------------------------------------------------
# GET /recommend/{user_id}
# ---------------------------------------------------------------------------

@router.get(
    "/recommend/{user_id}",
    response_model=RecommendationResponse,
    tags=["Recommendations"],
    summary="Top-N hybrid recommendations for a user",
)
def recommend(
    user_id: int,
    n: int = Query(default=10, ge=1, le=100, description="Number of recommendations"),
    model: str = Query(default="hybrid", description="Model to use: hybrid | cf | cb"),
):
    s = _get_state()
    if s.hybrid is None:
        raise HTTPException(status_code=503, detail="Models not loaded yet. Retry shortly.")

    try:
        if model == "cf":
            recs_raw = s.cf.recommend(user_id, n)
        elif model == "cb":
            if s.ratings_df is None:
                raise HTTPException(status_code=503, detail="Ratings not available for CB model.")
            recs_raw = s.cb.recommend(user_id, s.ratings_df, n)
        else:
            recs_raw = s.hybrid.recommend(user_id, n, ratings_df=s.ratings_df)
    except Exception as exc:
        logger.exception("Recommendation failed for user %d", user_id)
        raise HTTPException(status_code=500, detail=str(exc))

    if not recs_raw and model != "cb":
        # May indicate an unknown user
        raise HTTPException(
            status_code=404,
            detail=f"No recommendations found for user_id={user_id}. "
                   "The user may not exist in the training data.",
        )

    recommendations = [
        RecommendedItem(movie_id=movie_id, score=round(score, 6))
        for movie_id, score in recs_raw
    ]
    return RecommendationResponse(
        user_id=user_id,
        recommendations=recommendations,
        model=model,
        n=len(recommendations),
    )


# ---------------------------------------------------------------------------
# GET /similar/{item_id}
# ---------------------------------------------------------------------------

@router.get(
    "/similar/{item_id}",
    response_model=SimilarItemsResponse,
    tags=["Recommendations"],
    summary="Items similar to a given movie (content-based)",
)
def similar_items(
    item_id: int,
    n: int = Query(default=10, ge=1, le=100, description="Number of similar items"),
):
    s = _get_state()
    if s.cb is None:
        raise HTTPException(status_code=503, detail="Content-based model not loaded.")

    try:
        similar_raw = s.cb.similar_items(item_id, n)
    except Exception as exc:
        logger.exception("similar_items failed for item %d", item_id)
        raise HTTPException(status_code=500, detail=str(exc))

    if not similar_raw:
        raise HTTPException(
            status_code=404,
            detail=f"No similar items found for movie_id={item_id}. "
                   "The movie may not be in the catalogue.",
        )

    return SimilarItemsResponse(
        movie_id=item_id,
        similar_items=[
            SimilarItem(movie_id=mid, similarity=round(sim, 6))
            for mid, sim in similar_raw
        ],
        n=len(similar_raw),
    )


# ---------------------------------------------------------------------------
# POST /rate
# ---------------------------------------------------------------------------

@router.post(
    "/rate",
    response_model=RateResponse,
    tags=["Ratings"],
    summary="Submit a new user rating",
)
def rate(body: RateRequest):
    """Record a new rating.

    In a production system this would persist to a database and trigger
    incremental model updates. Here it appends to the in-memory DataFrame.
    """
    import pandas as pd

    s = _get_state()
    if s.ratings_df is None:
        raise HTTPException(status_code=503, detail="Ratings store not initialised.")

    new_row = pd.DataFrame(
        [{"userId": body.user_id, "movieId": body.movie_id, "rating": body.rating}]
    )
    s.ratings_df = pd.concat([s.ratings_df, new_row], ignore_index=True)

    logger.info(
        "New rating recorded: user=%d, movie=%d, rating=%.1f",
        body.user_id,
        body.movie_id,
        body.rating,
    )
    return RateResponse(
        user_id=body.user_id,
        movie_id=body.movie_id,
        rating=body.rating,
    )
