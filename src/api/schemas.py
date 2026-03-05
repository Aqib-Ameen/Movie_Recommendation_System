"""
src/api/schemas.py
------------------
Pydantic request / response models for the FastAPI layer.
"""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Common
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str = "ok"
    model_loaded: bool


# ---------------------------------------------------------------------------
# Recommendation
# ---------------------------------------------------------------------------

class RecommendedItem(BaseModel):
    movie_id: int = Field(..., description="MovieLens movie identifier")
    score: float = Field(..., description="Hybrid recommendation score (higher = better)")


class RecommendationResponse(BaseModel):
    user_id: int
    recommendations: List[RecommendedItem]
    model: str = Field(default="hybrid", description="Which model produced these results")
    n: int = Field(..., description="Number of recommendations returned")


# ---------------------------------------------------------------------------
# Similar items
# ---------------------------------------------------------------------------

class SimilarItem(BaseModel):
    movie_id: int
    similarity: float = Field(..., description="Cosine similarity score in [0, 1]")


class SimilarItemsResponse(BaseModel):
    movie_id: int
    similar_items: List[SimilarItem]
    n: int


# ---------------------------------------------------------------------------
# Rating submission
# ---------------------------------------------------------------------------

class RateRequest(BaseModel):
    user_id: int = Field(..., gt=0, description="User submitting the rating")
    movie_id: int = Field(..., gt=0, description="Movie being rated")
    rating: float = Field(..., ge=0.5, le=5.0, description="Rating in [0.5, 5.0]")


class RateResponse(BaseModel):
    user_id: int
    movie_id: int
    rating: float
    message: str = "Rating recorded successfully"


# ---------------------------------------------------------------------------
# Error
# ---------------------------------------------------------------------------

class ErrorResponse(BaseModel):
    detail: str
    error_code: Optional[str] = None
