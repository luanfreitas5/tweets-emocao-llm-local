"""Modelos de request/response da API FastAPI."""

from __future__ import annotations

from pydantic import BaseModel, Field


class SentimentRequest(BaseModel):
    """Corpo da requisição de classificação de sentimento."""

    texts: list[str] = Field(min_length=1, description="Textos a classificar.")


class SentimentItem(BaseModel):
    """Resultado de classificação de um único texto."""

    text: str
    label: str
    score: float


class SentimentResponse(BaseModel):
    """Resposta da classificação de sentimento em lote."""

    results: list[SentimentItem]


class HealthResponse(BaseModel):
    """Resposta do endpoint de verificação de saúde."""

    status: str = "ok"
    llm_model: str
