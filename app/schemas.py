"""Modelos de request/response da API FastAPI.

Os limites de tamanho dos textos vêm de ``configs/deploy.yaml`` e são aplicados
no roteador; aqui ficam apenas o formato e as proibições estruturais
(``extra="forbid"``: campo desconhecido é erro 422, não é ignorado).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class SentimentRequest(BaseModel):
    """Corpo da requisição de classificação de sentimento."""

    model_config = ConfigDict(extra="forbid")

    texts: list[str] = Field(min_length=1, description="Textos a classificar.")


class SentimentItem(BaseModel):
    """Resultado de classificação de um único texto."""

    text: str
    label: str
    score: float


class SentimentResponse(BaseModel):
    """Resposta da classificação de sentimento em lote."""

    model_config = ConfigDict(protected_namespaces=())

    results: list[SentimentItem]
    model_name: str


class HealthResponse(BaseModel):
    """Resposta do endpoint de verificação de saúde."""

    model_config = ConfigDict(protected_namespaces=())

    status: str = "ok"
    llm_model: str
    model_loaded: bool = False


class ModelMetadata(BaseModel):
    """Versões do serviço: API, modelos e commit que gerou a imagem."""

    model_config = ConfigDict(protected_namespaces=())

    api_version: str
    sentiment_model: str
    llm_model: str
    git_sha: str
