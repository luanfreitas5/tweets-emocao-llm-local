"""Roteador de verificação de saúde e de versão do serviço."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends

from app.dependencies import get_classifier, get_metadata, get_settings
from app.schemas import HealthResponse, ModelMetadata
from src.config.settings import Settings
from src.models.sentiment import SentimentClassifier

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health(
    settings: Annotated[Settings, Depends(get_settings)],
    classifier: Annotated[SentimentClassifier, Depends(get_classifier)],
) -> HealthResponse:
    """Retorna o estado do serviço: processo ativo e se o modelo já está carregado.

    Parameters
    ----------
    settings : Settings
        Configuração injetada.
    classifier : SentimentClassifier
        Classificador injetado (informa se o modelo está em memória).

    Returns
    -------
    HealthResponse
        Status ``ok``, nome do modelo LLM e ``model_loaded``.
    """
    return HealthResponse(
        status="ok", llm_model=settings.llm.model, model_loaded=classifier.is_loaded
    )


@router.get("/version", response_model=ModelMetadata)
def version(metadata: Annotated[ModelMetadata, Depends(get_metadata)]) -> ModelMetadata:
    """Informa a versão da API, dos modelos e o commit da imagem em execução.

    Parameters
    ----------
    metadata : ModelMetadata
        Metadados montados no startup.

    Returns
    -------
    ModelMetadata
        Versões e SHA do commit.
    """
    return metadata
