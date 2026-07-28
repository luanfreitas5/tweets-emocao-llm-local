"""Roteador de verificação de saúde."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.dependencies import get_settings
from app.schemas import HealthResponse
from src.config.settings import Settings

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health(settings: Settings = Depends(get_settings)) -> HealthResponse:
    """Retorna o estado do serviço e o modelo LLM configurado.

    Parameters
    ----------
    settings : Settings
        Configuração injetada.

    Returns
    -------
    HealthResponse
        Status ``ok`` e nome do modelo local.
    """
    return HealthResponse(status="ok", llm_model=settings.llm.model)
