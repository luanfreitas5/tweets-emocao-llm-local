"""Instância da aplicação FastAPI, ciclo de vida (``lifespan``) e roteadores.

Os componentes pesados são criados **uma vez** no startup e guardados em
``app.state``. Com ``preload_model`` verdadeiro o modelo de sentimento é
carregado já no startup: se o artefato estiver ausente/corrompido, o serviço
falha ao subir em vez de falhar na primeira requisição.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import yaml
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import health, sentiment, summary
from app.schemas import ModelMetadata
from src.config.logging import configure_logging
from src.config.paths import CONFIGS_DIR
from src.config.settings import load_settings
from src.llm.summarizer import InsightSummarizer
from src.models.sentiment import SentimentClassifier

_DEPLOY_YAML: Path = CONFIGS_DIR / "deploy.yaml"
_DEFAULT_LIMITS = {"max_texts_per_request": 256, "max_text_length": 512}


def _load_deploy_config() -> dict[str, Any]:
    """Carrega ``configs/deploy.yaml`` (título, versão, CORS, limites)."""
    if not _DEPLOY_YAML.exists():
        return {}
    with _DEPLOY_YAML.open(encoding="utf-8") as handler:
        return yaml.safe_load(handler) or {}


def create_app(*, preload_model: bool | None = None, configure_log: bool = True) -> FastAPI:
    """Cria e configura a aplicação FastAPI.

    Parameters
    ----------
    preload_model : bool | None, optional
        Carrega o modelo de sentimento no startup. Se ``None``, usa
        ``api.preload_model`` de ``configs/deploy.yaml``. Testes passam ``False``.
    configure_log : bool, optional
        Configura o logging no startup (ponto de entrada do serviço), by default True.
        Testes passam ``False`` para não substituir os handlers do pytest.

    Returns
    -------
    FastAPI
        Aplicação com CORS restrito a localhost e roteadores registrados.

    Examples
    --------
    >>> app = create_app(preload_model=False, configure_log=False)  # doctest: +SKIP
    """
    deploy = _load_deploy_config()
    api_cfg = deploy.get("api", {})
    should_preload = api_cfg.get("preload_model", True) if preload_model is None else preload_model

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        """Cria os componentes uma única vez; falha cedo se o modelo não carregar."""
        if configure_log:
            configure_logging()
        settings = load_settings()
        classifier = SentimentClassifier(settings.model.sentiment)
        if should_preload:
            classifier.load()
        app.state.settings = settings
        app.state.classifier = classifier
        app.state.summarizer = InsightSummarizer(settings.llm)
        app.state.limits = {**_DEFAULT_LIMITS, **deploy.get("limits", {})}
        app.state.metadata = ModelMetadata(
            api_version=api_cfg.get("version", "0.0.0"),
            sentiment_model=settings.model.sentiment.model_name,
            llm_model=settings.llm.model,
            git_sha=os.environ.get("GIT_SHA", "desconhecido"),
        )
        yield

    app = FastAPI(
        title=api_cfg.get("title", "Tweets Emoção & Tópicos — LLM Local"),
        version=api_cfg.get("version", "0.0.0"),
        description=(
            "API local: Python calcula (sentimento/tópicos) e o LLM apenas explica. "
            "Nenhum dado sai da máquina."
        ),
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=api_cfg.get("cors_allow_origins", ["http://127.0.0.1"]),
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    app.include_router(health.router)
    app.include_router(sentiment.router)
    app.include_router(summary.router)
    return app


app = create_app()
