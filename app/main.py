"""Instância da aplicação FastAPI e registro dos roteadores."""

from __future__ import annotations

from pathlib import Path

import yaml
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import health, sentiment, summary
from src.config.logging import configure_logging
from src.config.paths import CONFIGS_DIR

_DEPLOY_YAML: Path = CONFIGS_DIR / "deploy.yaml"


def _load_deploy_config() -> dict:
    """Carrega ``configs/deploy.yaml`` (título, versão, CORS)."""
    if not _DEPLOY_YAML.exists():
        return {}
    with _DEPLOY_YAML.open(encoding="utf-8") as handler:
        return yaml.safe_load(handler) or {}


def create_app() -> FastAPI:
    """Cria e configura a aplicação FastAPI.

    Returns
    -------
    FastAPI
        Aplicação com CORS restrito a localhost e roteadores registrados.
    """
    configure_logging()
    deploy = _load_deploy_config().get("api", {})

    app = FastAPI(
        title=deploy.get("title", "Tweets Emoção & Tópicos — LLM Local"),
        version=deploy.get("version", "0.1.0"),
        description=(
            "API local: Python calcula (sentimento/tópicos) e o LLM apenas explica. "
            "Nenhum dado sai da máquina."
        ),
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=deploy.get("cors_allow_origins", ["http://127.0.0.1"]),
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    app.include_router(health.router)
    app.include_router(sentiment.router)
    app.include_router(summary.router)
    return app


app = create_app()
