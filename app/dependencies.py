"""Injeção de dependências da API (componentes criados uma vez no ``lifespan``).

Os componentes pesados (classificador, summarizer) são instanciados no startup e
guardados em ``app.state``; as dependências abaixo apenas os expõem aos
roteadores. Nada é carregado dentro de um endpoint.
"""

from __future__ import annotations

from fastapi import Request

from app.schemas import ModelMetadata
from src.config.settings import Settings
from src.llm.summarizer import InsightSummarizer
from src.models.sentiment import SentimentClassifier


def get_settings(request: Request) -> Settings:
    """Retorna a configuração validada carregada no startup.

    Parameters
    ----------
    request : Request
        Requisição corrente (dá acesso a ``app.state``).

    Returns
    -------
    Settings
        Configuração do projeto.
    """
    return request.app.state.settings


def get_classifier(request: Request) -> SentimentClassifier:
    """Retorna o classificador de sentimento criado no startup.

    Parameters
    ----------
    request : Request
        Requisição corrente.

    Returns
    -------
    SentimentClassifier
        Classificador (carregado no startup quando ``preload_model`` é verdadeiro).
    """
    return request.app.state.classifier


def get_summarizer(request: Request) -> InsightSummarizer:
    """Retorna o summarizer (Ollama local) criado no startup.

    Parameters
    ----------
    request : Request
        Requisição corrente.

    Returns
    -------
    InsightSummarizer
        Componente de resumo baseado no Ollama local.
    """
    return request.app.state.summarizer


def get_metadata(request: Request) -> ModelMetadata:
    """Retorna os metadados de versão do serviço montados no startup.

    Parameters
    ----------
    request : Request
        Requisição corrente.

    Returns
    -------
    ModelMetadata
        Versões da API, dos modelos e do commit.
    """
    return request.app.state.metadata
