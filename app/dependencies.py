"""Injeção de dependências da API (settings e componentes cacheados).

Usa ``lru_cache`` para instanciar os componentes pesados (classificador,
summarizer) uma única vez por processo.
"""

from __future__ import annotations

from functools import lru_cache

from src.config.settings import Settings, load_settings
from src.llm.summarizer import InsightSummarizer
from src.models.sentiment import SentimentClassifier


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Carrega e cacheia a configuração do projeto.

    Returns
    -------
    Settings
        Configuração validada (instância única por processo).
    """
    return load_settings()


@lru_cache(maxsize=1)
def get_classifier() -> SentimentClassifier:
    """Instancia (uma vez) o classificador de sentimento.

    Returns
    -------
    SentimentClassifier
        Classificador pronto para inferência (carregamento lazy do modelo).
    """
    return SentimentClassifier(get_settings().model.sentiment)


@lru_cache(maxsize=1)
def get_summarizer() -> InsightSummarizer:
    """Instancia (uma vez) o summarizer que fala com o LLM local.

    Returns
    -------
    InsightSummarizer
        Componente de resumo baseado no Ollama local.
    """
    return InsightSummarizer(get_settings().llm)
