"""Orquestração completa do fluxo ponta a ponta."""

from __future__ import annotations

import logging

from src.config.paths import ProjectPaths
from src.config.settings import Settings
from src.pipelines.preprocessing import run_preprocessing
from src.pipelines.sentiment import run_sentiment
from src.pipelines.summarization import run_summarization
from src.pipelines.topics import run_topics
from src.schemas.insights import SummaryResponse
from src.utils.timing import timed

logger = logging.getLogger(__name__)


def run_full_pipeline(
    settings: Settings,
    paths: ProjectPaths,
    *,
    with_topics: bool = True,
) -> SummaryResponse:
    """Executa o fluxo completo: limpeza -> sentimento -> tópicos -> resumo.

    Parameters
    ----------
    settings : Settings
        Configuração validada do projeto.
    paths : ProjectPaths
        Caminhos do projeto (diretórios criados automaticamente).
    with_topics : bool, optional
        Inclui a etapa de modelagem de tópicos, by default True. Desative para
        execuções rápidas (apenas sentimento + resumo).

    Returns
    -------
    SummaryResponse
        Resumo final gerado pelo LLM local.
    """
    paths.ensure_directories()
    with timed("pipeline completo"):
        processed = run_preprocessing(settings, paths)
        with_sentiment = run_sentiment(settings, paths, processed)

        top_terms: dict[int, list[str]] | None = None
        df = with_sentiment
        if with_topics:
            df, top_terms = run_topics(settings, paths, with_sentiment)

        response = run_summarization(settings, paths, df, top_terms)
    logger.info("Pipeline concluído com sucesso.")
    return response
