"""Pipeline de resumo: predições -> InsightsReport (JSON) -> resumo do LLM."""

from __future__ import annotations

import logging

import polars as pl

from src.analysis.insights import build_insights_report
from src.config.paths import ProjectPaths
from src.config.settings import Settings
from src.llm.summarizer import InsightSummarizer
from src.schemas.insights import InsightsReport, SummaryResponse
from src.utils.hashing import hash_dataframe
from src.utils.io import read_parquet, write_json
from src.utils.timing import timed

logger = logging.getLogger(__name__)


def run_summarization(
    settings: Settings,
    paths: ProjectPaths,
    sentiment_df: pl.DataFrame | None = None,
    top_terms: dict[int, list[str]] | None = None,
) -> SummaryResponse:
    """Agrega os insights (Python) e gera o resumo em linguagem simples (LLM).

    Parameters
    ----------
    settings : Settings
        Configuração validada (inclui o LLM local).
    paths : ProjectPaths
        Caminhos do projeto.
    sentiment_df : pl.DataFrame | None, optional
        DataFrame com sentimento (e tópicos); se ``None``, lê do parquet de
        tópicos ou de predições.
    top_terms : dict[int, list[str]] | None, optional
        Mapa de termos por tópico (do BERTopic).

    Returns
    -------
    SummaryResponse
        Resumo gerado; o JSON estruturado é salvo em ``paths.insights_json`` e o
        resumo em ``paths.summary_markdown``.
    """
    if sentiment_df is None:
        source = (
            paths.topic_assignments
            if paths.topic_assignments.exists()
            else paths.sentiment_predictions
        )
        sentiment_df = read_parquet(source)

    with timed("agregação de insights (Python calcula)"):
        report: InsightsReport = build_insights_report(
            sentiment_df, top_terms_by_topic=top_terms, data_hash=hash_dataframe(sentiment_df)
        )
        write_json(report, paths.insights_json)
        logger.info("JSON de insights salvo em %s", paths.insights_json)

    with timed("geração do resumo (LLM explica)"):
        summarizer = InsightSummarizer(settings.llm)
        response = summarizer.summarize(report)
        paths.summary_markdown.parent.mkdir(parents=True, exist_ok=True)
        paths.summary_markdown.write_text(response.summary_markdown, encoding="utf-8")
        logger.info("Resumo salvo em %s", paths.summary_markdown)

    return response
