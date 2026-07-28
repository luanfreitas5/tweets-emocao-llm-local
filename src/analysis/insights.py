"""Construção do ``InsightsReport`` a partir das predições (Python calcula).

Todas as estatísticas — distribuições, contagens por tópico, exemplos — são
calculadas aqui em Python, de forma determinística. O resultado é o JSON que
alimenta o LLM.
"""

from __future__ import annotations

import logging

import polars as pl

from src.constants.columns import ProcessedColumns
from src.constants.labels import SENTIMENT_LABELS
from src.schemas.insights import (
    InsightsReport,
    SentimentDistribution,
    TopicInsight,
)

logger = logging.getLogger(__name__)


def compute_distribution(df: pl.DataFrame, sentiment_col: str) -> SentimentDistribution:
    """Calcula a distribuição de sentimento de um DataFrame.

    Parameters
    ----------
    df : pl.DataFrame
        DataFrame com a coluna de sentimento.
    sentiment_col : str
        Nome da coluna de sentimento (ex.: ``sentiment_pred``).

    Returns
    -------
    SentimentDistribution
        Contagens e proporções por rótulo canônico.

    Examples
    --------
    >>> dist = compute_distribution(df, "sentiment_pred")  # doctest: +SKIP
    """
    total = df.height
    counts_df = df.group_by(sentiment_col).len()
    raw_counts = dict(
        zip(counts_df[sentiment_col].to_list(), counts_df["len"].to_list(), strict=True)
    )

    counts = {label: int(raw_counts.get(label, 0)) for label in SENTIMENT_LABELS}
    proportions = {
        label: round(count / total, 4) if total else 0.0 for label, count in counts.items()
    }
    return SentimentDistribution(total=total, counts=counts, proportions=proportions)


def _build_topic_insight(
    group: pl.DataFrame,
    topic_id: int,
    sentiment_col: str,
    top_terms: list[str],
    n_examples: int,
) -> TopicInsight:
    """Monta o :class:`TopicInsight` de um único tópico."""
    label = top_terms[0] if top_terms else f"tópico {topic_id}"
    examples = group[ProcessedColumns.TEXT_CLEAN].head(n_examples).to_list()
    return TopicInsight(
        topic_id=topic_id,
        label=label,
        size=group.height,
        top_terms=top_terms,
        sentiment=compute_distribution(group, sentiment_col),
        example_texts=examples,
    )


def build_insights_report(
    df: pl.DataFrame,
    sentiment_col: str = ProcessedColumns.SENTIMENT_PRED,
    topic_col: str = ProcessedColumns.TOPIC_ID,
    top_terms_by_topic: dict[int, list[str]] | None = None,
    top_k_topics: int = 10,
    n_examples: int = 3,
    data_hash: str | None = None,
) -> InsightsReport:
    """Consolida predições em um relatório estruturado para o LLM.

    Parameters
    ----------
    df : pl.DataFrame
        DataFrame com sentimento previsto e (opcionalmente) tópico atribuído.
    sentiment_col : str, optional
        Coluna de sentimento previsto, by default ``sentiment_pred``.
    topic_col : str, optional
        Coluna de id do tópico, by default ``topic_id``.
    top_terms_by_topic : dict[int, list[str]] | None, optional
        Termos por tópico (do BERTopic); ``None`` se tópicos ausentes.
    top_k_topics : int, optional
        Número de tópicos maiores a incluir, by default 10.
    n_examples : int, optional
        Exemplos por tópico, by default 3.
    data_hash : str | None, optional
        Hash do dataset de origem (rastreabilidade), by default None.

    Returns
    -------
    InsightsReport
        Relatório estruturado (única fonte de fatos do LLM).
    """
    overall = compute_distribution(df, sentiment_col)

    topics: list[TopicInsight] = []
    if topic_col in df.columns:
        terms = top_terms_by_topic or {}
        sizes = (
            df.filter(pl.col(topic_col) != -1)
            .group_by(topic_col)
            .len()
            .sort("len", descending=True)
            .head(top_k_topics)
        )
        for topic_id in sizes[topic_col].to_list():
            group = df.filter(pl.col(topic_col) == topic_id)
            topics.append(
                _build_topic_insight(
                    group, int(topic_id), sentiment_col, terms.get(int(topic_id), []), n_examples
                )
            )

    logger.info("Relatório de insights: %d tweets, %d tópicos", df.height, len(topics))
    return InsightsReport(
        total_tweets=df.height,
        overall_sentiment=overall,
        topics=topics,
        data_hash=data_hash,
    )
