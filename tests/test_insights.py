"""Testes da agregação de insights (Python calcula)."""

from __future__ import annotations

import polars as pl

from src.analysis.insights import build_insights_report, compute_distribution
from src.constants.columns import ProcessedColumns


def test_compute_distribution_sums_to_total(predicted_df: pl.DataFrame):
    """As contagens por rótulo devem somar o total de tweets."""
    dist = compute_distribution(predicted_df, ProcessedColumns.SENTIMENT_PRED)
    assert dist.total == predicted_df.height
    assert sum(dist.counts.values()) == predicted_df.height


def test_compute_distribution_proportions_between_zero_and_one(predicted_df: pl.DataFrame):
    """As proporções devem estar no intervalo [0, 1]."""
    dist = compute_distribution(predicted_df, ProcessedColumns.SENTIMENT_PRED)
    assert all(0.0 <= p <= 1.0 for p in dist.proportions.values())


def test_build_insights_report_orders_topics_by_size(predicted_df: pl.DataFrame):
    """O relatório deve conter os tópicos e o total correto de tweets."""
    report = build_insights_report(
        predicted_df,
        top_terms_by_topic={0: ["filme"], 1: ["jogo"], 2: ["comida"]},
    )
    assert report.total_tweets == predicted_df.height
    assert len(report.topics) == 3
    sizes = [topic.size for topic in report.topics]
    assert sizes == sorted(sizes, reverse=True)


def test_dominant_returns_majority_label(predicted_df: pl.DataFrame):
    """O rótulo dominante deve ser o de maior contagem."""
    dist = compute_distribution(predicted_df, ProcessedColumns.SENTIMENT_PRED)
    assert dist.dominant() == "Positivo"
