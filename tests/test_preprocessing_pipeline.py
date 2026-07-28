"""Testes do pipeline de pré-processamento (entrada -> saída validada)."""

from __future__ import annotations

import polars as pl

from src.constants.columns import ProcessedColumns
from src.preprocessing.pipeline import preprocess_tweets


def test_preprocess_produces_valid_processed_schema(raw_df: pl.DataFrame):
    """A saída deve conter texto limpo, rótulo canônico e id de sentimento."""
    processed = preprocess_tweets(raw_df, min_length=3)
    assert ProcessedColumns.TEXT_CLEAN in processed.columns
    assert ProcessedColumns.SENTIMENT_ID in processed.columns
    assert set(processed[ProcessedColumns.SENTIMENT_ID].to_list()).issubset({0, 1, 2})


def test_preprocess_removes_leakage_from_clean_text(raw_df: pl.DataFrame):
    """Nenhum emoticon rotulador pode sobreviver na coluna limpa."""
    processed = preprocess_tweets(raw_df, min_length=1)
    joined = " ".join(processed[ProcessedColumns.TEXT_CLEAN].to_list())
    assert ":)" not in joined
    assert ":(" not in joined


def test_preprocess_drops_short_noise():
    """Tweets que viram ruído curto após a limpeza são descartados."""
    tweets_df = pl.DataFrame(
        {
            "id": [1, 2],
            "tweet_text": [":)", "conteúdo real e relevante"],
            "tweet_date": ["x", "y"],
            "sentiment": ["Positivo", "Neutro"],
            "query_used": [":)", "tema"],
        }
    )
    processed = preprocess_tweets(tweets_df, min_length=3)
    assert processed.height == 1
