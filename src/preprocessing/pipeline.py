"""Aplica a limpeza a um DataFrame e produz o estágio ``processed``.

Valida a entrada (``RawTweetsSchema``) e a saída (``ProcessedTweetsSchema``),
seguindo o princípio de validar nos limites de cada etapa.
"""

from __future__ import annotations

import logging

import polars as pl

from src.constants.columns import ProcessedColumns, RawColumns
from src.constants.labels import SENTIMENT_TO_ID
from src.preprocessing.cleaning import clean_tweet
from src.schemas.processed import validate_processed
from src.schemas.raw import validate_raw

logger = logging.getLogger(__name__)


def preprocess_tweets(df: pl.DataFrame, *, min_length: int = 3) -> pl.DataFrame:
    """Limpa os tweets e monta o DataFrame processado, validando entrada e saída.

    Parameters
    ----------
    df : pl.DataFrame
        DataFrame bruto (deve satisfazer ``RawTweetsSchema``).
    min_length : int, optional
        Comprimento mínimo (em caracteres) do texto limpo para manter a linha,
        by default 3. Remove tweets que viraram ruído após a limpeza.

    Returns
    -------
    pl.DataFrame
        DataFrame processado com colunas de :class:`ProcessedColumns`,
        satisfazendo ``ProcessedTweetsSchema``.

    Raises
    ------
    pandera.errors.SchemaError
        Se a entrada ou a saída violarem seus contratos.

    Examples
    --------
    >>> processed = preprocess_tweets(raw_df)
    """
    validate_raw(df)
    before = df.height

    processed = (
        df.with_columns(
            pl.col(RawColumns.TEXT)
            .map_elements(clean_tweet, return_dtype=pl.Utf8)
            .alias(ProcessedColumns.TEXT_CLEAN),
        )
        .with_columns(
            pl.col(RawColumns.SENTIMENT)
            .replace_strict(SENTIMENT_TO_ID, default=None)
            .alias(ProcessedColumns.SENTIMENT_ID),
        )
        .filter(pl.col(ProcessedColumns.TEXT_CLEAN).str.len_chars() >= min_length)
        .unique(subset=[ProcessedColumns.TEXT_CLEAN], keep="first")
        .select(
            RawColumns.ID,
            RawColumns.TEXT,
            ProcessedColumns.TEXT_CLEAN,
            RawColumns.DATE,
            RawColumns.SENTIMENT,
            ProcessedColumns.SENTIMENT_ID,
        )
    )

    logger.info(
        "Limpeza: %d -> %d tweets (%d removidos por ruído/duplicidade)",
        before,
        processed.height,
        before - processed.height,
    )
    return validate_processed(processed)
