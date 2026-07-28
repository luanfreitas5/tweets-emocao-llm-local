"""Nomes canônicos das colunas dos datasets de tweets.

Centraliza os nomes para evitar *strings mágicas* espalhadas pelo código e
manter consistência entre os estágios raw -> processed.
"""

from __future__ import annotations

from typing import Final


class RawColumns:
    """Colunas do CSV bruto (``NoThemeTweets.csv`` e afins).

    A base bruta usa vírgula como separador e rótulos textuais em
    ``sentiment``. A coluna ``query_used`` guarda o emoticon/hashtag que gerou
    o rótulo por supervisão distante — fonte de *leakage* a ser removida.
    """

    ID: Final = "id"
    TEXT: Final = "tweet_text"
    DATE: Final = "tweet_date"
    SENTIMENT: Final = "sentiment"
    QUERY_USED: Final = "query_used"

    ALL: Final = (ID, TEXT, DATE, SENTIMENT, QUERY_USED)


class ProcessedColumns:
    """Colunas do dataset processado (parquet) após a limpeza."""

    ID: Final = "id"
    TEXT_RAW: Final = "tweet_text"
    TEXT_CLEAN: Final = "text_clean"
    DATE: Final = "tweet_date"
    SENTIMENT: Final = "sentiment"
    SENTIMENT_ID: Final = "sentiment_id"
    # Preenchidas em etapas posteriores do pipeline.
    SENTIMENT_PRED: Final = "sentiment_pred"
    SENTIMENT_SCORE: Final = "sentiment_score"
    TOPIC_ID: Final = "topic_id"
    TOPIC_LABEL: Final = "topic_label"
