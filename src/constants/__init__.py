"""Constantes, enums e valores padrão do projeto.

Módulos
-------
columns
    Nomes canônicos das colunas dos datasets de tweets.
labels
    Rótulos de sentimento e mapeamentos (texto <-> id) da supervisão distante.
regex
    Padrões de limpeza de tweets (URLs, menções, hashtags, emoticons).
"""

from src.constants.columns import ProcessedColumns, RawColumns
from src.constants.labels import (
    ID_TO_SENTIMENT,
    SENTIMENT_LABELS,
    SENTIMENT_TO_ID,
    Sentiment,
    normalize_sentiment,
)

__all__ = [
    "ID_TO_SENTIMENT",
    "SENTIMENT_LABELS",
    "SENTIMENT_TO_ID",
    "ProcessedColumns",
    "RawColumns",
    "Sentiment",
    "normalize_sentiment",
]
