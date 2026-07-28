"""Contratos de dados (pandera) e modelos Pydantic de I/O do LLM.

Módulos
-------
raw
    Contrato do CSV bruto de tweets (``RawTweetsSchema``).
processed
    Contrato do parquet processado (``ProcessedTweetsSchema``).
insights
    Modelos Pydantic v2 do JSON estruturado que alimenta o LLM e do resumo
    gerado — a fronteira formal entre "Python calcula" e "LLM explica".
"""

from src.schemas.insights import (
    InsightsReport,
    SentimentDistribution,
    SummaryResponse,
    TopicInsight,
)
from src.schemas.processed import ProcessedTweetsSchema, validate_processed
from src.schemas.raw import RawTweetsSchema, validate_raw

__all__ = [
    "InsightsReport",
    "ProcessedTweetsSchema",
    "RawTweetsSchema",
    "SentimentDistribution",
    "SummaryResponse",
    "TopicInsight",
    "validate_processed",
    "validate_raw",
]
