"""Agregação de resultados em insights estruturados.

Este é o coração do "Python calcula": consolida as predições de sentimento e a
atribuição de tópicos em um :class:`~src.schemas.insights.InsightsReport` — o
JSON que será a única fonte de fatos do LLM.

Módulos
-------
insights
    Constrói o ``InsightsReport`` a partir dos DataFrames de predição.
"""

from src.analysis.insights import build_insights_report, compute_distribution

__all__ = [
    "build_insights_report",
    "compute_distribution",
]
