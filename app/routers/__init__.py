"""Roteadores da API, organizados por recurso.

Módulos
-------
health
    Verificação de saúde do serviço.
sentiment
    Classificação de sentimento de textos avulsos.
summary
    Geração de resumo a partir de um ``InsightsReport`` estruturado.
"""

from app.routers import health, sentiment, summary

__all__ = ["health", "sentiment", "summary"]
