"""Avaliação rigorosa do classificador de sentimento.

Módulos
-------
classification
    Métricas com incerteza (bootstrap), relatório por classe e avaliação por
    fatia (slice-based) — nunca um único número solto.
"""

from src.evaluation.classification import (
    ClassificationReport,
    bootstrap_f1_macro,
    evaluate_by_slice,
    evaluate_classification,
)

__all__ = [
    "ClassificationReport",
    "bootstrap_f1_macro",
    "evaluate_by_slice",
    "evaluate_classification",
]
