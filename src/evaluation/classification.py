"""Métricas de classificação com incerteza e avaliação por fatia.

Segue o "senior bar": nunca reportar uma métrica pontual sem intervalo de
confiança nem sem quebra por subgrupo. A métrica principal é o F1-macro
(classes desbalanceadas, todas importam igualmente).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import classification_report, f1_score

from src.constants.labels import SENTIMENT_LABELS

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ClassificationReport:
    """Resultado consolidado da avaliação de classificação.

    Attributes
    ----------
    f1_macro : float
        F1-macro pontual.
    f1_ci95 : tuple[float, float]
        Intervalo de confiança de 95% (bootstrap) do F1-macro.
    per_class : dict[str, dict[str, float]]
        Relatório por classe (precisão, recall, f1, suporte).
    per_slice : dict[str, float]
        F1-macro por fatia (preenchido por :func:`evaluate_by_slice`).
    """

    f1_macro: float
    f1_ci95: tuple[float, float]
    per_class: dict[str, dict[str, float]]
    per_slice: dict[str, float] = field(default_factory=dict)


def bootstrap_f1_macro(
    y_true: NDArray,
    y_pred: NDArray,
    iterations: int = 1000,
    seed: int = 42,
) -> tuple[float, float]:
    """Estima o intervalo de confiança de 95% do F1-macro por bootstrap.

    Parameters
    ----------
    y_true : numpy.ndarray
        Rótulos verdadeiros.
    y_pred : numpy.ndarray
        Rótulos previstos.
    iterations : int, optional
        Número de reamostragens, by default 1000.
    seed : int, optional
        Semente do gerador, by default 42.

    Returns
    -------
    tuple[float, float]
        Limites inferior e superior do IC 95%.

    Examples
    --------
    >>> lo, hi = bootstrap_f1_macro(y_true, y_pred)  # doctest: +SKIP
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    scores = np.empty(iterations, dtype=np.float64)
    for i in range(iterations):
        idx = rng.integers(0, n, size=n)
        scores[i] = f1_score(y_true[idx], y_pred[idx], average="macro", zero_division=0)  # pyright: ignore[reportArgumentType]
    return float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))


def evaluate_classification(
    y_true: NDArray,
    y_pred: NDArray,
    bootstrap_iterations: int = 1000,
    seed: int = 42,
) -> ClassificationReport:
    """Avalia a classificação com F1-macro, IC 95% e relatório por classe.

    Parameters
    ----------
    y_true : numpy.ndarray
        Rótulos verdadeiros (canônicos).
    y_pred : numpy.ndarray
        Rótulos previstos (canônicos).
    bootstrap_iterations : int, optional
        Reamostragens do bootstrap, by default 1000.
    seed : int, optional
        Semente, by default 42.

    Returns
    -------
    ClassificationReport
        Métricas consolidadas (ainda sem fatias).
    """
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))  # pyright: ignore[reportArgumentType]
    ci95 = bootstrap_f1_macro(y_true, y_pred, bootstrap_iterations, seed)
    per_class = classification_report(
        y_true,
        y_pred,
        labels=list(SENTIMENT_LABELS),
        output_dict=True,
        zero_division=0,  # pyright: ignore[reportArgumentType]
    )

    logger.info("F1-macro: %.4f (IC95%%: %.4f-%.4f)", f1_macro, ci95[0], ci95[1])
    return ClassificationReport(f1_macro=f1_macro, f1_ci95=ci95, per_class=per_class)  # type: ignore


def evaluate_by_slice(
    y_true: NDArray,
    y_pred: NDArray,
    slices: NDArray,
) -> dict[str, float]:
    """Calcula o F1-macro por fatia (subgrupo) para revelar falhas ocultas.

    Parameters
    ----------
    y_true : numpy.ndarray
        Rótulos verdadeiros.
    y_pred : numpy.ndarray
        Rótulos previstos.
    slices : numpy.ndarray
        Valor da fatia por amostra (ex.: faixa de comprimento do tweet).

    Returns
    -------
    dict[str, float]
        F1-macro por valor de fatia.
    """
    result: dict[str, float] = {}
    for value in np.unique(slices):
        mask = slices == value
        result[str(value)] = float(
            f1_score(y_true[mask], y_pred[mask], average="macro", zero_division=0)  # pyright: ignore[reportArgumentType]
        )
    logger.info("F1-macro por fatia: %s", result)
    return result
