"""Pipeline de avaliação: predições (gold) -> métricas com IC 95% e por fatia."""

from __future__ import annotations

import logging
from dataclasses import replace

import numpy as np
import polars as pl

from src.config.paths import ProjectPaths
from src.config.settings import Settings
from src.constants.columns import ProcessedColumns
from src.evaluation.classification import (
    ClassificationReport,
    evaluate_by_slice,
    evaluate_classification,
)
from src.experiment.tracking import log_evaluation
from src.utils.io import read_parquet, write_json
from src.utils.timing import timed

logger = logging.getLogger(__name__)

#: Fatias por tamanho do tweet (nº de palavras): (rótulo, mínimo, máximo).
_LENGTH_SLICES: tuple[tuple[str, int, int | None], ...] = (
    ("curto_1_5", 1, 5),
    ("medio_6_15", 6, 15),
    ("longo_16_mais", 16, None),
)


def _build_length_slices(texts: pl.Series) -> np.ndarray:
    """Atribui cada tweet a uma faixa de comprimento (palavras do texto limpo)."""
    words = texts.str.count_matches(r"\S+").to_numpy()
    slices = np.full(len(words), _LENGTH_SLICES[0][0], dtype=object)
    for name, low, high in _LENGTH_SLICES:
        mask = words >= low if high is None else (words >= low) & (words <= high)
        slices[mask] = name
    return slices


def run_evaluation(
    settings: Settings,
    paths: ProjectPaths,
    sentiment_df: pl.DataFrame | None = None,
) -> ClassificationReport | None:
    """Avalia o classificador contra os rótulos de referência e grava as métricas.

    A métrica principal é o F1-macro com IC 95% (bootstrap); também reporta
    precisão/recall/F1 por classe e o F1-macro por faixa de comprimento do tweet.
    Os rótulos de referência vêm de supervisão distante (emoticons): a métrica
    mede concordância com esse rótulo, não com um padrão-ouro humano.

    Parameters
    ----------
    settings : Settings
        Configuração validada (iterações de bootstrap e semente).
    paths : ProjectPaths
        Caminhos do projeto (``paths.classification_metrics`` é o destino).
    sentiment_df : pl.DataFrame | None, optional
        DataFrame com ``sentiment`` e ``sentiment_pred``; se ``None``, lê do
        parquet de tópicos ou de predições (camada gold).

    Returns
    -------
    ClassificationReport | None
        Métricas consolidadas, ou ``None`` se não houver rótulo de referência.

    Examples
    --------
    >>> report = run_evaluation(settings, paths)  # doctest: +SKIP
    """
    if sentiment_df is None:
        source = (
            paths.topic_assignments
            if paths.topic_assignments.exists()
            else paths.sentiment_predictions
        )
        sentiment_df = read_parquet(source)

    if ProcessedColumns.SENTIMENT not in sentiment_df.columns:
        logger.warning("Coluna de rótulo de referência ausente; avaliação ignorada.")
        return None

    y_true = sentiment_df[ProcessedColumns.SENTIMENT].to_numpy()
    y_pred = sentiment_df[ProcessedColumns.SENTIMENT_PRED].to_numpy()
    params = settings.model.evaluation

    with timed("avaliação do classificador"):
        report = evaluate_classification(
            y_true, y_pred, params.bootstrap_iterations, settings.random_seed
        )
        slices = _build_length_slices(sentiment_df[ProcessedColumns.TEXT_CLEAN])
        report = replace(report, per_slice=evaluate_by_slice(y_true, y_pred, slices))

    write_json(
        {
            "n_evaluated": len(y_true),
            "primary_metric": params.primary_metric,
            "f1_macro": report.f1_macro,
            "f1_macro_ci95": list(report.f1_ci95),
            "ci_method": f"bootstrap, {params.bootstrap_iterations} reamostragens",
            "seed": settings.random_seed,
            "per_class": report.per_class,
            "per_slice_f1_macro": report.per_slice,
        },
        paths.classification_metrics,
    )
    logger.info("Métricas gravadas em %s", paths.classification_metrics)
    log_evaluation(report, len(y_true))
    return report
