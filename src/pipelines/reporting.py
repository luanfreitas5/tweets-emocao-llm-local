"""Pipeline de figuras: predições -> gráficos em ``reports/figures``."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl

from src.analysis.insights import compute_distribution
from src.config.paths import ProjectPaths
from src.constants.columns import ProcessedColumns
from src.utils.io import read_parquet
from src.utils.timing import timed
from src.visualization.confusion_matrix import plot_confusion_matrix
from src.visualization.distributions import plot_sentiment_distribution

logger = logging.getLogger(__name__)


def run_reporting(
    paths: ProjectPaths,
    sentiment_df: pl.DataFrame | None = None,
) -> list[Path]:
    """Gera e salva as figuras do projeto (``.png`` 300 dpi + ``.svg``).

    Parameters
    ----------
    paths : ProjectPaths
        Caminhos do projeto (``paths.figures`` é o destino).
    sentiment_df : pl.DataFrame | None, optional
        DataFrame com ``sentiment_pred``; se ``None``, lê do parquet de tópicos
        ou de predições. A matriz de confusão só é gerada quando a coluna de
        rótulo de referência (``sentiment``) está presente.

    Returns
    -------
    list[pathlib.Path]
        Caminhos base (sem extensão) das figuras salvas.

    Examples
    --------
    >>> saved = run_reporting(paths)  # doctest: +SKIP
    """
    if sentiment_df is None:
        source = (
            paths.topic_assignments
            if paths.topic_assignments.exists()
            else paths.sentiment_predictions
        )
        sentiment_df = read_parquet(source)

    # Backend sem janela: o pipeline roda em terminal/CI e só grava arquivos.
    plt.switch_backend("Agg")
    saved: list[Path] = []

    with timed("geração de figuras"):
        distribution = compute_distribution(sentiment_df, ProcessedColumns.SENTIMENT_PRED)
        target = paths.figures / "distribuicao_sentimento"
        fig = plot_sentiment_distribution(distribution, save_path=target)
        plt.close(fig)
        saved.append(target)

        if ProcessedColumns.SENTIMENT in sentiment_df.columns:
            target = paths.figures / "matriz_confusao"
            fig = plot_confusion_matrix(
                sentiment_df[ProcessedColumns.SENTIMENT].to_numpy(),
                sentiment_df[ProcessedColumns.SENTIMENT_PRED].to_numpy(),
                save_path=target,
            )
            plt.close(fig)
            saved.append(target)
        else:
            logger.warning("Coluna de rótulo de referência ausente; matriz de confusão ignorada.")

    logger.info("%d figuras salvas em %s", len(saved), paths.figures)
    return saved
