"""Gráficos de distribuição de sentimento."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from src.constants.labels import SENTIMENT_LABELS
from src.schemas.insights import SentimentDistribution
from src.visualization.theme import SENTIMENT_COLORS, apply_theme


def plot_sentiment_distribution(
    distribution: SentimentDistribution,
    title: str = "Distribuição de Sentimento",
    save_path: Path | None = None,
) -> Figure:
    """Plota a distribuição de sentimento como gráfico de barras.

    Parameters
    ----------
    distribution : SentimentDistribution
        Distribuição calculada em Python.
    title : str, optional
        Título do gráfico.
    save_path : Path | None, optional
        Se informado, salva a figura em ``.png`` (300 dpi) e ``.svg``.

    Returns
    -------
    matplotlib.figure.Figure
        Figura gerada.

    Examples
    --------
    >>> fig = plot_sentiment_distribution(dist)  # doctest: +SKIP
    """
    apply_theme()
    labels = list(SENTIMENT_LABELS)
    values = [distribution.counts.get(label, 0) for label in labels]
    colors = [SENTIMENT_COLORS[label] for label in labels]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(labels, values, color=colors)
    ax.set_title(title)
    ax.set_xlabel("Sentimento")
    ax.set_ylabel("Nº de tweets")
    for i, value in enumerate(values):
        ax.text(i, value, f"{value:,}", ha="center", va="bottom")
    fig.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path.with_suffix(".png"), dpi=300)
        fig.savefig(save_path.with_suffix(".svg"))
    return fig
