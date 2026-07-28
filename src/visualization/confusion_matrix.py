"""Matriz de confusão do classificador de sentimento."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure
from numpy.typing import NDArray
from sklearn.metrics import confusion_matrix

from src.constants.labels import SENTIMENT_LABELS
from src.visualization.theme import apply_theme


def plot_confusion_matrix(
    y_true: NDArray,
    y_pred: NDArray,
    normalize: bool = True,
    save_path: Path | None = None,
) -> Figure:
    """Plota a matriz de confusão (opcionalmente normalizada por linha).

    Parameters
    ----------
    y_true : numpy.ndarray
        Rótulos verdadeiros.
    y_pred : numpy.ndarray
        Rótulos previstos.
    normalize : bool, optional
        Normaliza por classe verdadeira (recall visual), by default True.
    save_path : Path | None, optional
        Se informado, salva em ``.png`` (300 dpi) e ``.svg``.

    Returns
    -------
    matplotlib.figure.Figure
        Figura gerada.
    """
    apply_theme()
    labels = list(SENTIMENT_LABELS)
    matrix = confusion_matrix(
        y_true, y_pred, labels=labels, normalize="true" if normalize else None
    )

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".2f" if normalize else "d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_title("Matriz de Confusão")
    ax.set_xlabel("Previsto")
    ax.set_ylabel("Verdadeiro")
    fig.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path.with_suffix(".png"), dpi=300)
        fig.savefig(save_path.with_suffix(".svg"))
    return fig
