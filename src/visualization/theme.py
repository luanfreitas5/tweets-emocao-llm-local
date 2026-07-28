"""Paleta e estilo compartilhados do projeto.

Cores semânticas fixas por sentimento garantem consistência entre todos os
gráficos gerados.
"""

from __future__ import annotations

from typing import Final

import matplotlib.pyplot as plt
import seaborn as sns

#: Cores semânticas por rótulo de sentimento (reutilizadas em todo o projeto).
SENTIMENT_COLORS: Final[dict[str, str]] = {
    "Positivo": "#2ca02c",
    "Negativo": "#d62728",
    "Neutro": "#7f7f7f",
}

FIGURE_DPI: Final = 120


def apply_theme() -> None:
    """Aplica o tema visual padrão (seaborn ``whitegrid`` + DPI do projeto).

    Examples
    --------
    >>> apply_theme()
    """
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams["figure.dpi"] = FIGURE_DPI
    plt.rcParams["axes.titlesize"] = 13
    plt.rcParams["axes.titleweight"] = "bold"
