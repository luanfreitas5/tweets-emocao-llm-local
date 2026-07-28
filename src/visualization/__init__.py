"""Utilitários de visualização com paleta consistente.

Módulos
-------
theme
    Paleta e estilo compartilhados do projeto.
distributions
    Gráficos de distribuição de sentimento.
confusion_matrix
    Matriz de confusão do classificador.
"""

from src.visualization.confusion_matrix import plot_confusion_matrix
from src.visualization.distributions import plot_sentiment_distribution
from src.visualization.theme import SENTIMENT_COLORS, apply_theme

__all__ = [
    "SENTIMENT_COLORS",
    "apply_theme",
    "plot_confusion_matrix",
    "plot_sentiment_distribution",
]
