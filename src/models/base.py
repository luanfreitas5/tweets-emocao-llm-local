"""Contrato base para classificadores de texto."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class Prediction:
    """Resultado de uma predição de classificação.

    Attributes
    ----------
    label : str
        Rótulo previsto (canônico).
    score : float
        Confiança/probabilidade associada (0-1).
    """

    label: str
    score: float


@runtime_checkable
class TextClassifier(Protocol):
    """Protocolo de um classificador de texto.

    Qualquer implementação deve expor ``predict`` (lote de textos -> lote de
    predições), permitindo baixo acoplamento entre pipeline e modelo concreto.
    """

    def predict(self, texts: list[str]) -> list[Prediction]:
        """Classifica um lote de textos.

        Parameters
        ----------
        texts : list[str]
            Textos a classificar.

        Returns
        -------
        list[Prediction]
            Uma predição por texto de entrada.
        """
        ...
