"""Rótulos de sentimento e mapeamentos texto <-> id.

A base do Kaggle rotula os tweets por **supervisão distante**: emoticons como
``:)`` / ``:(`` e hashtags como ``#fato`` (na coluna ``query_used``) definem o
rótulo. Nos arquivos de treino/teste o ``sentiment`` é numérico; nos brutos é
textual. Este módulo unifica ambos em uma enumeração canônica.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Final


class Sentiment(IntEnum):
    """Enumeração canônica de sentimento (3 classes).

    A codificação numérica segue a convenção dos datasets ``*3Classes`` do
    Kaggle: ``0`` negativo, ``1`` positivo, ``2`` neutro.
    """

    NEGATIVO = 0
    POSITIVO = 1
    NEUTRO = 2

    @property
    def label(self) -> str:
        """Retorna o rótulo textual canônico (capitalizado)."""
        return self.name.capitalize()


#: Ordem canônica dos rótulos textuais.
SENTIMENT_LABELS: Final[tuple[str, ...]] = tuple(s.label for s in Sentiment)

#: Mapeia rótulo textual -> id inteiro.
SENTIMENT_TO_ID: Final[dict[str, int]] = {s.label: int(s) for s in Sentiment}

#: Mapeia id inteiro -> rótulo textual.
ID_TO_SENTIMENT: Final[dict[int, str]] = {int(s): s.label for s in Sentiment}

#: Sinônimos aceitos na normalização (tolera variações de origem).
_SYNONYMS: Final[dict[str, str]] = {
    "positivo": "Positivo",
    "positive": "Positivo",
    "pos": "Positivo",
    "negativo": "Negativo",
    "negative": "Negativo",
    "neg": "Negativo",
    "neutro": "Neutro",
    "neutral": "Neutro",
    "neu": "Neutro",
}


def normalize_sentiment(value: str | int) -> str:
    """Normaliza um rótulo de sentimento (texto ou id) para a forma canônica.

    Parameters
    ----------
    value : str | int
        Rótulo textual (em pt/en, qualquer caixa) ou id inteiro (0/1/2).

    Returns
    -------
    str
        Rótulo canônico: ``"Positivo"``, ``"Negativo"`` ou ``"Neutro"``.

    Raises
    ------
    ValueError
        Se o valor não corresponder a nenhum rótulo conhecido.

    Examples
    --------
    >>> normalize_sentiment("positive")
    'Positivo'
    >>> normalize_sentiment(0)
    'Negativo'
    """
    if isinstance(value, int) or (isinstance(value, str) and value.lstrip("-").isdigit()):
        as_id = int(value)
        if as_id in ID_TO_SENTIMENT:
            return ID_TO_SENTIMENT[as_id]
        raise ValueError(f"Id de sentimento desconhecido: {value!r}")

    key = value.strip().lower()
    if key in _SYNONYMS:
        return _SYNONYMS[key]
    raise ValueError(f"Rótulo de sentimento desconhecido: {value!r}")
