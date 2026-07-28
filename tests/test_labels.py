"""Testes da normalização de rótulos de sentimento."""

from __future__ import annotations

import pytest

from src.constants.labels import (
    ID_TO_SENTIMENT,
    SENTIMENT_TO_ID,
    Sentiment,
    normalize_sentiment,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("Positivo", "Positivo"),
        ("positive", "Positivo"),
        ("NEG", "Negativo"),
        ("neutral", "Neutro"),
        (0, "Negativo"),
        (1, "Positivo"),
        (2, "Neutro"),
        ("1", "Positivo"),
    ],
)
def test_normalize_sentiment_valid(value, expected):
    """Rótulos textuais e numéricos convergem para a forma canônica."""
    assert normalize_sentiment(value) == expected


@pytest.mark.parametrize("value", ["desconhecido", 9, "-5"])
def test_normalize_sentiment_invalid_raises(value):
    """Valores fora do vocabulário levantam ValueError."""
    with pytest.raises(ValueError):
        normalize_sentiment(value)


def test_id_and_label_maps_are_consistent():
    """Os mapas id<->rótulo são inversos e cobrem as 3 classes."""
    assert set(SENTIMENT_TO_ID.values()) == {0, 1, 2}
    for sentiment in Sentiment:
        assert ID_TO_SENTIMENT[int(sentiment)] == sentiment.label
