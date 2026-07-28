"""Testes comportamentais do classificador de sentimento (sem baixar modelo).

Injeta um pipeline falso em ``_pipeline`` para exercitar o mapeamento canônico e
o comportamento em lote sem depender de rede/pesos.
"""

from __future__ import annotations

from src.config.settings import SentimentParams
from src.models.sentiment import SentimentClassifier


def _make_classifier() -> SentimentClassifier:
    """Cria um classificador com pipeline falso já carregado."""
    params = SentimentParams(
        model_name="fake-model",
        label_map={"positive": "Positivo", "negative": "Negativo", "neutral": "Neutro"},
    )
    clf = SentimentClassifier(params)

    def fake_pipeline(texts, batch_size=32):
        mapping = {
            "bom": {"label": "positive", "score": 0.9},
            "ruim": {"label": "negative", "score": 0.8},
        }
        return [mapping.get(t, {"label": "neutral", "score": 0.5}) for t in texts]

    clf._pipeline = fake_pipeline  # type: ignore[assignment]
    return clf


def test_predict_maps_labels_to_canonical():
    """Os rótulos do modelo são convertidos para a forma canônica do projeto."""
    clf = _make_classifier()
    preds = clf.predict(["bom", "ruim", "neutro"])
    assert [p.label for p in preds] == ["Positivo", "Negativo", "Neutro"]


def test_predict_empty_returns_empty():
    """Lote vazio retorna lista vazia sem carregar modelo."""
    clf = _make_classifier()
    assert clf.predict([]) == []


def test_predict_scores_are_floats_in_range():
    """Os scores estão em [0, 1] e são floats."""
    clf = _make_classifier()
    preds = clf.predict(["bom", "ruim"])
    assert all(isinstance(p.score, float) and 0.0 <= p.score <= 1.0 for p in preds)
