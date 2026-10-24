"""Testes da API FastAPI (pulados se o extra 'api' não estiver instalado).

Usa um classificador falso e determinístico: nenhum modelo do Hugging Face é
baixado. O ``TestClient`` roda dentro de ``with`` para executar o ``lifespan``.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from app.main import create_app  # noqa: E402
from src.models.base import Prediction  # noqa: E402
from src.preprocessing.cleaning import clean_tweet  # noqa: E402


class FakeClassifier:
    """Classificador determinístico: 'adoro' é positivo; o resto, negativo."""

    is_loaded = True

    def predict(self, texts: list[str]) -> list[Prediction]:
        """Devolve uma predição por texto limpo."""
        return [
            Prediction(label="Positivo" if "adoro" in text else "Negativo", score=0.9)
            for text in texts
        ]


@pytest.fixture
def client() -> Iterator[TestClient]:
    """Cliente de teste com o lifespan executado e o classificador substituído."""
    app = create_app(preload_model=False, configure_log=False)
    with TestClient(app) as test_client:
        app.state.classifier = FakeClassifier()
        yield test_client


@pytest.mark.smoke
def test_health_endpoint_ok(client: TestClient):
    """O endpoint /health responde 200 e informa o modelo LLM e o estado do modelo."""
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "llm_model" in body
    assert body["model_loaded"] is True


def test_version_endpoint_reports_models(client: TestClient):
    """O endpoint /version informa API, modelos e commit."""
    body = client.get("/version").json()
    assert set(body) == {"api_version", "sentiment_model", "llm_model", "git_sha"}


def test_sentiment_matches_classifier_on_cleaned_texts(client: TestClient):
    """Paridade: a API devolve o mesmo que o classificador aplicado ao texto limpo."""
    texts = ["Adoro isso :)", "que dia horrível"]
    response = client.post("/sentiment", json={"texts": texts})
    assert response.status_code == 200
    expected = FakeClassifier().predict([clean_tweet(text) for text in texts])
    got = response.json()["results"]
    assert [item["label"] for item in got] == [p.label for p in expected]
    assert [item["score"] for item in got] == [p.score for p in expected]


@pytest.mark.parametrize(
    "payload",
    [
        {"texts": []},  # lote vazio
        {"texts": ["ok"], "extra": 1},  # campo desconhecido (extra="forbid")
        {"texts": "não é lista"},  # tipo inválido
        {},  # campo obrigatório ausente
    ],
)
def test_sentiment_rejects_invalid_payload(client: TestClient, payload: dict):
    """Payloads fora do contrato retornam 422."""
    assert client.post("/sentiment", json=payload).status_code == 422


def test_sentiment_enforces_configured_limits(client: TestClient):
    """Lotes e textos acima dos limites de ``deploy.yaml`` retornam 422."""
    limits = client.app.state.limits  # type: ignore[attr-defined]
    too_many = ["ok"] * (limits["max_texts_per_request"] + 1)
    too_long = ["a" * (limits["max_text_length"] + 1)]
    assert client.post("/sentiment", json={"texts": too_many}).status_code == 422
    assert client.post("/sentiment", json={"texts": too_long}).status_code == 422
