"""Testes da API FastAPI (pulados se o extra 'api' não estiver instalado)."""

from __future__ import annotations

import pytest
from app.main import create_app
from fastapi.testclient import TestClient

pytest.importorskip("fastapi")


@pytest.fixture
def client() -> TestClient:
    """Cliente de teste da aplicação."""
    return TestClient(create_app())


def test_health_endpoint_ok(client: TestClient):
    """O endpoint /health responde 200 e informa o modelo LLM."""
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "llm_model" in body
