"""Testes do summarizer com um cliente Ollama falso (sem rede)."""

from __future__ import annotations

from src.config.settings import LLMSettings
from src.llm.prompts import SYSTEM_PROMPT, build_messages
from src.llm.summarizer import InsightSummarizer
from src.schemas.insights import InsightsReport, SentimentDistribution


def _make_report() -> InsightsReport:
    """Cria um InsightsReport mínimo para os testes."""
    dist = SentimentDistribution(
        total=3,
        counts={"Positivo": 2, "Negativo": 1, "Neutro": 0},
        proportions={"Positivo": 0.67, "Negativo": 0.33, "Neutro": 0.0},
    )
    return InsightsReport(total_tweets=3, overall_sentiment=dist, data_hash="abc123")


class _FakeClient:
    """Cliente Ollama falso que devolve um resumo fixo e registra as mensagens."""

    def __init__(self) -> None:
        self.received: list[dict[str, str]] = []

    def chat(self, messages: list[dict[str, str]]) -> str:
        self.received = messages
        return "## Resumo\nA maioria dos tweets é positiva."


def test_build_messages_includes_system_and_json():
    """As mensagens contêm o prompt de sistema e o JSON do relatório."""
    messages = build_messages(_make_report())
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == SYSTEM_PROMPT
    assert "total_tweets" in messages[1]["content"]


def test_summarizer_returns_response_with_model_and_hash():
    """O summarizer devolve o markdown, o modelo e o hash de origem."""
    settings = LLMSettings(model="llama3.1:8b")
    summarizer = InsightSummarizer(settings, client=_FakeClient())  # type: ignore[arg-type]
    response = summarizer.summarize(_make_report())
    assert "positiva" in response.summary_markdown
    assert response.model == "llama3.1:8b"
    assert response.source_report_hash == "abc123"
