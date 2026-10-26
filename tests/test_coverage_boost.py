"""Testes de loader, seed, cliente Ollama, modelos e pipelines (com dublês).

Nenhum modelo é baixado e nenhum servidor é contatado: dependências pesadas
são substituídas por dublês injetados.
"""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import polars as pl
import pytest

from src.config.paths import ProjectPaths
from src.config.seed import seed_everything
from src.config.settings import LLMSettings, SentimentParams
from src.constants.columns import ProcessedColumns
from src.data.loader import load_raw_tweets
from src.exceptions.data import RawDataError
from src.exceptions.llm import LLMGenerationError, OllamaConnectionError
from src.exceptions.model import ModelInferenceError, ModelLoadError
from src.llm.ollama_client import OllamaClient
from src.models.base import Prediction
from src.models.sentiment import SentimentClassifier, _resolve_device
from src.schemas.insights import SummaryResponse

CSV_COMMA = (
    "id,tweet_text,tweet_date,sentiment,query_used\n"
    "1,que dia lindo :),d,Positivo,:)\n"
    "2,odeio isso :(,d,Negativo,:(\n"
    "3,reuniao amanha,d,Neutro,tema\n"
)


@pytest.fixture
def tmp_paths(tmp_path: Path) -> ProjectPaths:
    """ProjectPaths apontando para um diretório temporário."""
    d = tmp_path
    return ProjectPaths(
        root=d,
        data_bronze=d / "bronze",
        data_silver=d / "silver",
        data_gold=d / "gold",
        bronze_tweets=d / "bronze" / "t.csv",
        silver_tweets=d / "silver" / "t.parquet",
        sentiment_predictions=d / "gold" / "s.parquet",
        topic_assignments=d / "gold" / "topics.parquet",
        insights_json=d / "gold" / "insights.json",
        summary_markdown=d / "gold" / "resumo.md",
        classification_metrics=d / "metrics" / "c.json",
        models_root=d / "models",
        topic_model=d / "models" / "topic",
        reports_root=d / "reports",
        figures=d / "reports" / "figures",
        metrics=d / "reports" / "metrics",
        logs=d / "logs",
        directories=(d / "bronze", d / "silver", d / "gold"),
    )


# ---------------------------------------------------------------- loader


def test_load_raw_tweets_comma_and_sample(tmp_path: Path):
    """Lê CSV com vírgula, normaliza o rótulo e amostra."""
    path = tmp_path / "a.csv"
    path.write_text(CSV_COMMA, encoding="utf-8")
    full = load_raw_tweets(path)
    assert full.height == 3
    assert set(full["sentiment"]) == {"Positivo", "Negativo", "Neutro"}
    assert load_raw_tweets(path, sample_size=2, seed=1).height == 2


def test_load_raw_tweets_semicolon_numeric(tmp_path: Path):
    """Detecta ';' e normaliza rótulo numérico."""
    path = tmp_path / "b.csv"
    path.write_text(
        CSV_COMMA.replace(",", ";").replace("Positivo", "1").replace("Negativo", "0"),
        encoding="utf-8",
    )
    assert load_raw_tweets(path).height == 3


def test_load_raw_tweets_missing_file(tmp_path: Path):
    """Arquivo inexistente levanta RawDataError."""
    with pytest.raises(RawDataError):
        load_raw_tweets(tmp_path / "nao_existe.csv")


def test_load_raw_tweets_missing_columns(tmp_path: Path):
    """Sem colunas obrigatórias levanta RawDataError."""
    path = tmp_path / "c.csv"
    path.write_text("a,b\n1,2\n", encoding="utf-8")
    with pytest.raises(RawDataError):
        load_raw_tweets(path)


# ---------------------------------------------------------------- seed


def test_seed_everything_sets_hashseed():
    """PYTHONHASHSEED é fixado."""
    seed_everything(123)
    assert os.environ["PYTHONHASHSEED"] == "123"


# ---------------------------------------------------------------- ollama


def _install_fake_ollama(monkeypatch: pytest.MonkeyPatch, chat: Any) -> None:
    module = types.ModuleType("ollama")

    class Client:
        def __init__(self, host: str, timeout: float) -> None:
            self.chat = chat

    module.Client = Client  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ollama", module)


def test_ollama_chat_success(monkeypatch: pytest.MonkeyPatch):
    """Retorna o conteúdo da resposta."""
    _install_fake_ollama(monkeypatch, lambda **_: {"message": {"content": " oi "}})
    client = OllamaClient(LLMSettings())
    assert client.chat([{"role": "user", "content": "x"}]) == "oi"
    assert client._ensure_client() is client._ensure_client()


@pytest.mark.parametrize(
    ("error", "expected"),
    [(ConnectionError("x"), OllamaConnectionError), (ValueError("x"), LLMGenerationError)],
)
def test_ollama_chat_errors(monkeypatch: pytest.MonkeyPatch, error: Exception, expected: type):
    """Traduz erros de rede e genéricos para exceções do domínio."""

    def boom(**_: Any) -> None:
        raise error

    _install_fake_ollama(monkeypatch, boom)
    with pytest.raises(expected):
        OllamaClient(LLMSettings()).chat([])


def test_ollama_chat_empty_response(monkeypatch: pytest.MonkeyPatch):
    """Resposta vazia levanta LLMGenerationError."""
    _install_fake_ollama(monkeypatch, lambda **_: {"message": {"content": "  "}})
    with pytest.raises(LLMGenerationError):
        OllamaClient(LLMSettings()).chat([])


# ---------------------------------------------------------------- sentiment


def test_resolve_device_cpu():
    """'cpu' resolve para -1."""
    assert _resolve_device("cpu") == -1
    assert _resolve_device("auto") in {-1, 1}


def test_sentiment_lazy_load_with_fake_transformers(monkeypatch: pytest.MonkeyPatch):
    """Carrega o pipeline sob demanda e classifica."""
    module = types.ModuleType("transformers")
    module.pipeline = lambda **_: (
        lambda texts, batch_size=1: [  # type: ignore[attr-defined]
            {"label": "positive", "score": 0.9} for _ in texts
        ]
    )
    monkeypatch.setitem(sys.modules, "transformers", module)
    clf = SentimentClassifier(
        SentimentParams(model_name="fake", device="cpu", label_map={"positive": "Positivo"})
    )
    assert not clf.is_loaded
    clf.load()
    assert clf.is_loaded
    assert clf.predict(["x"])[0].label == "Positivo"


def test_sentiment_load_failure(monkeypatch: pytest.MonkeyPatch):
    """Falha de carga vira ModelLoadError."""
    module = types.ModuleType("transformers")

    def boom(**_: Any) -> None:
        raise OSError("sem rede")

    module.pipeline = boom  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "transformers", module)
    with pytest.raises(ModelLoadError):
        SentimentClassifier(SentimentParams(model_name="fake", device="cpu")).load()


def test_sentiment_inference_failure():
    """Falha de inferência vira ModelInferenceError."""
    clf = SentimentClassifier(SentimentParams(model_name="fake", device="cpu"))

    def boom(*_: Any, **__: Any) -> None:
        raise RuntimeError("x")

    clf._pipeline = boom
    with pytest.raises(ModelInferenceError):
        clf.predict(["a"])


# ---------------------------------------------------------------- pipelines


def test_run_preprocessing_and_sentiment(monkeypatch: pytest.MonkeyPatch, tmp_paths: ProjectPaths):
    """Bronze -> silver -> predições, com classificador falso."""
    from src.pipelines import sentiment as sentiment_pipeline
    from src.pipelines.preprocessing import run_preprocessing

    tmp_paths.ensure_directories()
    tmp_paths.bronze_tweets.write_text(CSV_COMMA, encoding="utf-8")
    settings = SimpleNamespace(
        sample_size=None, random_seed=42, model=SimpleNamespace(sentiment=None)
    )

    processed = run_preprocessing(settings, tmp_paths)  # type: ignore[arg-type]
    assert tmp_paths.silver_tweets.exists()

    class FakeClassifier:
        def __init__(self, _: Any) -> None: ...

        def predict(self, texts: list[str]) -> list[Prediction]:
            return [Prediction(label="Positivo", score=0.7) for _ in texts]

    monkeypatch.setattr(sentiment_pipeline, "SentimentClassifier", FakeClassifier)
    result = sentiment_pipeline.run_sentiment(settings, tmp_paths, processed, chunk_size=2)  # type: ignore[arg-type]
    assert ProcessedColumns.SENTIMENT_PRED in result.columns
    assert tmp_paths.sentiment_predictions.exists()


def test_run_summarization(
    monkeypatch: pytest.MonkeyPatch, tmp_paths: ProjectPaths, predicted_df: pl.DataFrame
):
    """Gera JSON de insights e resumo com sumarizador falso."""
    from src.pipelines import summarization

    class FakeSummarizer:
        def __init__(self, _: Any) -> None: ...

        def summarize(self, report: Any) -> SummaryResponse:
            return SummaryResponse(summary_markdown="# resumo", model="fake")

    monkeypatch.setattr(summarization, "InsightSummarizer", FakeSummarizer)
    tmp_paths.ensure_directories()
    settings = SimpleNamespace(llm=None)
    response = summarization.run_summarization(settings, tmp_paths, predicted_df)  # type: ignore[arg-type]
    assert response.summary_markdown == "# resumo"
    assert tmp_paths.insights_json.exists()
    assert tmp_paths.summary_markdown.read_text(encoding="utf-8") == "# resumo"

    # fallback de leitura: sem DataFrame, usa o parquet de predições
    predicted_df.write_parquet(tmp_paths.sentiment_predictions)
    summarization.run_summarization(settings, tmp_paths)  # type: ignore[arg-type]


def test_run_topics(monkeypatch: pytest.MonkeyPatch, tmp_paths: ProjectPaths, predicted_df):
    """Atribui tópicos com encoder e modelo de tópicos falsos."""
    from src.pipelines import topics

    class FakeEncoder:
        def __init__(self, _: Any) -> None: ...

        def encode(self, texts: list[str]) -> list[list[float]]:
            return [[0.0] for _ in texts]

    class FakeTopicModel:
        def __init__(self, _: Any) -> None: ...

        def fit_transform(self, texts: list[str], embeddings: Any) -> list[int]:
            return [i % 2 for i in range(len(texts))]

        def top_terms(self, topic_id: int) -> list[str]:
            return ["a", "b"]

        def save(self, path: Path) -> Path:
            return path

    monkeypatch.setattr(topics, "EmbeddingEncoder", FakeEncoder)
    monkeypatch.setattr(topics, "TopicModel", FakeTopicModel)
    tmp_paths.ensure_directories()
    settings = SimpleNamespace(model=SimpleNamespace(embeddings=None, topics=None))

    result, terms = topics.run_topics(settings, tmp_paths, predicted_df)  # type: ignore[arg-type]
    assert set(terms) == {0, 1}
    assert ProcessedColumns.TOPIC_ID in result.columns

    predicted_df.write_parquet(tmp_paths.sentiment_predictions)
    topics.run_topics(settings, tmp_paths)  # type: ignore[arg-type]


def test_run_full_pipeline_orchestrates(monkeypatch: pytest.MonkeyPatch, tmp_paths: ProjectPaths):
    """O workflow chama as etapas na ordem e devolve o resumo."""
    from src.pipelines import workflow

    calls: list[str] = []
    df = pl.DataFrame({"x": [1]})

    def step(name: str, ret: Any):
        def _run(*_: Any, **__: Any) -> Any:
            calls.append(name)
            return ret

        return _run

    monkeypatch.setattr(workflow, "run_preprocessing", step("pre", df))
    monkeypatch.setattr(workflow, "run_sentiment", step("sent", df))
    monkeypatch.setattr(workflow, "run_topics", step("topics", (df, {0: ["a"]})))
    monkeypatch.setattr(workflow, "run_evaluation", step("eval", None))
    monkeypatch.setattr(workflow, "run_reporting", step("rep", None))
    summary = SummaryResponse(summary_markdown="ok", model="fake")
    monkeypatch.setattr(workflow, "run_summarization", step("sum", summary))

    assert workflow.run_full_pipeline(SimpleNamespace(), tmp_paths) is summary  # type: ignore[arg-type]
    assert calls == ["pre", "sent", "topics", "eval", "rep", "sum"]

    calls.clear()
    workflow.run_full_pipeline(SimpleNamespace(), tmp_paths, with_topics=False)  # type: ignore[arg-type]
    assert "topics" not in calls
