"""Testes do pipeline de avaliação (métricas com IC 95% e por fatia)."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import polars as pl
import pytest

from src.config.paths import ProjectPaths, get_paths
from src.config.settings import Settings, load_settings
from src.constants.columns import ProcessedColumns
from src.pipelines.evaluation import run_evaluation


@pytest.fixture
def tmp_paths(tmp_path: Path) -> ProjectPaths:
    """Caminhos do projeto redirecionados para um diretório temporário."""
    return dataclasses.replace(
        get_paths(),
        topic_assignments=tmp_path / "topics.parquet",
        sentiment_predictions=tmp_path / "predictions.parquet",
        classification_metrics=tmp_path / "metrics" / "classification.json",
    )


@pytest.fixture
def fast_settings() -> Settings:
    """Configuração real com poucas iterações de bootstrap."""
    settings = load_settings()
    evaluation = settings.model.evaluation.model_copy(update={"bootstrap_iterations": 20})
    model = settings.model.model_copy(update={"evaluation": evaluation})
    return settings.model_copy(update={"model": model})


def test_run_evaluation_writes_metrics_json(
    fast_settings: Settings, tmp_paths: ProjectPaths, predicted_df: pl.DataFrame
):
    """Com rótulo de referência, grava F1-macro, IC 95% e métricas por fatia."""
    labeled = predicted_df.with_columns(
        pl.Series(
            ProcessedColumns.SENTIMENT,
            ["Positivo", "Negativo", "Negativo", "Negativo", "Positivo", "Positivo"],
        )
    )
    report = run_evaluation(fast_settings, tmp_paths, labeled)

    assert report is not None
    low, high = report.f1_ci95
    assert 0.0 <= low <= high <= 1.0
    payload = json.loads(tmp_paths.classification_metrics.read_text(encoding="utf-8"))
    assert payload["n_evaluated"] == 6
    assert len(payload["f1_macro_ci95"]) == 2
    assert "curto_1_5" in payload["per_slice_f1_macro"]


def test_run_evaluation_without_reference_label_returns_none(
    fast_settings: Settings, tmp_paths: ProjectPaths, predicted_df: pl.DataFrame
):
    """Sem a coluna ``sentiment`` a avaliação é ignorada, sem gravar arquivo."""
    assert run_evaluation(fast_settings, tmp_paths, predicted_df) is None
    assert not tmp_paths.classification_metrics.exists()
