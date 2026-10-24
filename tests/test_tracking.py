"""Testes do contrato de run do MLflow (linhagem, URI SQLite e no-op desativado)."""

from __future__ import annotations

import dataclasses
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.config.paths import get_paths
from src.config.settings import ExperimentSettings, load_settings
from src.evaluation.classification import ClassificationReport
from src.experiment import tracking
from src.experiment.tracking import (
    build_run_tags,
    compute_git_sha,
    log_artifacts,
    log_evaluation,
    tracked_run,
)


def _fake_git(status: str, sha: str = "a" * 40):
    """Cria um substituto de ``subprocess.run`` que simula o ``git``."""

    def _run(command: list[str], **_: object) -> SimpleNamespace:
        return SimpleNamespace(stdout=status if "status" in command else sha)

    return _run


@pytest.mark.smoke
def test_compute_git_sha_returns_head_when_clean(monkeypatch: pytest.MonkeyPatch):
    """Árvore limpa: devolve o SHA do HEAD."""
    monkeypatch.setattr(subprocess, "run", _fake_git(status=""))
    assert compute_git_sha() == "a" * 40


@pytest.mark.smoke
def test_compute_git_sha_refuses_dirty_tree(monkeypatch: pytest.MonkeyPatch):
    """Árvore suja: levanta RuntimeError, a menos que ``allow_dirty``."""
    monkeypatch.setattr(subprocess, "run", _fake_git(status=" M src/main.py"))
    with pytest.raises(RuntimeError, match="Árvore Git suja"):
        compute_git_sha()
    assert compute_git_sha(allow_dirty=True) == "a" * 40


def test_build_run_tags_has_lineage(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """As tags de linhagem trazem git_sha, hash SHA-256 dos dados e semente."""
    data = tmp_path / "bronze.csv"
    data.write_text("id,tweet_text\n1,oi\n", encoding="utf-8")
    monkeypatch.setattr(tracking, "compute_git_sha", lambda *_, **__: "b" * 40)
    tags = build_run_tags(data, seed=42)
    assert tags["git_sha"] == "b" * 40
    assert len(tags["data_hash"]) == 64
    assert tags["seed"] == "42"


def test_build_run_tags_missing_data_raises(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Dados ausentes falham cedo, antes de abrir o run."""
    monkeypatch.setattr(tracking, "compute_git_sha", lambda *_, **__: "b" * 40)
    with pytest.raises(FileNotFoundError):
        build_run_tags(tmp_path / "inexistente.csv", seed=1)


def test_resolve_tracking_uri_makes_sqlite_absolute(tmp_path: Path):
    """URI SQLite relativo vira absoluto (o MLflow resolveria contra o cwd)."""
    uri = tracking._resolve_tracking_uri("sqlite:///mlruns/mlflow.db", tmp_path)
    assert uri == f"sqlite:///{(tmp_path / 'mlruns' / 'mlflow.db').as_posix()}"
    assert (tmp_path / "mlruns").is_dir()


def test_resolve_tracking_uri_keeps_http_uri(tmp_path: Path):
    """URIs de servidor remoto não são alterados."""
    assert tracking._resolve_tracking_uri("http://mlflow:5000", tmp_path) == "http://mlflow:5000"


@pytest.mark.smoke
def test_tracked_run_is_noop_when_disabled():
    """Com o rastreamento desativado nada é aberto nem importado."""
    settings = load_settings()
    assert settings.experiment.enabled is False
    with tracked_run(settings, get_paths(), "teste"):
        pass


def test_log_helpers_are_noop_without_active_run(tmp_path: Path):
    """Sem run ativo, registrar métricas/artefatos não levanta erro."""
    report = ClassificationReport(f1_macro=0.5, f1_ci95=(0.4, 0.6), per_class={})
    log_evaluation(report, n_samples=10)
    log_artifacts([tmp_path / "nada.json"])


@pytest.mark.integration
def test_tracked_run_logs_tags_and_metrics(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Integração: run no SQLite temporário com tags de linhagem e F1 com IC."""
    mlflow = pytest.importorskip("mlflow")
    data = tmp_path / "bronze.csv"
    data.write_text("id,tweet_text\n1,oi\n", encoding="utf-8")
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.setattr(tracking, "compute_git_sha", lambda *_, **__: "c" * 40)

    experiment = ExperimentSettings(
        enabled=True, tracking_uri=f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    )
    settings = load_settings().model_copy(update={"experiment": experiment})
    paths = dataclasses.replace(get_paths(), bronze_tweets=data, root=tmp_path)

    report = ClassificationReport(f1_macro=0.5, f1_ci95=(0.4, 0.6), per_class={})
    with tracked_run(settings, paths, "teste"):
        log_evaluation(report, n_samples=10)

    run = mlflow.search_runs(experiment_names=[experiment.experiment_name], output_format="list")[0]
    assert run.data.tags["git_sha"] == "c" * 40
    assert run.data.metrics["f1_macro_ci95_low"] == pytest.approx(0.4)
