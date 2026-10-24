"""Contrato de run do MLflow: linhagem + parâmetros + métricas com IC + artefatos.

Um run que registra só a métrica final responde "qual nota?", mas não "de qual
código, quais dados, qual semente?". Aqui todo run carrega:

* **tags** ``git_sha``, ``data_hash`` e ``seed`` (filtráveis na UI);
* **parâmetros** dos modelos e da amostragem;
* **métricas** com incerteza (F1-macro + IC 95% por bootstrap);
* **artefatos** (figuras, métricas JSON, insights).

O classificador de sentimento é um modelo pré-treinado do Hugging Face (sem
ajuste fino neste projeto); por isso o run registra ``model_name`` como
parâmetro em vez de ``log_model``. O backend é sempre um banco (SQLite), nunca o
*file store* ``./mlruns``.
"""

from __future__ import annotations

import logging
import os
import subprocess  # nosec B404 - usado apenas para consultar o git local
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.config.paths import ROOT, ProjectPaths
from src.config.settings import ExperimentSettings, Settings
from src.utils.hashing import hash_file

if TYPE_CHECKING:
    from src.evaluation.classification import ClassificationReport

logger = logging.getLogger(__name__)

_SQLITE_PREFIX = "sqlite:///"


def _get_active_mlflow() -> Any | None:
    """Retorna o módulo ``mlflow`` se houver run ativo; senão None.

    Evita importar o ``mlflow`` (pesado) quando o rastreamento está desativado:
    se o módulo nunca foi importado, não existe run ativo.
    """
    mlflow = sys.modules.get("mlflow")
    if mlflow is None or mlflow.active_run() is None:
        return None
    return mlflow


def compute_git_sha(root: Path = ROOT, *, allow_dirty: bool = False) -> str:
    """Retorna o SHA do ``HEAD``; recusa árvore Git suja.

    Parameters
    ----------
    root : Path, optional
        Raiz do repositório, by default a raiz do projeto.
    allow_dirty : bool, optional
        Permite alterações não commitadas (o SHA deixa de descrever o código
        executado), by default False.

    Returns
    -------
    str
        SHA completo do commit atual.

    Raises
    ------
    RuntimeError
        Se houver alterações não commitadas e ``allow_dirty`` for False.
    subprocess.CalledProcessError
        Se o comando ``git`` falhar (ex.: fora de um repositório).

    Examples
    --------
    >>> compute_git_sha()  # doctest: +SKIP
    '2ce7e7d64f11fc416cd75cab49a9c92ec3151f24'
    """
    status = subprocess.run(  # nosec B603 B607 - comando fixo, sem entrada externa
        ["git", "status", "--porcelain"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if status and not allow_dirty:
        raise RuntimeError("Árvore Git suja: faça commit antes de registrar o experimento.")
    return subprocess.run(  # nosec B603 B607 - comando fixo, sem entrada externa
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def build_run_tags(
    data_path: Path, seed: int, root: Path = ROOT, *, allow_dirty: bool = False
) -> dict[str, str]:
    """Monta as tags de linhagem de um run.

    Parameters
    ----------
    data_path : Path
        Arquivo bronze que alimenta o pipeline (hash SHA-256 vira ``data_hash``).
    seed : int
        Semente global da execução.
    root : Path, optional
        Raiz do repositório Git, by default a raiz do projeto.
    allow_dirty : bool, optional
        Ver :func:`compute_git_sha`, by default False.

    Returns
    -------
    dict[str, str]
        Tags ``git_sha``, ``data_hash`` e ``seed``.

    Raises
    ------
    RuntimeError
        Se a árvore Git estiver suja.
    FileNotFoundError
        Se ``data_path`` não existir.
    """
    return {
        "git_sha": compute_git_sha(root, allow_dirty=allow_dirty),
        "data_hash": hash_file(data_path),
        "seed": str(seed),
    }


def build_run_params(settings: Settings) -> dict[str, str | int | float | bool | None]:
    """Extrai os parâmetros que definem um run (para comparar runs na UI).

    Parameters
    ----------
    settings : Settings
        Configuração validada do projeto.

    Returns
    -------
    dict[str, str | int | float | bool | None]
        Parâmetros planos (MLflow não aceita estruturas aninhadas).
    """
    model = settings.model
    return {
        "sample_size": settings.sample_size,
        "sentiment_model": model.sentiment.model_name,
        "sentiment_batch_size": model.sentiment.batch_size,
        "sentiment_max_length": model.sentiment.max_length,
        "embedding_model": model.embeddings.model_name,
        "topic_min_size": model.topics.min_topic_size,
        "bootstrap_iterations": model.evaluation.bootstrap_iterations,
        "llm_model": settings.llm.model,
        "llm_temperature": settings.llm.options.temperature,
    }


def _resolve_tracking_uri(uri: str, root: Path) -> str:
    """Torna absoluto o caminho de um URI SQLite (o MLflow resolve contra o cwd)."""
    if not uri.startswith(_SQLITE_PREFIX):
        return uri
    db_path = Path(uri.removeprefix(_SQLITE_PREFIX))
    if not db_path.is_absolute():
        db_path = root / db_path
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return f"{_SQLITE_PREFIX}{db_path.as_posix()}"


def configure_tracking(settings: ExperimentSettings, root: Path = ROOT) -> str:
    """Define explicitamente o *tracking URI* e o experimento do MLflow.

    ``MLFLOW_TRACKING_URI`` (``.env``/ambiente) tem precedência sobre o YAML,
    o que permite apontar para o servidor do ``docker-compose``.

    Parameters
    ----------
    settings : ExperimentSettings
        Configuração de rastreamento validada.
    root : Path, optional
        Raiz usada para resolver caminhos SQLite relativos.

    Returns
    -------
    str
        URI efetivamente configurada.
    """
    import mlflow  # noqa: PLC0415 - import tardio: mlflow é pesado e opcional por execução

    raw_uri = os.environ.get("MLFLOW_TRACKING_URI") or settings.tracking_uri
    uri = _resolve_tracking_uri(raw_uri, root)
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(settings.experiment_name)
    logger.info("MLflow configurado: %s (experimento %s)", uri, settings.experiment_name)
    return uri


@contextmanager
def tracked_run(
    settings: Settings,
    paths: ProjectPaths,
    run_name: str,
    *,
    allow_dirty: bool = False,
) -> Iterator[None]:
    """Abre um run do MLflow com tags de linhagem e parâmetros; no-op se desativado.

    Parameters
    ----------
    settings : Settings
        Configuração validada (``settings.experiment.enabled`` liga o rastreamento).
    paths : ProjectPaths
        Caminhos do projeto (``bronze_tweets`` alimenta o ``data_hash``).
    run_name : str
        Nome do run na UI.
    allow_dirty : bool, optional
        Permite árvore Git suja (apenas desenvolvimento), by default False.

    Yields
    ------
    None

    Raises
    ------
    RuntimeError
        Se o rastreamento estiver ativo e a árvore Git estiver suja.

    Examples
    --------
    >>> with tracked_run(settings, paths, "pipeline"):  # doctest: +SKIP
    ...     run_full_pipeline(settings, paths)
    """
    if not settings.experiment.enabled:
        yield
        return

    import mlflow  # noqa: PLC0415

    tags = build_run_tags(paths.bronze_tweets, settings.random_seed, allow_dirty=allow_dirty)
    configure_tracking(settings.experiment, paths.root)
    with mlflow.start_run(run_name=run_name, tags=tags):
        mlflow.log_params(build_run_params(settings))
        logger.info("Run do MLflow iniciado: %s (git %s)", run_name, tags["git_sha"][:7])
        yield


def log_evaluation(report: ClassificationReport, n_samples: int) -> None:
    """Registra no run ativo o F1-macro com IC 95% e o F1 por classe/fatia.

    Não faz nada se não houver run ativo (rastreamento desativado).

    Parameters
    ----------
    report : ClassificationReport
        Resultado da avaliação (``evaluate_classification`` + fatias).
    n_samples : int
        Quantidade de tweets avaliados.
    """
    mlflow = _get_active_mlflow()
    if mlflow is None:
        return
    metrics: dict[str, float] = {
        "n_evaluated": float(n_samples),
        "f1_macro": report.f1_macro,
        "f1_macro_ci95_low": report.f1_ci95[0],
        "f1_macro_ci95_high": report.f1_ci95[1],
    }
    for label, values in report.per_class.items():
        if isinstance(values, dict) and "f1-score" in values:
            metrics[f"f1_{label.lower()}"] = float(values["f1-score"])
    for slice_name, value in report.per_slice.items():
        metrics[f"f1_macro_slice_{slice_name}"] = value
    mlflow.log_metrics(metrics)


def log_artifacts(paths: list[Path]) -> None:
    """Anexa arquivos (figuras, métricas, insights) ao run ativo; ignora ausentes.

    Parameters
    ----------
    paths : list[Path]
        Arquivos a anexar.
    """
    mlflow = _get_active_mlflow()
    if mlflow is None:
        return
    for path in paths:
        if path.is_file():
            mlflow.log_artifact(str(path))
        else:
            logger.warning("Artefato não encontrado, ignorado: %s", path)
