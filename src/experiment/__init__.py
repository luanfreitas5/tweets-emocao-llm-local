"""Rastreamento de experimentos com MLflow (backend SQLite).

Cada execução do pipeline vira um run reprodutível: tags de linhagem
(``git_sha``, ``data_hash``, ``seed``), parâmetros dos modelos, métricas com
incerteza (IC 95%) e artefatos (figuras, métricas, insights).

Módulos
-------
tracking
    ``tracked_run`` (contexto do run), ``compute_git_sha``, ``build_run_tags``,
    ``log_evaluation`` e ``log_artifacts``.
"""

from src.experiment.tracking import (
    build_run_tags,
    compute_git_sha,
    configure_tracking,
    log_artifacts,
    log_evaluation,
    tracked_run,
)

__all__ = [
    "build_run_tags",
    "compute_git_sha",
    "configure_tracking",
    "log_artifacts",
    "log_evaluation",
    "tracked_run",
]
