"""Centralização de caminhos do projeto usando ``pathlib.Path``.

Todos os caminhos derivam da raiz do repositório (nunca *hardcoded* em string
relativa espalhada pelo código) e são carregados de ``configs/paths.yaml``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

# Raiz do projeto: este arquivo está em src/config/paths.py -> parents[2].
ROOT: Path = Path(__file__).resolve().parents[2]
CONFIGS_DIR: Path = ROOT / "configs"
PATHS_YAML: Path = CONFIGS_DIR / "paths.yaml"


@dataclass(frozen=True, slots=True)
class ProjectPaths:
    """Coleção imutável de caminhos absolutos do projeto.

    Attributes
    ----------
    root : Path
        Raiz do repositório.
    data_raw, data_interim, data_processed, data_external : Path
        Diretórios dos estágios de dados.
    raw_labeled : Path
        CSV bruto rotulado por supervisão distante.
    processed_tweets : Path
        Parquet de tweets limpos (contrato ``ProcessedTweetsSchema``).
    sentiment_predictions : Path
        Parquet com as predições de sentimento.
    topic_assignments : Path
        Parquet com a atribuição de tópicos.
    insights_json : Path
        JSON estruturado consumido pelo LLM.
    summary_markdown : Path
        Resumo em linguagem simples gerado pelo LLM.
    models_root, topic_model : Path
        Diretórios de modelos.
    reports_root, figures : Path
        Diretórios de relatórios e figuras.
    logs : Path
        Diretório de logs.
    """

    root: Path
    data_raw: Path
    data_interim: Path
    data_processed: Path
    data_external: Path
    raw_labeled: Path
    processed_tweets: Path
    sentiment_predictions: Path
    topic_assignments: Path
    insights_json: Path
    summary_markdown: Path
    models_root: Path
    topic_model: Path
    reports_root: Path
    figures: Path
    logs: Path

    directories: tuple[Path, ...] = field(default=(), compare=False)

    def ensure_directories(self) -> None:
        """Cria os diretórios de saída do projeto, se ainda não existirem."""
        for directory in self.directories:
            directory.mkdir(parents=True, exist_ok=True)


def _abs(root: Path, relative: str) -> Path:
    """Resolve um caminho relativo do YAML para absoluto a partir da raiz."""
    return (root / relative).resolve()


def get_paths(paths_yaml: Path = PATHS_YAML, root: Path = ROOT) -> ProjectPaths:
    """Carrega ``configs/paths.yaml`` e resolve todos os caminhos para absolutos.

    Parameters
    ----------
    paths_yaml : Path, optional
        Arquivo YAML de caminhos, by default ``configs/paths.yaml``.
    root : Path, optional
        Raiz do projeto usada para resolver os caminhos relativos.

    Returns
    -------
    ProjectPaths
        Estrutura imutável com todos os caminhos absolutos.

    Raises
    ------
    FileNotFoundError
        Se o arquivo de caminhos não for encontrado.

    Examples
    --------
    >>> paths = get_paths()
    >>> paths.processed_tweets.name
    'tweets_processed.parquet'
    """
    if not paths_yaml.exists():
        raise FileNotFoundError(f"Arquivo de caminhos não encontrado: {paths_yaml}")

    with paths_yaml.open(encoding="utf-8") as handler:
        cfg = yaml.safe_load(handler)

    data = cfg["data"]
    files = cfg["files"]
    models = cfg["models"]
    reports = cfg["reports"]

    paths = ProjectPaths(
        root=root,
        data_raw=_abs(root, data["raw"]),
        data_interim=_abs(root, data["interim"]),
        data_processed=_abs(root, data["processed"]),
        data_external=_abs(root, data["external"]),
        raw_labeled=_abs(root, files["raw_labeled"]),
        processed_tweets=_abs(root, files["processed_tweets"]),
        sentiment_predictions=_abs(root, files["sentiment_predictions"]),
        topic_assignments=_abs(root, files["topic_assignments"]),
        insights_json=_abs(root, files["insights_json"]),
        summary_markdown=_abs(root, files["summary_markdown"]),
        models_root=_abs(root, models["root"]),
        topic_model=_abs(root, models["topic_model"]),
        reports_root=_abs(root, reports["root"]),
        figures=_abs(root, reports["figures"]),
        logs=_abs(root, cfg["logs"]),
    )
    object.__setattr__(
        paths,
        "directories",
        (
            paths.data_interim,
            paths.data_processed,
            paths.models_root,
            paths.reports_root,
            paths.figures,
            paths.logs,
        ),
    )
    return paths
