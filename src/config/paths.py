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
    data_bronze, data_silver, data_gold : Path
        Diretórios das camadas medalhão (bronze → silver → gold).
    bronze_tweets : Path
        CSV bruto rotulado por supervisão distante (camada bronze, imutável).
    silver_tweets : Path
        Parquet de tweets limpos (camada silver; contrato ``ProcessedTweetsSchema``).
    sentiment_predictions : Path
        Parquet com as predições de sentimento (camada gold).
    topic_assignments : Path
        Parquet com a atribuição de tópicos (camada gold).
    insights_json : Path
        JSON estruturado consumido pelo LLM.
    summary_markdown : Path
        Resumo em linguagem simples gerado pelo LLM.
    models_root, topic_model : Path
        Diretórios de modelos.
    classification_metrics : Path
        JSON com as métricas de classificação (IC 95% e por classe).
    reports_root, figures, metrics : Path
        Diretórios de relatórios, figuras e métricas.
    logs : Path
        Diretório de logs.
    """

    root: Path
    data_bronze: Path
    data_silver: Path
    data_gold: Path
    bronze_tweets: Path
    silver_tweets: Path
    sentiment_predictions: Path
    topic_assignments: Path
    insights_json: Path
    summary_markdown: Path
    classification_metrics: Path
    models_root: Path
    topic_model: Path
    reports_root: Path
    figures: Path
    metrics: Path
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
    >>> paths.silver_tweets.name
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
        data_bronze=_abs(root, data["bronze"]),
        data_silver=_abs(root, data["silver"]),
        data_gold=_abs(root, data["gold"]),
        bronze_tweets=_abs(root, files["bronze_tweets"]),
        silver_tweets=_abs(root, files["silver_tweets"]),
        sentiment_predictions=_abs(root, files["sentiment_predictions"]),
        topic_assignments=_abs(root, files["topic_assignments"]),
        insights_json=_abs(root, files["insights_json"]),
        summary_markdown=_abs(root, files["summary_markdown"]),
        classification_metrics=_abs(root, files["classification_metrics"]),
        models_root=_abs(root, models["root"]),
        topic_model=_abs(root, models["topic_model"]),
        reports_root=_abs(root, reports["root"]),
        figures=_abs(root, reports["figures"]),
        metrics=_abs(root, reports["metrics"]),
        logs=_abs(root, cfg["logs"]),
    )
    object.__setattr__(
        paths,
        "directories",
        (
            paths.data_silver,
            paths.data_gold,
            paths.models_root,
            paths.reports_root,
            paths.figures,
            paths.metrics,
            paths.logs,
        ),
    )
    return paths
