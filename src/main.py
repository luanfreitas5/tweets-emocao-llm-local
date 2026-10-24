"""Orquestração principal do pipeline (Python calcula, LLM explica).

Executa o fluxo ponta a ponta ou uma única etapa via ``--stage``: limpeza
(bronze -> silver) -> classificação de sentimento -> modelagem de tópicos (gold) ->
avaliação -> figuras -> agregação de insights -> resumo pelo LLM local.

Uso
---
    uv run python -m src.main \
        [--stage all|preprocess|classify|topics|evaluate|plot|summarize] \
        [--sample-size N] [--no-topics] [--log-level DEBUG] [--track]
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Literal

from src.config.logging import configure_logging
from src.config.paths import ProjectPaths, get_paths
from src.config.seed import seed_everything
from src.config.settings import Settings, load_settings
from src.experiment.tracking import log_artifacts, tracked_run
from src.pipelines.evaluation import run_evaluation
from src.pipelines.preprocessing import run_preprocessing
from src.pipelines.reporting import run_reporting
from src.pipelines.sentiment import run_sentiment
from src.pipelines.summarization import run_summarization
from src.pipelines.topics import run_topics
from src.pipelines.workflow import run_full_pipeline

Stage = Literal["all", "preprocess", "classify", "topics", "evaluate", "plot", "summarize"]


@dataclass(frozen=True, slots=True)
class Context:
    """Contexto de execução compartilhado por todas as etapas.

    Attributes
    ----------
    settings : Settings
        Configuração validada.
    paths : ProjectPaths
        Caminhos do projeto (diretórios já criados).
    """

    settings: Settings
    paths: ProjectPaths


def bootstrap(
    sample_size: int | None = None,
    log_level: str | None = None,
    *,
    track: bool = False,
) -> Context:
    """Inicializa logging, semente, configuração e caminhos.

    Parameters
    ----------
    sample_size : int | None, optional
        Sobrescreve ``sample_size`` da configuração, by default None.
    log_level : str | None, optional
        Sobrescreve o nível de log do ``configs/logging.yaml``, by default None.
    track : bool, optional
        Ativa o rastreamento de experimentos com MLflow, by default False.

    Returns
    -------
    Context
        Contexto pronto para as etapas do pipeline.

    Examples
    --------
    >>> ctx = bootstrap(sample_size=1000)  # doctest: +SKIP
    """
    configure_logging(level=log_level)
    settings = load_settings()
    if sample_size is not None:
        settings = settings.model_copy(update={"sample_size": sample_size})
    if track:
        experiment = settings.experiment.model_copy(update={"enabled": True})
        settings = settings.model_copy(update={"experiment": experiment})
    seed_everything(settings.random_seed)
    paths = get_paths()
    paths.ensure_directories()
    return Context(settings=settings, paths=paths)


def build_parser() -> argparse.ArgumentParser:
    """Constrói o parser de argumentos do pipeline principal.

    Returns
    -------
    argparse.ArgumentParser
        Parser configurado.
    """
    parser = argparse.ArgumentParser(
        prog="tweets",
        description="Análise de emoção e tópicos em tweets pt-BR com LLM local.",
    )
    parser.add_argument(
        "--stage",
        choices=["all", "preprocess", "classify", "topics", "evaluate", "plot", "summarize"],
        default="all",
        help="Etapa a executar isoladamente, ou 'all' para o fluxo completo (padrão).",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Amostra apenas N tweets (execução rápida). Padrão: usa toda a base.",
    )
    parser.add_argument(
        "--no-topics",
        action="store_true",
        help="Pula a modelagem de tópicos (execução mais rápida). Só se aplica a --stage all.",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default=None,
        help="Nível de log (padrão: configs/logging.yaml).",
    )
    parser.add_argument(
        "--track",
        action="store_true",
        help="Registra o run no MLflow (SQLite). Exige árvore Git limpa.",
    )
    return parser


def run_stage(stage: Stage, ctx: Context, *, with_topics: bool = True) -> None:
    """Executa a etapa selecionada do pipeline.

    Parameters
    ----------
    stage : Stage
        Etapa a executar: ``"all"``, ``"preprocess"``, ``"classify"``,
        ``"topics"``, ``"evaluate"``, ``"plot"`` ou ``"summarize"``.
    ctx : Context
        Contexto já inicializado (settings + paths).
    with_topics : bool, optional
        Inclui a modelagem de tópicos quando ``stage="all"``, by default True.
    """
    match stage:
        case "all":
            response = run_full_pipeline(ctx.settings, ctx.paths, with_topics=with_topics)
            print(response.summary_markdown)
        case "preprocess":
            run_preprocessing(ctx.settings, ctx.paths)
        case "classify":
            run_sentiment(ctx.settings, ctx.paths)
        case "topics":
            run_topics(ctx.settings, ctx.paths)
        case "evaluate":
            run_evaluation(ctx.settings, ctx.paths)
        case "plot":
            run_reporting(ctx.paths)
        case "summarize":
            response = run_summarization(ctx.settings, ctx.paths)
            print(response.summary_markdown)


def main() -> None:
    """Ponto de entrada do pipeline (fluxo completo ou etapa isolada via ``--stage``)."""
    args = build_parser().parse_args()
    ctx = bootstrap(sample_size=args.sample_size, log_level=args.log_level, track=args.track)
    with tracked_run(ctx.settings, ctx.paths, run_name=f"pipeline-{args.stage}"):
        run_stage(args.stage, ctx, with_topics=not args.no_topics)
        log_artifacts(
            [
                ctx.paths.classification_metrics,
                ctx.paths.insights_json,
                ctx.paths.summary_markdown,
                *sorted(ctx.paths.figures.glob("*.png")),
            ]
        )


if __name__ == "__main__":  # pragma: no cover
    main()
