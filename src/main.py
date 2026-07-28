"""Orquestração principal do pipeline (Python calcula, LLM explica).

Executa o fluxo ponta a ponta ou uma única etapa via ``--stage``: limpeza ->
classificação de sentimento -> modelagem de tópicos -> agregação de insights ->
resumo pelo LLM local.

Uso
---
    uv run python -m src.main [--stage all|preprocess|classify|topics|summarize] \
        [--sample-size N] [--no-topics]
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Literal

from src.config.logging import configure_logging
from src.config.paths import ProjectPaths, get_paths
from src.config.seed import seed_everything
from src.config.settings import Settings, load_settings
from src.pipelines.preprocessing import run_preprocessing
from src.pipelines.sentiment import run_sentiment
from src.pipelines.summarization import run_summarization
from src.pipelines.topics import run_topics
from src.pipelines.workflow import run_full_pipeline

Stage = Literal["all", "preprocess", "classify", "topics", "summarize"]


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


def bootstrap(sample_size: int | None = None) -> Context:
    """Inicializa logging, semente, configuração e caminhos.

    Parameters
    ----------
    sample_size : int | None, optional
        Sobrescreve ``sample_size`` da configuração, by default None.

    Returns
    -------
    Context
        Contexto pronto para as etapas do pipeline.

    Examples
    --------
    >>> ctx = bootstrap(sample_size=1000)  # doctest: +SKIP
    """
    configure_logging()
    settings = load_settings()
    if sample_size is not None:
        settings = settings.model_copy(update={"sample_size": sample_size})
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
        choices=["all", "preprocess", "classify", "topics", "summarize"],
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
    return parser


def run_stage(stage: Stage, ctx: Context, *, with_topics: bool = True) -> None:
    """Executa a etapa selecionada do pipeline.

    Parameters
    ----------
    stage : Stage
        Etapa a executar: ``"all"``, ``"preprocess"``, ``"classify"``,
        ``"topics"`` ou ``"summarize"``.
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
        case "summarize":
            response = run_summarization(ctx.settings, ctx.paths)
            print(response.summary_markdown)


def main() -> None:
    """Ponto de entrada do pipeline (fluxo completo ou etapa isolada via ``--stage``)."""
    args = build_parser().parse_args()
    ctx = bootstrap(sample_size=args.sample_size)
    run_stage(args.stage, ctx, with_topics=not args.no_topics)


if __name__ == "__main__":  # pragma: no cover
    main()
