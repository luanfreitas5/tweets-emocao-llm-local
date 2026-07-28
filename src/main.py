"""Orquestração principal do pipeline (Python calcula, LLM explica).

Executa o fluxo ponta a ponta: limpeza -> classificação de sentimento ->
modelagem de tópicos -> agregação de insights -> resumo pelo LLM local.

Uso
---
    uv run python -m src.main [--sample-size N] [--no-topics]
"""

from __future__ import annotations

import argparse

from src.pipelines.workflow import run_full_pipeline

import argparse
from dataclasses import dataclass

from src.config.logging import configure_logging
from src.config.paths import ProjectPaths, get_paths
from src.config.seed import seed_everything
from src.config.settings import Settings, load_settings


@dataclass(frozen=True, slots=True)
class Context:
    """Contexto de execução compartilhado por todos os comandos.

    Attributes
    ----------
    settings : Settings
        Configuração validada.
    paths : ProjectPaths
        Caminhos do projeto (diretórios já criados).
    """

    settings: Settings
    paths: ProjectPaths


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    """Adiciona argumentos comuns (``--sample-size``) a um parser.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        Parser ao qual anexar os argumentos.
    """
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Amostra apenas N tweets (execução rápida). Padrão: usa toda a base.",
    )


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
    add_common_arguments(parser)
    parser.add_argument(
        "--no-topics",
        action="store_true",
        help="Pula a modelagem de tópicos (execução mais rápida).",
    )
    return parser


def main() -> None:
    """Ponto de entrada do pipeline completo."""
    args = build_parser().parse_args()
    ctx = bootstrap(sample_size=args.sample_size)
    response = run_full_pipeline(ctx.settings, ctx.paths, with_topics=not args.no_topics)
    print(response.summary_markdown)


if __name__ == "__main__":  # pragma: no cover
    main()
