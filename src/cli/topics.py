"""Comando de CLI: modelagem de tópicos."""

from __future__ import annotations

import argparse

from src.cli.common import add_common_arguments, bootstrap
from src.pipelines.topics import run_topics


def main() -> None:
    """Ponto de entrada do comando ``tweets-topics``."""
    parser = argparse.ArgumentParser(description="Extrai tópicos com embeddings + BERTopic.")
    add_common_arguments(parser)
    args = parser.parse_args()

    ctx = bootstrap(sample_size=args.sample_size)
    run_topics(ctx.settings, ctx.paths)


if __name__ == "__main__":  # pragma: no cover
    main()
