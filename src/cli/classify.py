"""Comando de CLI: classificação de sentimento."""

from __future__ import annotations

import argparse

from src.cli.common import add_common_arguments, bootstrap
from src.pipelines.sentiment import run_sentiment


def main() -> None:
    """Ponto de entrada do comando ``tweets-classify``."""
    parser = argparse.ArgumentParser(description="Classifica o sentimento dos tweets (BERTimbau).")
    add_common_arguments(parser)
    args = parser.parse_args()

    ctx = bootstrap(sample_size=args.sample_size)
    run_sentiment(ctx.settings, ctx.paths)


if __name__ == "__main__":  # pragma: no cover
    main()
