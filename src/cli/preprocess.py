"""Comando de CLI: limpeza dos tweets brutos."""

from __future__ import annotations

import argparse

from src.cli.common import add_common_arguments, bootstrap
from src.pipelines.preprocessing import run_preprocessing


def main() -> None:
    """Ponto de entrada do comando ``tweets-preprocess``."""
    parser = argparse.ArgumentParser(description="Limpa os tweets brutos (remove leakage).")
    add_common_arguments(parser)
    args = parser.parse_args()

    ctx = bootstrap(sample_size=args.sample_size)
    run_preprocessing(ctx.settings, ctx.paths)


if __name__ == "__main__":  # pragma: no cover
    main()
