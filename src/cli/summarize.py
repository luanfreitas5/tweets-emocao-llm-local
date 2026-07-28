"""Comando de CLI: geração do resumo via LLM local."""

from __future__ import annotations

import argparse

from src.cli.common import bootstrap
from src.pipelines.summarization import run_summarization


def main() -> None:
    """Ponto de entrada do comando ``tweets-summarize``."""
    parser = argparse.ArgumentParser(
        description="Gera o resumo em linguagem simples (Ollama) a partir do JSON estruturado."
    )
    parser.parse_args()

    ctx = bootstrap()
    response = run_summarization(ctx.settings, ctx.paths)
    print(response.summary_markdown)


if __name__ == "__main__":  # pragma: no cover
    main()
