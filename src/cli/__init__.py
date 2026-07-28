"""Comandos de linha de comando para cada etapa do pipeline.

Cada módulo expõe uma função ``main()`` registrada como script em
``pyproject.toml`` (ex.: ``uv run tweets-preprocess``).

Módulos
-------
common
    Helpers de bootstrap (settings, logging, seed, paths) compartilhados.
preprocess, classify, topics, summarize
    Um comando por etapa do fluxo.
"""

from src.cli.common import bootstrap

__all__ = ["bootstrap"]
