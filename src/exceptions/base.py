"""Exceção base do projeto."""

from __future__ import annotations


class TweetsProjectError(Exception):
    """Exceção base da qual todas as exceções do projeto herdam.

    Permite capturar qualquer erro específico do domínio com um único
    ``except TweetsProjectError``.
    """
