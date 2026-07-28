"""Exceções relacionadas a dados."""

from __future__ import annotations

from src.exceptions.base import TweetsProjectError


class RawDataError(TweetsProjectError):
    """Erro ao ler ou interpretar os dados brutos (arquivo ausente/corrompido)."""


class DataValidationError(TweetsProjectError):
    """Violação de um contrato de dados (schema pandera)."""
