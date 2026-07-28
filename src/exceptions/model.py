"""Exceções relacionadas a modelos."""

from __future__ import annotations

from src.exceptions.base import TweetsProjectError


class ModelLoadError(TweetsProjectError):
    """Falha ao carregar um modelo (pesos ausentes, download, incompatibilidade)."""


class ModelInferenceError(TweetsProjectError):
    """Falha durante a inferência de um modelo."""
