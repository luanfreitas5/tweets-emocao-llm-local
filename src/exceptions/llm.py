"""Exceções relacionadas ao LLM local (Ollama)."""

from __future__ import annotations

from src.exceptions.base import TweetsProjectError


class OllamaConnectionError(TweetsProjectError):
    """Não foi possível conectar ao servidor Ollama local."""


class LLMGenerationError(TweetsProjectError):
    """Falha ao gerar o resumo (timeout, resposta inválida, modelo ausente)."""
