"""Exceções customizadas do projeto.

Módulos
-------
base
    Exceção base ``TweetsProjectError`` da qual todas herdam.
data
    Erros de ingestão/validação de dados.
model
    Erros de carregamento/inferência de modelos.
llm
    Erros de comunicação/geração com o LLM local (Ollama).
"""

from src.exceptions.base import TweetsProjectError
from src.exceptions.data import DataValidationError, RawDataError
from src.exceptions.llm import LLMGenerationError, OllamaConnectionError
from src.exceptions.model import ModelInferenceError, ModelLoadError

__all__ = [
    "DataValidationError",
    "LLMGenerationError",
    "ModelInferenceError",
    "ModelLoadError",
    "OllamaConnectionError",
    "RawDataError",
    "TweetsProjectError",
]
