"""LLM local (Ollama) — a camada "LLM explica".

O LLM **não calcula nada**: recebe o :class:`~src.schemas.insights.InsightsReport`
(JSON já computado em Python) e produz um resumo em linguagem simples. Prompt e
temperatura baixa reduzem o risco de alucinação.

Módulos
-------
prompts
    Templates de prompt (sistema + usuário) para o resumo.
ollama_client
    Cliente fino sobre o pacote ``ollama`` (100% local).
summarizer
    Orquestra prompt + cliente e devolve ``SummaryResponse``.
"""

from src.llm.ollama_client import OllamaClient
from src.llm.summarizer import InsightSummarizer

__all__ = [
    "InsightSummarizer",
    "OllamaClient",
]
