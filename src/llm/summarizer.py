"""Geração do resumo a partir do JSON estruturado (``InsightsReport``).

Orquestra o prompt e o cliente Ollama, devolvendo um :class:`SummaryResponse`
tipado. É o ponto onde "Python calcula" encontra "LLM explica".
"""

from __future__ import annotations

import logging

from src.config.settings import LLMSettings
from src.llm.ollama_client import OllamaClient
from src.llm.prompts import build_messages
from src.schemas.insights import InsightsReport, SummaryResponse

logger = logging.getLogger(__name__)


class InsightSummarizer:
    """Transforma um :class:`InsightsReport` em resumo textual via LLM local.

    Parameters
    ----------
    settings : LLMSettings
        Configuração do LLM.
    client : OllamaClient | None, optional
        Cliente injetável (facilita testes); criado a partir de ``settings`` se
        omitido.

    Examples
    --------
    >>> summarizer = InsightSummarizer(settings)
    >>> resposta = summarizer.summarize(report)  # doctest: +SKIP
    >>> print(resposta.summary_markdown)  # doctest: +SKIP
    """

    def __init__(self, settings: LLMSettings, client: OllamaClient | None = None) -> None:
        self.settings = settings
        self.client = client or OllamaClient(settings)

    def summarize(self, report: InsightsReport) -> SummaryResponse:
        """Gera o resumo em linguagem simples do relatório estruturado.

        Parameters
        ----------
        report : InsightsReport
            Relatório computado em Python (única fonte de fatos do LLM).

        Returns
        -------
        SummaryResponse
            Resumo em Markdown, modelo usado e hash de origem.

        Raises
        ------
        LLMGenerationError
            Se a geração do resumo falhar.
        """
        messages = build_messages(report)
        markdown = self.client.chat(messages)
        logger.info("Resumo gerado (%d caracteres)", len(markdown))
        return SummaryResponse(
            summary_markdown=markdown,
            model=self.settings.model,
            source_report_hash=report.data_hash,
        )
