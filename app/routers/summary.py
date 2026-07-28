"""Roteador de geração de resumo (LLM local a partir de JSON estruturado)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from app.dependencies import get_summarizer
from src.exceptions.llm import LLMGenerationError, OllamaConnectionError
from src.llm.summarizer import InsightSummarizer
from src.schemas.insights import InsightsReport, SummaryResponse

router = APIRouter(prefix="/summary", tags=["summary"])


@router.post("", response_model=SummaryResponse)
def summarize(
    report: InsightsReport,
    summarizer: InsightSummarizer = Depends(get_summarizer),
) -> SummaryResponse:
    """Gera um resumo em linguagem simples a partir de um relatório estruturado.

    O corpo é o ``InsightsReport`` já computado (fonte única de fatos): o LLM só
    explica, não calcula.

    Parameters
    ----------
    report : InsightsReport
        Relatório estruturado (Python calcula).
    summarizer : InsightSummarizer
        Componente de resumo injetado.

    Returns
    -------
    SummaryResponse
        Resumo em Markdown.

    Raises
    ------
    HTTPException
        503 se o Ollama estiver indisponível; 500 em falha de geração.
    """
    try:
        return summarizer.summarize(report)
    except OllamaConnectionError as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(error)
        ) from error
    except LLMGenerationError as error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(error)
        ) from error
