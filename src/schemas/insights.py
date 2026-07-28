"""Modelos Pydantic v2 da fronteira "Python calcula, LLM explica".

O :class:`InsightsReport` é o **JSON estruturado** produzido inteiramente em
Python (distribuições, contagens, tópicos, exemplos). Ele é a *única* entrada de
fatos do LLM: o modelo local não recebe os tweets crus nem calcula nada — apenas
transforma esses números em texto. :class:`SummaryResponse` tipa a saída.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class SentimentDistribution(BaseModel):
    """Distribuição de sentimento agregada (contagens e proporções)."""

    total: int = Field(ge=0, description="Total de tweets considerados.")
    counts: dict[str, int] = Field(description="Contagem por rótulo canônico.")
    proportions: dict[str, float] = Field(description="Proporção (0-1) por rótulo.")

    def dominant(self) -> str:
        """Retorna o rótulo de sentimento majoritário.

        Returns
        -------
        str
            Rótulo com maior contagem; ``"Neutro"`` se não houver dados.
        """
        if not self.counts:
            return "Neutro"
        return max(self.counts, key=lambda label: self.counts[label])


class TopicInsight(BaseModel):
    """Resumo estruturado de um tópico descoberto pelo BERTopic."""

    topic_id: int = Field(description="Id do tópico (-1 = outliers).")
    label: str = Field(description="Rótulo curto e legível do tópico.")
    size: int = Field(ge=0, description="Número de tweets no tópico.")
    top_terms: list[str] = Field(description="Termos mais representativos.")
    sentiment: SentimentDistribution = Field(description="Sentimento dentro do tópico.")
    example_texts: list[str] = Field(
        default_factory=list, description="Exemplos anonimizados (sem PII)."
    )


class InsightsReport(BaseModel):
    """JSON estruturado que alimenta o LLM (contrato anti-alucinação).

    Todos os campos são *calculados em Python*. O LLM só pode descrever o que
    está aqui — não deve introduzir números novos.

    Attributes
    ----------
    generated_at : datetime
        Momento da geração do relatório.
    total_tweets : int
        Total de tweets analisados.
    overall_sentiment : SentimentDistribution
        Distribuição de sentimento global.
    topics : list[TopicInsight]
        Tópicos descobertos, ordenados por tamanho.
    data_hash : str | None
        Hash do dataset de origem (rastreabilidade/reprodutibilidade).
    """

    generated_at: datetime = Field(default_factory=datetime.now)
    total_tweets: int = Field(ge=0)
    overall_sentiment: SentimentDistribution
    topics: list[TopicInsight] = Field(default_factory=list)
    data_hash: str | None = None


class SummaryResponse(BaseModel):
    """Saída tipada do resumo gerado pelo LLM local."""

    summary_markdown: str = Field(description="Resumo em linguagem simples (pt-BR).")
    model: str = Field(description="Modelo Ollama usado na geração.")
    source_report_hash: str | None = Field(
        default=None, description="Hash do InsightsReport que originou o resumo."
    )
