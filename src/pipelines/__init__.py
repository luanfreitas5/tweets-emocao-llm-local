"""Pipelines ponta a ponta do projeto.

Cada função ``run_*`` executa uma etapa isolada e persiste seu artefato; o
``run_full_pipeline`` encadeia todas na ordem
preprocess -> classify -> topics -> evaluate -> plot -> summarize.

Módulos
-------
preprocessing
    Limpeza dos tweets brutos (bronze -> silver).
evaluation
    Métricas com IC 95% e por fatia em ``reports/metrics``.
sentiment
    Classificação de sentimento.
topics
    Modelagem de tópicos.
reporting
    Geração das figuras em ``reports/figures``.
summarization
    Agregação de insights + resumo via LLM local.
workflow
    Orquestração completa.
"""

from src.pipelines.evaluation import run_evaluation
from src.pipelines.preprocessing import run_preprocessing
from src.pipelines.reporting import run_reporting
from src.pipelines.sentiment import run_sentiment
from src.pipelines.summarization import run_summarization
from src.pipelines.topics import run_topics
from src.pipelines.workflow import run_full_pipeline

__all__ = [
    "run_evaluation",
    "run_full_pipeline",
    "run_preprocessing",
    "run_reporting",
    "run_sentiment",
    "run_summarization",
    "run_topics",
]
