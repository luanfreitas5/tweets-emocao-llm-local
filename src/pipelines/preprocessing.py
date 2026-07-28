"""Pipeline de limpeza: raw CSV -> parquet processado."""

from __future__ import annotations

import logging

import polars as pl

from src.config.paths import ProjectPaths
from src.config.settings import Settings
from src.data.loader import load_raw_tweets
from src.preprocessing.pipeline import preprocess_tweets
from src.utils.io import write_parquet
from src.utils.timing import timed

logger = logging.getLogger(__name__)


def run_preprocessing(settings: Settings, paths: ProjectPaths) -> pl.DataFrame:
    """Executa a limpeza dos tweets brutos e grava o parquet processado.

    Parameters
    ----------
    settings : Settings
        Configuração validada (usa ``sample_size`` e ``random_seed``).
    paths : ProjectPaths
        Caminhos do projeto.

    Returns
    -------
    pl.DataFrame
        DataFrame processado (também persistido em ``paths.processed_tweets``).
    """
    with timed("pré-processamento dos tweets"):
        raw = load_raw_tweets(
            paths.raw_labeled, sample_size=settings.sample_size, seed=settings.random_seed
        )
        processed = preprocess_tweets(raw)
        write_parquet(processed, paths.processed_tweets)
        logger.info("Parquet processado gravado em %s", paths.processed_tweets)
    return processed
