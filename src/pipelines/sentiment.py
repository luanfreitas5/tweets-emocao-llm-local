"""Pipeline de classificação de sentimento: processado -> predições."""

from __future__ import annotations

import logging

import polars as pl

from src.config.paths import ProjectPaths
from src.config.settings import Settings
from src.constants.columns import ProcessedColumns
from src.models.sentiment import SentimentClassifier
from src.utils.io import read_parquet, write_parquet
from src.utils.timing import build_progress, timed

logger = logging.getLogger(__name__)


def run_sentiment(
    settings: Settings,
    paths: ProjectPaths,
    tweets_df: pl.DataFrame | None = None,
    chunk_size: int = 10_000,
) -> pl.DataFrame:
    """Classifica o sentimento dos tweets em lotes e persiste as predições.

    Parameters
    ----------
    settings : Settings
        Configuração validada (hiperparâmetros do classificador).
    paths : ProjectPaths
        Caminhos do projeto.
    tweets_df : pl.DataFrame | None, optional
        DataFrame processado; se ``None``, lê de ``paths.processed_tweets``.
    chunk_size : int, optional
        Tamanho do bloco de textos por chamada, by default 10_000.

    Returns
    -------
    pl.DataFrame
        DataFrame com ``sentiment_pred`` e ``sentiment_score`` adicionados,
        persistido em ``paths.sentiment_predictions``.
    """
    if tweets_df is None:
        tweets_df = read_parquet(paths.processed_tweets)

    classifier = SentimentClassifier(settings.model.sentiment)
    texts = tweets_df[ProcessedColumns.TEXT_CLEAN].to_list()

    labels: list[str] = []
    scores: list[float] = []
    with timed("classificação de sentimento"), build_progress() as progress:
        task = progress.add_task("Classificando", total=len(texts))
        for start in range(0, len(texts), chunk_size):
            batch = texts[start : start + chunk_size]
            predictions = classifier.predict(batch)
            labels.extend(p.label for p in predictions)
            scores.extend(p.score for p in predictions)
            progress.update(task, advance=len(batch))

    result = tweets_df.with_columns(
        pl.Series(ProcessedColumns.SENTIMENT_PRED, labels),
        pl.Series(ProcessedColumns.SENTIMENT_SCORE, scores),
    )
    write_parquet(result, paths.sentiment_predictions)
    logger.info("Predições de sentimento gravadas em %s", paths.sentiment_predictions)
    return result
