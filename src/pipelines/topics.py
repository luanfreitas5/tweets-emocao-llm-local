"""Pipeline de modelagem de tópicos: embeddings -> BERTopic -> atribuição."""

from __future__ import annotations

import logging

import polars as pl

from src.config.paths import ProjectPaths
from src.config.settings import Settings
from src.constants.columns import ProcessedColumns
from src.models.embeddings import EmbeddingEncoder
from src.models.topics import TopicModel
from src.utils.io import read_parquet, write_parquet
from src.utils.timing import timed

logger = logging.getLogger(__name__)


def run_topics(
    settings: Settings,
    paths: ProjectPaths,
    tweets_df: pl.DataFrame | None = None,
) -> tuple[pl.DataFrame, dict[int, list[str]]]:
    """Extrai tópicos dos tweets e atribui um tópico a cada um.

    Parameters
    ----------
    settings : Settings
        Configuração validada (embeddings + tópicos).
    paths : ProjectPaths
        Caminhos do projeto.
    tweets_df : pl.DataFrame | None, optional
        DataFrame de entrada; se ``None``, tenta ``paths.sentiment_predictions``
        e, na ausência, ``paths.processed_tweets``.

    Returns
    -------
    tuple[pl.DataFrame, dict[int, list[str]]]
        DataFrame com ``topic_id`` e o mapa ``{topic_id: top_terms}``.
    """
    if tweets_df is None:
        source = (
            paths.sentiment_predictions
            if paths.sentiment_predictions.exists()
            else paths.processed_tweets
        )
        tweets_df = read_parquet(source)

    texts = tweets_df[ProcessedColumns.TEXT_CLEAN].to_list()

    with timed("modelagem de tópicos"):
        encoder = EmbeddingEncoder(settings.model.embeddings)
        embeddings = encoder.encode(texts)

        topic_model = TopicModel(settings.model.topics)
        topic_ids = topic_model.fit_transform(texts, embeddings)
        topic_model.save(paths.topic_model)

        top_terms = {tid: topic_model.top_terms(tid) for tid in set(topic_ids) if tid != -1}

    result = tweets_df.with_columns(pl.Series(ProcessedColumns.TOPIC_ID, topic_ids))
    write_parquet(result, paths.topic_assignments)
    logger.info("Atribuição de tópicos gravada em %s", paths.topic_assignments)
    return result, top_terms
