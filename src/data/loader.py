"""Carregamento dos CSVs brutos de tweets.

A base do Kaggle vem em dois formatos:

* **Sem tema** (``NoThemeTweets.csv`` etc.): separador vírgula, ``sentiment``
  textual (``Positivo``/``Negativo``/``Neutro``).
* **Treino/Teste** (``Train*.csv``, ``Test*.csv``): separador ponto e vírgula,
  ``sentiment`` numérico (0/1/2).

Este módulo detecta o separador, normaliza o rótulo para a forma canônica e
devolve um :class:`polars.DataFrame` consistente.
"""

from __future__ import annotations

import logging
from pathlib import Path

import polars as pl

from src.constants.columns import RawColumns
from src.constants.labels import normalize_sentiment
from src.exceptions.data import RawDataError

logger = logging.getLogger(__name__)


def _detect_separator(path: Path) -> str:
    """Detecta o separador (``,`` ou ``;``) pela primeira linha do arquivo.

    Parameters
    ----------
    path : Path
        Caminho do CSV.

    Returns
    -------
    str
        ``";"`` se o cabeçalho usar ponto e vírgula, senão ``","``.
    """
    with path.open(encoding="utf-8", errors="replace") as handler:
        header = handler.readline()
    return ";" if header.count(";") >= header.count(",") else ","


def load_raw_tweets(
    path: Path,
    sample_size: int | None = None,
    seed: int = 42,
) -> pl.DataFrame:
    """Carrega um CSV bruto de tweets e normaliza o rótulo de sentimento.

    Parameters
    ----------
    path : Path
        Caminho do CSV bruto.
    sample_size : int | None, optional
        Se informado, amostra aleatoriamente ``sample_size`` linhas (execução
        rápida/dev), by default ``None`` (usa tudo).
    seed : int, optional
        Semente da amostragem, by default 42.

    Returns
    -------
    pl.DataFrame
        DataFrame com as colunas de :class:`RawColumns` e ``sentiment``
        normalizado para ``Positivo``/``Negativo``/``Neutro``.

    Raises
    ------
    RawDataError
        Se o arquivo não existir ou não puder ser lido.

    Examples
    --------
    >>> df = load_raw_tweets(Path("data/raw/NoThemeTweets.csv"), sample_size=100)
    """
    if not path.exists():
        raise RawDataError(f"Arquivo bruto não encontrado: {path}")

    separator = _detect_separator(path)
    logger.info("Lendo %s (separador=%r)", path.name, separator)

    try:
        tweets_df = pl.read_csv(
            path,
            separator=separator,
            infer_schema_length=10_000,
            ignore_errors=True,
        )
    except (pl.exceptions.PolarsError, OSError) as error:
        logger.exception("Falha ao ler o CSV bruto: %s", path)
        raise RawDataError(f"Não foi possível ler {path}: {error}") from error

    missing = set(RawColumns.ALL) - set(tweets_df.columns)  # type: ignore
    if RawColumns.TEXT in missing or RawColumns.SENTIMENT in missing:
        raise RawDataError(f"Colunas obrigatórias ausentes em {path.name}: {sorted(missing)}")

    tweets_df = tweets_df.with_columns(
        pl.col(RawColumns.SENTIMENT)
        .cast(pl.Utf8)
        .map_elements(normalize_sentiment, return_dtype=pl.Utf8)
        .alias(RawColumns.SENTIMENT)
    )

    if sample_size is not None and sample_size < tweets_df.height:
        tweets_df = tweets_df.sample(n=sample_size, seed=seed)
        logger.info("Amostra de %d tweets aplicada", sample_size)

    logger.info("Carregados %d tweets de %s", tweets_df.height, path.name)
    return tweets_df
