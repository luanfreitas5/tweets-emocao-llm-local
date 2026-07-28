"""Particionamento estratificado dos dados em treino/validação/teste."""

from __future__ import annotations

import logging

import polars as pl

from src.constants.columns import ProcessedColumns

logger = logging.getLogger(__name__)


def stratified_split(
    df: pl.DataFrame,
    label_col: str = ProcessedColumns.SENTIMENT,
    test_size: float = 0.2,
    val_size: float = 0.1,
    seed: int = 42,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Divide o DataFrame em treino/validação/teste preservando a proporção de classes.

    A estratificação é feita por amostragem dentro de cada grupo de rótulo,
    garantindo que a distribuição de sentimento seja mantida nos três conjuntos.

    Parameters
    ----------
    df : pl.DataFrame
        DataFrame completo.
    label_col : str, optional
        Coluna de rótulo usada na estratificação, by default ``sentiment``.
    test_size : float, optional
        Fração para teste (0-1), by default 0.2.
    val_size : float, optional
        Fração para validação (0-1), by default 0.1.
    seed : int, optional
        Semente de reprodutibilidade, by default 42.

    Returns
    -------
    tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]
        Tupla ``(train, val, test)``.

    Raises
    ------
    ValueError
        Se ``test_size + val_size >= 1``.

    Examples
    --------
    >>> train, val, test = stratified_split(df, test_size=0.2, val_size=0.1)
    """
    if test_size + val_size >= 1:
        raise ValueError("A soma de test_size e val_size deve ser menor que 1.")

    shuffled = df.sample(fraction=1.0, shuffle=True, seed=seed)

    train_parts: list[pl.DataFrame] = []
    val_parts: list[pl.DataFrame] = []
    test_parts: list[pl.DataFrame] = []

    for _, group in shuffled.group_by(label_col, maintain_order=True):
        n = group.height
        n_test = int(round(n * test_size))
        n_val = int(round(n * val_size))
        test_parts.append(group.slice(0, n_test))
        val_parts.append(group.slice(n_test, n_val))
        train_parts.append(group.slice(n_test + n_val))

    train = pl.concat(train_parts).sample(fraction=1.0, shuffle=True, seed=seed)
    val = pl.concat(val_parts).sample(fraction=1.0, shuffle=True, seed=seed)
    test = pl.concat(test_parts).sample(fraction=1.0, shuffle=True, seed=seed)

    logger.info(
        "Split estratificado: treino=%d, val=%d, teste=%d", train.height, val.height, test.height
    )
    return train, val, test
