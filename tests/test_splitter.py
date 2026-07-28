"""Testes do particionamento estratificado."""

from __future__ import annotations

import polars as pl
import pytest

from src.constants.columns import ProcessedColumns
from src.data.splitter import stratified_split


@pytest.fixture
def labeled_df() -> pl.DataFrame:
    """DataFrame sintético balanceado para testar o split."""
    labels = (["Positivo"] * 100) + (["Negativo"] * 100) + (["Neutro"] * 100)
    return pl.DataFrame(
        {
            ProcessedColumns.ID: list(range(300)),
            ProcessedColumns.SENTIMENT: labels,
        }
    )


def test_split_partitions_are_disjoint_and_complete(labeled_df: pl.DataFrame):
    """Treino/val/teste não se sobrepõem e cobrem todo o dataset."""
    train, val, test = stratified_split(labeled_df, test_size=0.2, val_size=0.1, seed=42)
    total = train.height + val.height + test.height
    assert total == labeled_df.height
    ids = (
        set(train[ProcessedColumns.ID])
        | set(val[ProcessedColumns.ID])
        | set(test[ProcessedColumns.ID])
    )
    assert len(ids) == labeled_df.height


def test_split_preserves_class_proportions(labeled_df: pl.DataFrame):
    """Cada classe aparece nos três conjuntos (estratificação)."""
    train, val, test = stratified_split(labeled_df, test_size=0.2, val_size=0.1)
    for part in (train, val, test):
        assert part[ProcessedColumns.SENTIMENT].n_unique() == 3


def test_split_rejects_invalid_sizes(labeled_df: pl.DataFrame):
    """test_size + val_size >= 1 deve levantar ValueError."""
    with pytest.raises(ValueError):
        stratified_split(labeled_df, test_size=0.7, val_size=0.4)
