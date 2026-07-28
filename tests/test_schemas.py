"""Testes dos contratos de dados (pandera)."""

from __future__ import annotations

import polars as pl
import pytest
from pandera.errors import SchemaError

from src.constants.columns import ProcessedColumns
from src.schemas.processed import validate_processed
from src.schemas.raw import validate_raw


def test_validate_raw_accepts_valid(raw_df: pl.DataFrame):
    """O contrato bruto aceita um DataFrame bem formado."""
    assert validate_raw(raw_df).height == raw_df.height


def test_validate_processed_rejects_unknown_label():
    """O contrato processado rejeita rótulos fora do vocabulário."""
    df = pl.DataFrame(
        {
            ProcessedColumns.ID: [1],
            ProcessedColumns.TEXT_CLEAN: ["texto limpo"],
            ProcessedColumns.SENTIMENT: ["Alegre"],  # inválido
            ProcessedColumns.SENTIMENT_ID: [1],
        }
    )
    with pytest.raises(SchemaError):
        validate_processed(df)


def test_validate_processed_rejects_empty_text():
    """O contrato processado rejeita texto limpo vazio."""
    df = pl.DataFrame(
        {
            ProcessedColumns.ID: [1],
            ProcessedColumns.TEXT_CLEAN: [""],
            ProcessedColumns.SENTIMENT: ["Positivo"],
            ProcessedColumns.SENTIMENT_ID: [1],
        }
    )
    with pytest.raises(SchemaError):
        validate_processed(df)
