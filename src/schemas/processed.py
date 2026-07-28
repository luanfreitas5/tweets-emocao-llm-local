"""Contrato de dados do parquet processado (pós-limpeza).

Valida a saída da limpeza (estágio ``processed``): texto limpo não vazio,
rótulos canônicos e ids de sentimento no conjunto esperado.
"""

from __future__ import annotations

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import Series

from src.constants.labels import SENTIMENT_LABELS


class ProcessedTweetsSchema(pa.DataFrameModel):
    """Contrato do parquet de tweets limpos.

    Garante que o vazamento de emoticons foi removido (``text_clean``) e que os
    rótulos estão no vocabulário canônico.
    """

    id: Series[int] = pa.Field(ge=0)
    text_clean: Series[str] = pa.Field(nullable=False, str_length={"min_value": 1})
    sentiment: Series[str] = pa.Field(isin=list(SENTIMENT_LABELS))
    sentiment_id: Series[int] = pa.Field(isin=[0, 1, 2])

    class Config:  # type: ignore
        """Configuração do contrato."""

        strict = False
        coerce = True


def validate_processed(df: pl.DataFrame) -> pl.DataFrame:
    """Valida um DataFrame processado contra :class:`ProcessedTweetsSchema`.

    Parameters
    ----------
    df : pl.DataFrame
        DataFrame de tweets limpos.

    Returns
    -------
    pl.DataFrame
        O mesmo DataFrame, validado.

    Raises
    ------
    pandera.errors.SchemaError
        Se o DataFrame violar o contrato.
    """
    return ProcessedTweetsSchema.validate(df)
