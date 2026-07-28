"""Contrato de dados do CSV bruto de tweets.

Valida a entrada do pipeline (estágio ``raw``) antes de qualquer transformação,
falhando cedo com erro claro em caso de corrupção silenciosa.
"""

from __future__ import annotations

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import Series


class RawTweetsSchema(pa.DataFrameModel):
    """Contrato do CSV bruto (``NoThemeTweets.csv`` e afins).

    Notes
    -----
    ``sentiment`` chega como texto (``Positivo``/``Negativo``/``Neutro``) na base
    sem tema, e ``query_used`` guarda o emoticon/hashtag rotulador.
    """

    id: Series[int] = pa.Field(ge=0, coerce=True)
    tweet_text: Series[str] = pa.Field(nullable=False)
    tweet_date: Series[str] = pa.Field(nullable=True)
    sentiment: Series[str] = pa.Field(nullable=False)
    query_used: Series[str] = pa.Field(nullable=True)

    class Config:  # type: ignore
        """Configuração do contrato."""

        strict = False  # tolera colunas extras específicas de cada arquivo
        coerce = True


def validate_raw(df: pl.DataFrame) -> pl.DataFrame:
    """Valida um DataFrame bruto contra :class:`RawTweetsSchema`.

    Parameters
    ----------
    df : pl.DataFrame
        DataFrame lido do CSV bruto.

    Returns
    -------
    pl.DataFrame
        O mesmo DataFrame, validado.

    Raises
    ------
    pandera.errors.SchemaError
        Se o DataFrame violar o contrato.

    Examples
    --------
    >>> import polars as pl
    >>> df = pl.DataFrame(
    ...     {
    ...         RawColumns.ID: [1],
    ...         RawColumns.TEXT: ["oi :)"],
    ...         RawColumns.DATE: ["Tue Aug 21 04:35:39 +0000 2018"],
    ...         RawColumns.SENTIMENT: ["Positivo"],
    ...         RawColumns.QUERY_USED: [":)"],
    ...     }
    ... )
    >>> _ = validate_raw(df)
    """
    return RawTweetsSchema.validate(df)
