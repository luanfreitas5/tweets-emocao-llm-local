"""Fixtures compartilhadas dos testes.

Usa apenas DataFrames sintéticos pequenos — nunca dados de produção.
"""

from __future__ import annotations

import polars as pl
import pytest

from src.constants.columns import ProcessedColumns, RawColumns


@pytest.fixture
def raw_df() -> pl.DataFrame:
    """DataFrame bruto sintético no formato de ``NoThemeTweets.csv``."""
    return pl.DataFrame(
        {
            RawColumns.ID: [1, 2, 3, 4],
            RawColumns.TEXT: [
                "que dia lindo :) #feliz",
                "@joao odeio isso :( http://x.co",
                "reunião marcada para amanhã",
                "amooooo esse lugar :)",
            ],
            RawColumns.DATE: ["Tue Aug 21 04:35:39 +0000 2018"] * 4,
            RawColumns.SENTIMENT: ["Positivo", "Negativo", "Neutro", "positive"],
            RawColumns.QUERY_USED: [":)", ":(", "#reuniao", ":)"],
        }
    )


@pytest.fixture
def predicted_df() -> pl.DataFrame:
    """DataFrame com sentimento previsto e tópicos, para testar a agregação."""
    return pl.DataFrame(
        {
            ProcessedColumns.ID: [1, 2, 3, 4, 5, 6],
            ProcessedColumns.TEXT_CLEAN: [
                "adorei o filme",
                "que filme ruim",
                "o jogo foi otimo",
                "jogo horrivel",
                "comida deliciosa",
                "comida ok",
            ],
            ProcessedColumns.SENTIMENT_PRED: [
                "Positivo",
                "Negativo",
                "Positivo",
                "Negativo",
                "Positivo",
                "Neutro",
            ],
            ProcessedColumns.TOPIC_ID: [0, 0, 1, 1, 2, 2],
        }
    )
