"""Testes do pipeline de figuras (``run_reporting``)."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import polars as pl
import pytest

from src.config.paths import ProjectPaths, get_paths
from src.constants.columns import ProcessedColumns
from src.pipelines.reporting import run_reporting
from src.utils.io import write_parquet


@pytest.fixture
def tmp_paths(tmp_path: Path) -> ProjectPaths:
    """Caminhos do projeto redirecionados para um diretório temporário."""
    return dataclasses.replace(
        get_paths(),
        figures=tmp_path / "figures",
        topic_assignments=tmp_path / "topics.parquet",
        sentiment_predictions=tmp_path / "predictions.parquet",
    )


@pytest.fixture
def labeled_df(predicted_df: pl.DataFrame) -> pl.DataFrame:
    """DataFrame previsto acrescido do rótulo de referência."""
    return predicted_df.with_columns(
        pl.Series(
            ProcessedColumns.SENTIMENT,
            ["Positivo", "Negativo", "Negativo", "Negativo", "Positivo", "Positivo"],
        )
    )


def _assert_figure_files(base: Path) -> None:
    """Verifica que ``.png`` e ``.svg`` existem e não estão vazios."""
    for suffix in (".png", ".svg"):
        file = base.with_suffix(suffix)
        assert file.exists(), f"Figura ausente: {file}"
        assert file.stat().st_size > 0


def test_run_reporting_saves_both_figures(
    tmp_paths: ProjectPaths, labeled_df: pl.DataFrame
) -> None:
    """Com rótulo de referência, gera distribuição e matriz de confusão."""
    saved = run_reporting(tmp_paths, labeled_df)

    assert [p.name for p in saved] == ["distribuicao_sentimento", "matriz_confusao"]
    for base in saved:
        _assert_figure_files(base)


def test_run_reporting_skips_confusion_matrix_without_reference(
    tmp_paths: ProjectPaths, predicted_df: pl.DataFrame
) -> None:
    """Sem a coluna ``sentiment``, só a distribuição é gerada."""
    saved = run_reporting(tmp_paths, predicted_df)

    assert [p.name for p in saved] == ["distribuicao_sentimento"]
    _assert_figure_files(saved[0])
    assert not (tmp_paths.figures / "matriz_confusao.png").exists()


def test_run_reporting_prefers_topic_assignments(
    tmp_paths: ProjectPaths, labeled_df: pl.DataFrame
) -> None:
    """Sem DataFrame explícito, lê o parquet de tópicos quando ele existe."""
    write_parquet(labeled_df, tmp_paths.topic_assignments)

    saved = run_reporting(tmp_paths)

    assert len(saved) == 2


def test_run_reporting_falls_back_to_predictions(
    tmp_paths: ProjectPaths, predicted_df: pl.DataFrame
) -> None:
    """Sem parquet de tópicos, usa o de predições de sentimento."""
    write_parquet(predicted_df, tmp_paths.sentiment_predictions)

    saved = run_reporting(tmp_paths)

    assert [p.name for p in saved] == ["distribuicao_sentimento"]


def test_run_reporting_raises_when_no_input(tmp_paths: ProjectPaths) -> None:
    """Sem DataFrame e sem parquets, falha de forma explícita."""
    with pytest.raises(FileNotFoundError):
        run_reporting(tmp_paths)
