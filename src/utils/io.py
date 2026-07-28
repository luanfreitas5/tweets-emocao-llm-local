"""Utilitários de I/O para parquet e JSON.

Escreve/lê parquet (via polars) e JSON — incluindo serialização de modelos
Pydantic. Cria diretórios de saída sob demanda.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl
from pydantic import BaseModel


def write_parquet(df: pl.DataFrame, path: Path) -> Path:
    """Grava um DataFrame polars em parquet, criando o diretório se preciso.

    Parameters
    ----------
    df : pl.DataFrame
        DataFrame a persistir.
    path : Path
        Caminho de destino ``.parquet``.

    Returns
    -------
    Path
        O caminho gravado.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)
    return path


def read_parquet(path: Path) -> pl.DataFrame:
    """Lê um arquivo parquet como DataFrame polars.

    Parameters
    ----------
    path : Path
        Caminho do arquivo ``.parquet``.

    Returns
    -------
    pl.DataFrame
        DataFrame lido.

    Raises
    ------
    FileNotFoundError
        Se o arquivo não existir.
    """
    if not path.exists():
        raise FileNotFoundError(f"Parquet não encontrado: {path}")
    return pl.read_parquet(path)


def write_json(data: BaseModel | dict[str, Any], path: Path) -> Path:
    """Grava um dicionário ou modelo Pydantic como JSON (UTF-8, indentado).

    Parameters
    ----------
    data : BaseModel | dict
        Conteúdo a serializar. Modelos Pydantic usam ``model_dump_json``.
    path : Path
        Caminho de destino ``.json``.

    Returns
    -------
    Path
        O caminho gravado.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, BaseModel):
        path.write_text(data.model_dump_json(indent=2), encoding="utf-8")
    else:
        path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )
    return path


def read_json(path: Path) -> dict[str, Any]:
    """Lê um arquivo JSON e retorna um dicionário.

    Parameters
    ----------
    path : Path
        Caminho do arquivo ``.json``.

    Returns
    -------
    dict
        Conteúdo desserializado.

    Raises
    ------
    FileNotFoundError
        Se o arquivo não existir.
    """
    if not path.exists():
        raise FileNotFoundError(f"JSON não encontrado: {path}")
    return json.loads(path.read_text(encoding="utf-8"))
