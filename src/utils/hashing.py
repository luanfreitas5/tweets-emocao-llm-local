"""Hashing para rastreabilidade e reprodutibilidade de dados.

Um hash estável do dataset permite detectar mudanças silenciosas e vincular um
modelo/relatório aos dados exatos que o produziram.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import polars as pl

_CHUNK = 1 << 20  # 1 MiB


def hash_file(path: Path) -> str:
    """Retorna o hash SHA-256 de um arquivo (leitura em blocos).

    Parameters
    ----------
    path : Path
        Caminho do arquivo.

    Returns
    -------
    str
        Digest hexadecimal SHA-256.

    Raises
    ------
    FileNotFoundError
        Se o arquivo não existir.
    """
    if not path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado para hash: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handler:
        while chunk := handler.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def hash_dataframe(df: pl.DataFrame) -> str:
    """Retorna um hash SHA-256 estável do conteúdo de um DataFrame.

    Usa o hash por linha do polars, tornando o resultado independente da ordem
    de leitura mas sensível ao conteúdo.

    Parameters
    ----------
    df : pl.DataFrame
        DataFrame a resumir.

    Returns
    -------
    str
        Digest hexadecimal SHA-256.
    """
    row_hashes = df.hash_rows().sort().to_numpy().tobytes()
    return hashlib.sha256(row_hashes).hexdigest()
