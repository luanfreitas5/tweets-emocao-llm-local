"""Utilitários compartilhados do projeto.

Módulos
-------
io
    Leitura/escrita de parquet e JSON (Pydantic-aware).
hashing
    Hash SHA-256 de arquivos e DataFrames (rastreabilidade de dados).
timing
    Cronometragem de blocos e barras de progresso ``rich``.
"""

from src.utils.hashing import hash_dataframe, hash_file
from src.utils.io import read_json, read_parquet, write_json, write_parquet
from src.utils.timing import build_progress, timed

__all__ = [
    "build_progress",
    "hash_dataframe",
    "hash_file",
    "read_json",
    "read_parquet",
    "timed",
    "write_json",
    "write_parquet",
]
