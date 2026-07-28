"""Catálogo dos arquivos brutos disponíveis em ``data/raw``."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class DatasetEntry:
    """Metadados de um arquivo de dados bruto.

    Attributes
    ----------
    path : Path
        Caminho do arquivo.
    name : str
        Nome do arquivo.
    size_mb : float
        Tamanho em megabytes.
    """

    path: Path
    name: str
    size_mb: float


def list_raw_datasets(data_raw: Path) -> list[DatasetEntry]:
    """Lista os arquivos CSV disponíveis em ``data/raw`` (recursivamente).

    Parameters
    ----------
    data_raw : Path
        Diretório ``data/raw``.

    Returns
    -------
    list[DatasetEntry]
        Entradas ordenadas por tamanho decrescente.

    Examples
    --------
    >>> entries = list_raw_datasets(Path("data/raw"))
    """
    entries = [
        DatasetEntry(
            path=csv,
            name=csv.name,
            size_mb=round(csv.stat().st_size / (1024 * 1024), 2),
        )
        for csv in sorted(data_raw.rglob("*.csv"))
    ]
    return sorted(entries, key=lambda entry: entry.size_mb, reverse=True)
