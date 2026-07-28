"""Ingestão, escrita e particionamento dos datasets de tweets.

Módulos
-------
loader
    Carrega os CSVs brutos (formatos com vírgula e com ponto e vírgula).
splitter
    Particiona os dados em treino/validação/teste de forma estratificada.
catalog
    Cataloga os arquivos disponíveis em ``data/raw``.
"""

from src.data.catalog import DatasetEntry, list_raw_datasets
from src.data.loader import load_raw_tweets
from src.data.splitter import stratified_split

__all__ = [
    "DatasetEntry",
    "list_raw_datasets",
    "load_raw_tweets",
    "stratified_split",
]
