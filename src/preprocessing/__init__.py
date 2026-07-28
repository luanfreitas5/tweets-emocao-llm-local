"""Limpeza de texto e preparação dos tweets.

Módulos
-------
cleaning
    Funções puras de limpeza de um texto (remoção de leakage, URLs, menções).
pipeline
    Aplica a limpeza a um DataFrame inteiro e produz o estágio ``processed``.
"""

from src.preprocessing.cleaning import clean_tweet, remove_label_leakage
from src.preprocessing.pipeline import preprocess_tweets

__all__ = [
    "clean_tweet",
    "preprocess_tweets",
    "remove_label_leakage",
]
