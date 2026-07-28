"""Modelos computacionais do projeto ("Python calcula").

Módulos
-------
base
    Protocolo comum de classificador de texto.
sentiment
    Classificador de sentimento pt-BR via Hugging Face (BERTimbau/afins).
embeddings
    Codificador de embeddings de sentença (sentence-transformers).
topics
    Modelagem de tópicos com BERTopic sobre os embeddings.
"""

from src.models.base import TextClassifier
from src.models.embeddings import EmbeddingEncoder
from src.models.sentiment import SentimentClassifier
from src.models.topics import TopicModel

__all__ = [
    "EmbeddingEncoder",
    "SentimentClassifier",
    "TextClassifier",
    "TopicModel",
]
