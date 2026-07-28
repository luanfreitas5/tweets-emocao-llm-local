"""Modelagem de tópicos com BERTopic.

Descobre tópicos de forma não supervisionada a partir dos embeddings dos tweets.
Envolve o BERTopic com carregamento *lazy* e persistência do modelo treinado.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from src.config.settings import TopicParams
from src.exceptions.model import ModelLoadError

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class TopicModel:
    """Modelo de tópicos baseado em BERTopic.

    Parameters
    ----------
    params : TopicParams
        Hiperparâmetros validados (tamanho mínimo de tópico, redução, etc.).

    Examples
    --------
    >>> topic_model = TopicModel(params)
    >>> ids = topic_model.fit_transform(texts, embeddings)  # doctest: +SKIP
    """

    def __init__(self, params: TopicParams) -> None:
        self.params = params
        self._model: Any | None = None

    def _build(self) -> Any:
        """Instancia o BERTopic com os parâmetros do projeto.

        Returns
        -------
        Any
            Instância de ``BERTopic``.

        Raises
        ------
        ModelLoadError
            Se o BERTopic não puder ser instanciado.
        """
        try:
            from bertopic import BERTopic

            nr_topics = None if self.params.nr_topics == "auto" else self.params.nr_topics
            return BERTopic(
                language=self.params.language,
                min_topic_size=self.params.min_topic_size,
                nr_topics="auto" if self.params.nr_topics == "auto" else nr_topics,
                top_n_words=self.params.top_n_words,
                calculate_probabilities=self.params.calculate_probabilities,
                low_memory=self.params.low_memory,
                verbose=True,
            )
        except Exception as error:
            logger.error("Falha ao instanciar o BERTopic: %s", error)
            raise ModelLoadError(f"Não foi possível criar o BERTopic: {error}") from error

    def fit_transform(self, texts: list[str], embeddings: NDArray[np.float32]) -> list[int]:
        """Treina o modelo e retorna o tópico de cada documento.

        Parameters
        ----------
        texts : list[str]
            Textos limpos.
        embeddings : numpy.ndarray
            Embeddings pré-computados correspondentes a ``texts``.

        Returns
        -------
        list[int]
            Id do tópico por documento (``-1`` = outlier).
        """
        self._model = self._build()

        if self._model is None:
            raise ModelLoadError("Falha ao criar o modelo de tópicos.")

        logger.info("Treinando BERTopic em %d documentos", len(texts))
        topic_ids, _ = self._model.fit_transform(texts, embeddings=embeddings)
        n_topics = len({t for t in topic_ids if t != -1})
        logger.info("Tópicos descobertos: %d", n_topics)
        return list(topic_ids)

    def top_terms(self, topic_id: int) -> list[str]:
        """Retorna os termos mais representativos de um tópico.

        Parameters
        ----------
        topic_id : int
            Id do tópico.

        Returns
        -------
        list[str]
            Termos ordenados por relevância (vazio se modelo não treinado).
        """
        if self._model is None:
            return []
        return [term for term, _ in self._model.get_topic(topic_id) or []]

    def save(self, path: Path) -> Path:
        """Salva o modelo de tópicos treinado.

        Parameters
        ----------
        path : Path
            Diretório de destino do modelo.

        Returns
        -------
        Path
            O caminho salvo.

        Raises
        ------
        ModelLoadError
            Se não houver modelo treinado para salvar.
        """
        if self._model is None:
            raise ModelLoadError("Nenhum modelo de tópicos treinado para salvar.")
        path.parent.mkdir(parents=True, exist_ok=True)
        self._model.save(str(path), serialization="safetensors", save_ctfidf=True)
        logger.info("Modelo de tópicos salvo em %s", path)
        return path
