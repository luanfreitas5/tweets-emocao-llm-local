"""Codificador de embeddings de sentença (sentence-transformers).

Gera vetores densos multilíngues para os tweets limpos, usados como entrada da
modelagem de tópicos. Carregamento *lazy* do modelo.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from src.config.settings import EmbeddingParams
from src.exceptions.model import ModelLoadError

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class EmbeddingEncoder:
    """Codifica textos em embeddings densos com sentence-transformers.

    Parameters
    ----------
    params : EmbeddingParams
        Hiperparâmetros validados (modelo, batch, normalização).

    Examples
    --------
    >>> encoder = EmbeddingEncoder(params)
    >>> vectors = encoder.encode(["bom dia", "boa noite"])  # doctest: +SKIP
    >>> vectors.shape  # doctest: +SKIP
    (2, 384)
    """

    def __init__(self, params: EmbeddingParams) -> None:
        self.params = params
        self._model: Any | None = None

    def _ensure_loaded(self) -> Any:
        """Carrega o modelo de embeddings sob demanda (lazy).

        Returns
        -------
        Any
            Instância de ``SentenceTransformer``.

        Raises
        ------
        ModelLoadError
            Se o modelo não puder ser carregado.
        """
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer

            logger.info("Carregando modelo de embeddings: %s", self.params.model_name)
            self._model = SentenceTransformer(self.params.model_name)
        except Exception as error:
            logger.error("Falha ao carregar embeddings: %s", error)
            raise ModelLoadError(
                f"Não foi possível carregar {self.params.model_name}: {error}"
            ) from error
        return self._model

    def encode(self, texts: list[str], show_progress: bool = True) -> NDArray[np.float32]:
        """Gera embeddings para um lote de textos.

        Parameters
        ----------
        texts : list[str]
            Textos a codificar.
        show_progress : bool, optional
            Exibe barra de progresso do sentence-transformers, by default True.

        Returns
        -------
        numpy.ndarray
            Matriz ``(n_textos, dim)`` de embeddings ``float32``.
        """
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        model = self._ensure_loaded()
        embeddings = model.encode(
            texts,
            batch_size=self.params.batch_size,
            normalize_embeddings=self.params.normalize,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)
