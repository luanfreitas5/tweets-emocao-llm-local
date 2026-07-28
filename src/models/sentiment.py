"""Classificador de sentimento pt-BR via Hugging Face (BERTimbau/afins).

Envolve um pipeline ``text-classification`` do ``transformers``, mapeando os
rótulos do modelo para a forma canônica do projeto. Carregamento *lazy*: o
modelo só é baixado/instanciado no primeiro ``predict``.
"""

from __future__ import annotations

import logging
from contextlib import suppress
from typing import TYPE_CHECKING, Any

from src.config.settings import SentimentParams
from src.exceptions.model import ModelInferenceError, ModelLoadError
from src.models.base import Prediction

if TYPE_CHECKING:  # evita custo de import do torch no import do módulo
    pass

logger = logging.getLogger(__name__)


def _resolve_device(device: str) -> int:
    """Resolve o device para o índice esperado pelo ``transformers.pipeline``.

    Parameters
    ----------
    device : str
        ``"auto"``, ``"cpu"`` ou ``"cuda"``.

    Returns
    -------
    int
        ``1`` para GPU, ``-1`` para CPU.
    """
    if device == "cpu":
        return -1
    with suppress(ImportError):
        import torch  # noqa: PLC0415

        if device in {"auto", "cuda"} and torch.cuda.is_available():
            return 1
    return -1


class SentimentClassifier:
    """Classificador de sentimento baseado em Transformer pt-BR.

    Parameters
    ----------
    params : SentimentParams
        Hiperparâmetros validados (modelo, batch, device, mapa de rótulos).

    Examples
    --------
    >>> clf = SentimentClassifier(params)
    >>> clf.predict(["adorei o produto", "que serviço horrível"])  # doctest: +SKIP
    [Prediction(label='Positivo', score=0.98), Prediction(label='Negativo', ...)]
    """

    def __init__(self, params: SentimentParams) -> None:
        self.params = params
        self._pipeline: Any | None = None

    def _ensure_loaded(self) -> Any:
        """Carrega o pipeline do Hugging Face sob demanda (lazy).

        Returns
        -------
        Any
            Pipeline ``text-classification`` pronto para inferência.

        Raises
        ------
        ModelLoadError
            Se o modelo não puder ser carregado.
        """
        if self._pipeline is not None:
            return self._pipeline
        try:
            from transformers import pipeline  # noqa: PLC0415

            logger.info("Carregando modelo de sentimento: %s", self.params.model_name)
            self._pipeline = pipeline(
                task="text-classification",
                model=self.params.model_name,
                device=_resolve_device(self.params.device),
                truncation=True,
                max_length=self.params.max_length,
            )
        except Exception as error:
            logger.exception("Falha ao carregar o modelo de sentimento")
            raise ModelLoadError(
                f"Não foi possível carregar {self.params.model_name}: {error}"
            ) from error
        return self._pipeline

    def _to_canonical(self, raw_label: str) -> str:
        """Converte o rótulo do modelo para a forma canônica do projeto."""
        return self.params.label_map.get(raw_label.lower(), raw_label.capitalize())

    def predict(self, texts: list[str]) -> list[Prediction]:
        """Classifica um lote de textos quanto ao sentimento.

        Parameters
        ----------
        texts : list[str]
            Textos limpos a classificar.

        Returns
        -------
        list[Prediction]
            Uma predição canônica por texto.

        Raises
        ------
        ModelInferenceError
            Se a inferência falhar.
        """
        if not texts:
            return []
        classifier = self._ensure_loaded()
        try:
            outputs = classifier(texts, batch_size=self.params.batch_size)
        except Exception as error:
            logger.exception("Falha na inferência de sentimento")
            raise ModelInferenceError(f"Erro ao classificar sentimento: {error}") from error

        return [
            Prediction(label=self._to_canonical(out["label"]), score=float(out["score"]))
            for out in outputs
        ]
