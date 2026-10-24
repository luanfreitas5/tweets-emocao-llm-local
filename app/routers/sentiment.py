"""Roteador de classificação de sentimento."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status

from app.dependencies import get_classifier, get_settings
from app.schemas import SentimentItem, SentimentRequest, SentimentResponse
from src.config.settings import Settings
from src.exceptions.model import ModelInferenceError, ModelLoadError
from src.models.sentiment import SentimentClassifier
from src.preprocessing.cleaning import clean_tweet

router = APIRouter(prefix="/sentiment", tags=["sentiment"])


@router.post("", response_model=SentimentResponse)
def classify(
    payload: SentimentRequest,
    request: Request,
    classifier: Annotated[SentimentClassifier, Depends(get_classifier)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> SentimentResponse:
    """Classifica o sentimento de uma lista de textos.

    Os textos são limpos (mesma limpeza do treino) antes da inferência, para
    evitar *train-serve skew*.

    Parameters
    ----------
    payload : SentimentRequest
        Textos a classificar.
    request : Request
        Requisição corrente (limites de ``configs/deploy.yaml`` em ``app.state``).
    classifier : SentimentClassifier
        Classificador injetado.
    settings : Settings
        Configuração injetada (nome do modelo na resposta).

    Returns
    -------
    SentimentResponse
        Rótulo e score por texto.

    Raises
    ------
    HTTPException
        422 se o lote ou algum texto exceder os limites; 503 se o modelo não
        puder carregar; 500 em falha de inferência.
    """
    limits = request.app.state.limits
    if len(payload.texts) > limits["max_texts_per_request"]:
        raise HTTPException(
            status_code=422,
            detail=f"No máximo {limits['max_texts_per_request']} textos por requisição.",
        )
    if any(len(text) > limits["max_text_length"] for text in payload.texts):
        raise HTTPException(
            status_code=422,
            detail=f"Cada texto deve ter no máximo {limits['max_text_length']} caracteres.",
        )
    cleaned = [clean_tweet(text) for text in payload.texts]
    try:
        predictions = classifier.predict(cleaned)
    except ModelLoadError as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(error)
        ) from error
    except ModelInferenceError as error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(error)
        ) from error

    results = [
        SentimentItem(text=text, label=pred.label, score=pred.score)
        for text, pred in zip(payload.texts, predictions, strict=True)
    ]
    return SentimentResponse(results=results, model_name=settings.model.sentiment.model_name)
