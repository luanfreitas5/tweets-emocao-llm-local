"""Roteador de classificação de sentimento."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from app.dependencies import get_classifier
from app.schemas import SentimentItem, SentimentRequest, SentimentResponse
from src.exceptions.model import ModelInferenceError, ModelLoadError
from src.models.sentiment import SentimentClassifier
from src.preprocessing.cleaning import clean_tweet

router = APIRouter(prefix="/sentiment", tags=["sentiment"])


@router.post("", response_model=SentimentResponse)
def classify(
    payload: SentimentRequest,
    classifier: SentimentClassifier = Depends(get_classifier),
) -> SentimentResponse:
    """Classifica o sentimento de uma lista de textos.

    Os textos são limpos (mesma limpeza do treino) antes da inferência, para
    evitar *train-serve skew*.

    Parameters
    ----------
    payload : SentimentRequest
        Textos a classificar.
    classifier : SentimentClassifier
        Classificador injetado.

    Returns
    -------
    SentimentResponse
        Rótulo e score por texto.

    Raises
    ------
    HTTPException
        503 se o modelo não puder carregar; 500 em falha de inferência.
    """
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
    return SentimentResponse(results=results)
