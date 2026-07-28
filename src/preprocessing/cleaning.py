"""Limpeza de texto de tweets — com remoção de *data leakage*.

Ponto crítico de projeto: o rótulo da base foi gerado por **supervisão
distante** a partir de emoticons (``:)``/``:(``) e hashtags (``#fato``). Esses
símbolos aparecem no próprio ``tweet_text``. Se não forem removidos, qualquer
modelo aprende o atalho (o emoticon) em vez do conteúdo — métrica inflada e
inútil em produção. :func:`remove_label_leakage` elimina essa fuga.
"""

from __future__ import annotations

from src.constants import regex


def remove_label_leakage(text: str) -> str:
    """Remove emoticons e hashtags que originaram o rótulo (anti-leakage).

    Parameters
    ----------
    text : str
        Texto bruto do tweet.

    Returns
    -------
    str
        Texto sem os emoticons/hashtags rotuladores.

    Examples
    --------
    >>> remove_label_leakage("que dia lindo :) #feliz")
    'que dia lindo'
    """
    without_emoticons = regex.EMOTICON.sub(" ", text)
    without_hashtags = regex.HASHTAG.sub(" ", without_emoticons)
    return regex.WHITESPACE.sub(" ", without_hashtags).strip()


def clean_tweet(text: str, *, drop_leakage: bool = True) -> str:
    """Normaliza o texto de um tweet para modelagem.

    Passos: remoção opcional de leakage, URLs, menções, números, colapso de
    caracteres repetidos, normalização de espaços e *casefold*.

    Parameters
    ----------
    text : str
        Texto bruto do tweet.
    drop_leakage : bool, optional
        Se ``True`` (padrão), remove emoticons/hashtags rotuladores.

    Returns
    -------
    str
        Texto limpo. Pode ser vazio se o tweet só continha ruído.

    Examples
    --------
    >>> clean_tweet("@joao amooooo isso :) http://x.co #top")
    'amoo isso'
    """
    if not isinstance(text, str):
        return ""

    cleaned = text
    if drop_leakage:
        cleaned = remove_label_leakage(cleaned)

    cleaned = regex.URL.sub(" ", cleaned)
    cleaned = regex.MENTION.sub(" ", cleaned)
    cleaned = regex.HASHTAG.sub(" ", cleaned)
    cleaned = regex.EMOTICON.sub(" ", cleaned)
    cleaned = regex.NUMBER.sub(" ", cleaned)
    cleaned = regex.REPEATED_CHARS.sub(r"\1\1", cleaned)
    cleaned = regex.WHITESPACE.sub(" ", cleaned).strip()
    return cleaned.casefold()
