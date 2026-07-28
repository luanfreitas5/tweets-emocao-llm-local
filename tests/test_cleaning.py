"""Testes da limpeza de texto — foco no anti-leakage."""

from __future__ import annotations

import pytest

from src.constants import regex
from src.preprocessing.cleaning import clean_tweet, remove_label_leakage


@pytest.mark.smoke
def test_remove_label_leakage_strips_emoticons_and_hashtags():
    """O emoticon e a hashtag rotuladores devem sumir do texto."""
    result = remove_label_leakage("que dia lindo :) #feliz")
    assert ":)" not in result
    assert "#feliz" not in result
    assert "lindo" in result


@pytest.mark.smoke
def test_clean_tweet_removes_url_mention_and_lowercases():
    """URLs e menções são removidas e o texto vira minúsculo."""
    result = clean_tweet("@joao ODEIO isso :( http://x.co")
    assert "@joao" not in result
    assert "httpx2" not in result
    assert result == result.casefold()
    assert "odeio" in result


def test_clean_tweet_collapses_repeated_chars():
    """Caracteres repetidos 3+ vezes são reduzidos a dois."""
    assert clean_tweet("amooooo") == "amoo"


def test_clean_tweet_handles_non_string():
    """Entrada não textual retorna string vazia sem erro."""
    assert clean_tweet(None) == ""  # type: ignore[arg-type]


@pytest.mark.parametrize("emoticon", [":)", ":-)", ":(", ":D", ";)", "<3"])
def test_emoticon_regex_matches_common_variants(emoticon: str):
    """O padrão de emoticons cobre as variações comuns da base."""
    assert regex.EMOTICON.search(emoticon) is not None
