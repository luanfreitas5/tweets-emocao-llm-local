"""Testes baseados em propriedades (hypothesis) das transformações de texto."""

from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st

from src.constants import regex
from src.preprocessing.cleaning import clean_tweet, remove_label_leakage

# Alfabeto realista de tweet (evita curiosidades de casefold unicode que não
# são o alvo destes testes).
_TWEET_TEXT = st.text(
    alphabet=st.characters(
        whitelist_categories=("Lu", "Ll", "Nd", "Zs"),
        whitelist_characters="@#:)(-/.!? ",
    ),
    max_size=280,
)


@given(text=_TWEET_TEXT)
def test_clean_tweet_is_idempotent(text: str):
    """Invariante: limpar um texto já limpo não o altera novamente."""
    once = clean_tweet(text)
    assert clean_tweet(once) == once


@given(text=_TWEET_TEXT)
def test_clean_tweet_never_contains_urls_or_mentions(text: str):
    """Invariante: o texto limpo nunca contém URL ou menção."""
    cleaned = clean_tweet(text)
    assert regex.URL.search(cleaned) is None
    assert regex.MENTION.search(cleaned) is None


@given(text=_TWEET_TEXT)
def test_remove_leakage_has_no_emoticons(text: str):
    """Invariante: após remover leakage, não sobra emoticon reconhecível."""
    cleaned = remove_label_leakage(text)
    assert regex.EMOTICON.search(cleaned) is None
