"""Padrões regex compilados para limpeza de tweets.

Inclui o padrão de **emoticons/hashtags rotuladores** que precisa ser removido
para evitar *data leakage*: como o rótulo foi gerado por esses símbolos, deixá-los
no texto faria o modelo "trapacear" ao invés de aprender o conteúdo.
"""

from __future__ import annotations

import re
from typing import Final

#: URLs (http/https e encurtadores tipo t.co).
URL: Final = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)

#: Menções a usuários (@usuario).
MENTION: Final = re.compile(r"@\w+")

#: Hashtags completas (#tema) — o símbolo e o termo.
HASHTAG: Final = re.compile(r"#\w+")

#: Espaços em branco repetidos.
WHITESPACE: Final = re.compile(r"\s+")

#: Emoticons ocidentais mais comuns na base (fonte do rótulo — remover!).
#: Cobre :) :-) :( :-( :D :P ;) e variações com nariz.
EMOTICON: Final = re.compile(r"[:;=8xX][-o*']?[)(\][DPpOo/\\|@]+|<3")

#: Sequência de caracteres repetidos 3+ vezes (ex.: "amooooo" -> "amoo").
REPEATED_CHARS: Final = re.compile(r"(.)\1{2,}")

#: Números isolados.
NUMBER: Final = re.compile(r"\b\d+\b")
