"""Testes de carregamento e validação da configuração."""

from __future__ import annotations

import pytest

from src.config.paths import get_paths
from src.config.settings import load_settings


@pytest.mark.smoke
def test_load_settings_from_configs():
    """A configuração real do projeto carrega e valida sem erros."""
    settings = load_settings()
    assert settings.random_seed == 42
    assert settings.model.evaluation.primary_metric == "f1_macro"
    assert settings.llm.provider == "ollama"


@pytest.mark.smoke
def test_paths_resolve_to_absolute():
    """Os caminhos do projeto resolvem para absolutos e coerentes."""
    paths = get_paths()
    assert paths.processed_tweets.is_absolute()
    assert paths.processed_tweets.suffix == ".parquet"
    assert paths.root.name == "tweets-emocao-llm-local"
