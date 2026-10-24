"""Testes das fábricas de handlers do logging (sem chamar ``configure_logging``).

``configure_logging`` substitui os handlers do root e esvaziaria o ``caplog`` do
pytest; por isso os testes exercitam apenas as funções puras e as fábricas.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.config import logging as log_config
from src.config.paths import CONFIGS_DIR


@pytest.mark.smoke
def test_rotated_file_is_renamed_to_daily_pattern():
    """O arquivo rotacionado vira ``log_YYYY-MM-DD.log``."""
    renamed = log_config._rename_rotated(str(Path("logs") / "log.log.2026-09-27"))
    assert Path(renamed).name == "log_2026-09-27.log"


def test_daily_file_handler_creates_directory(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """A fábrica cria o diretório de logs (dictConfig falharia se ele não existisse)."""
    monkeypatch.setattr(log_config, "ROOT", tmp_path)
    handler = log_config.create_daily_file_handler("logs/log.log", backup_count=3)
    try:
        assert (tmp_path / "logs").is_dir()
        assert handler.backupCount == 3
    finally:
        handler.close()


@pytest.mark.smoke
def test_logging_yaml_keeps_existing_loggers_enabled():
    """``disable_existing_loggers: false`` é obrigatório; handlers só no root."""
    config = yaml.safe_load((CONFIGS_DIR / "logging.yaml").read_text(encoding="utf-8"))
    assert config["disable_existing_loggers"] is False
    assert config["root"]["handlers"] == ["console", "file"]
    assert all("handlers" not in logger for logger in config["loggers"].values())


def test_configure_logging_missing_yaml_raises(tmp_path: Path):
    """YAML ausente levanta ``FileNotFoundError`` antes de tocar nos handlers."""
    with pytest.raises(FileNotFoundError):
        log_config.configure_logging(config_path=tmp_path / "nao_existe.yaml")
