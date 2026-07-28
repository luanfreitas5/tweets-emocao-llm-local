"""Configuração de logging com ``RichHandler`` e rotação diária de arquivo.

Todas as mensagens de log do projeto são em pt-BR. Console usa ``rich`` (cor +
tracebacks); o arquivo usa formato separado por TAB com rotação à meia-noite.
"""

from __future__ import annotations

import logging
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

import yaml
from rich.logging import RichHandler

from src.config.paths import CONFIGS_DIR, ROOT

LOGGING_YAML: Path = CONFIGS_DIR / "logging.yaml"


def configure_logging(logging_yaml: Path = LOGGING_YAML, root: Path = ROOT) -> logging.Logger:
    """Configura os handlers de console (rich) e arquivo (rotação diária).

    Parameters
    ----------
    logging_yaml : Path, optional
        Arquivo de configuração de logging, by default ``configs/logging.yaml``.
    root : Path, optional
        Raiz do projeto (para resolver o diretório de logs).

    Returns
    -------
    logging.Logger
        Logger raiz já configurado.

    Examples
    --------
    >>> logger = configure_logging()
    >>> logger.info("Pipeline iniciado")
    """
    with logging_yaml.open(encoding="utf-8") as handler:
        cfg = yaml.safe_load(handler)

    level = logging.getLevelName(cfg.get("level", "INFO"))
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.handlers.clear()

    console_cfg = cfg.get("console", {})
    console_handler = RichHandler(
        rich_tracebacks=console_cfg.get("rich_tracebacks", True),
        show_path=console_cfg.get("show_path", True),
        show_time=console_cfg.get("show_time", True),
    )
    console_handler.setFormatter(logging.Formatter("%(name)s \t %(message)s"))
    root_logger.addHandler(console_handler)

    file_cfg = cfg.get("file", {})
    if file_cfg.get("enabled", True):
        logs_dir = root / file_cfg.get("directory", "logs")
        logs_dir.mkdir(parents=True, exist_ok=True)
        filename = datetime.now().strftime(file_cfg.get("filename_pattern", "log_%Y-%m-%d.log"))
        file_handler = TimedRotatingFileHandler(
            filename=logs_dir / filename,
            when=file_cfg.get("when", "midnight"),
            backupCount=file_cfg.get("backup_count", 14),
            encoding=file_cfg.get("encoding", "utf-8"),
        )
        file_handler.setFormatter(
            logging.Formatter(
                file_cfg.get("format", "%(asctime)s \t %(levelname)s \t %(name)s \t %(message)s")
            )
        )
        root_logger.addHandler(file_handler)

    return root_logger


def get_logger(name: str) -> logging.Logger:
    """Retorna um logger nomeado (atalho para ``logging.getLogger``).

    Parameters
    ----------
    name : str
        Nome do logger, tipicamente ``__name__``.

    Returns
    -------
    logging.Logger
        Logger nomeado.
    """
    return logging.getLogger(name)
