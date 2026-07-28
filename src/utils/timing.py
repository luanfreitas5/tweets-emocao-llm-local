"""Cronometragem e barras de progresso ``rich``."""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager

from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

logger = logging.getLogger(__name__)


def build_progress() -> Progress:
    """Cria uma barra de progresso ``rich`` com as colunas padrão do projeto.

    Returns
    -------
    rich.progress.Progress
        Barra pronta para uso em ``with build_progress() as progress:``.
    """
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    )


def convert_seconds_to_time(seconds: float | int) -> str:
    """Formata segundos em horas, minutos e segundos.

    Parameters
    ----------
    seconds : float | int
        Número de segundos a ser formatado.

    Returns
    -------
    str
        String formatada no padrão "HH:MM:SS".
    """
    hours, remainder = divmod(int(seconds), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}"


@contextmanager
def timed(label: str) -> Iterator[None]:
    """Cronometra um bloco de código e registra a duração em log.

    Parameters
    ----------
    label : str
        Descrição da etapa cronometrada (usada na mensagem de log).

    Yields
    ------
    None

    Examples
    --------
    >>> with timed("etapa de exemplo"):
    ...     _ = sum(range(10))
    """
    start = time.perf_counter()
    logger.info("Iniciando: %s", label)
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        logger.info("Concluído: %s - %s", label, convert_seconds_to_time(elapsed))
