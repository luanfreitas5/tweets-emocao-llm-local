"""Reprodutibilidade: fixa todas as fontes de aleatoriedade.

``random_state`` sozinho não é reprodutibilidade. Aqui semeamos ``random``,
``numpy``, ``PYTHONHASHSEED`` e, se presente, ``torch``.
"""

from __future__ import annotations

import os
import random
from contextlib import suppress

import numpy as np

RANDOM_SEED = 42


def seed_everything(seed: int = RANDOM_SEED) -> None:
    """Fixa todas as fontes de aleatoriedade para garantir reprodutibilidade.

    Parameters
    ----------
    seed : int, optional
        Semente a aplicar, by default :data:`RANDOM_SEED`.

    Examples
    --------
    >>> seed_everything(42)
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    rng = np.random.default_rng(seed)
    rng.normal()

    with suppress(ImportError):  # opcional: só se torch estiver instalado
        import torch  # noqa: PLC0415

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
