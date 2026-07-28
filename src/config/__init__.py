"""Gerenciamento e validação de configuração do projeto.

Módulos
-------
paths
    Centraliza todos os caminhos do projeto com ``pathlib.Path``.
settings
    Carrega e valida os YAMLs de ``configs/`` e o ``.env`` com Pydantic.
logging
    Configura logging com ``RichHandler`` e rotação diária de arquivo.
seed
    Fixa todas as fontes de aleatoriedade para reprodutibilidade.
"""

from src.config.paths import ProjectPaths, get_paths
from src.config.seed import seed_everything
from src.config.settings import Settings, load_settings

__all__ = [
    "ProjectPaths",
    "Settings",
    "get_paths",
    "load_settings",
    "seed_everything",
]
