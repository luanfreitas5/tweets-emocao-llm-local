"""Carregamento e validação de configuração com Pydantic.

Os YAMLs de ``configs/`` são carregados e validados em runtime: uma configuração
inválida falha no *startup* com um erro tipado e claro — nunca no meio da
execução. Segredos vêm apenas do ``.env`` (nunca commitados).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.config.paths import CONFIGS_DIR


class SentimentParams(BaseModel):
    """Hiperparâmetros do classificador de sentimento (BERTimbau/HF)."""

    model_config = {"protected_namespaces": ()}

    model_name: str
    batch_size: int = Field(gt=0, default=64)
    max_length: int = Field(gt=0, default=128)
    device: Literal["auto", "cpu", "cuda"] = "auto"
    label_map: dict[str, str] = Field(default_factory=dict)


class EmbeddingParams(BaseModel):
    """Hiperparâmetros do codificador de embeddings de sentença."""

    model_config = {"protected_namespaces": ()}

    model_name: str
    batch_size: int = Field(gt=0, default=128)
    normalize: bool = True


class TopicParams(BaseModel):
    """Hiperparâmetros da modelagem de tópicos (BERTopic)."""

    language: str = "multilingual"
    min_topic_size: int = Field(gt=1, default=50)
    nr_topics: int | Literal["auto"] = "auto"
    top_n_words: int = Field(gt=0, default=10)
    calculate_probabilities: bool = False
    low_memory: bool = True


class EvaluationParams(BaseModel):
    """Parâmetros de avaliação rigorosa (incerteza + regressão de métrica)."""

    primary_metric: str = "f1_macro"
    cv_folds: int = Field(gt=1, default=5)
    bootstrap_iterations: int = Field(gt=0, default=1000)
    min_f1_macro: float = Field(ge=0, le=1, default=0.70)


class ModelParams(BaseModel):
    """Agrupa os hiperparâmetros de ``configs/model_params.yaml``."""

    sentiment: SentimentParams
    embeddings: EmbeddingParams
    topics: TopicParams
    evaluation: EvaluationParams


class LLMOptions(BaseModel):
    """Opções de geração do LLM local (determinismo/anti-alucinação)."""

    temperature: float = Field(ge=0, le=2, default=0.1)
    top_p: float = Field(ge=0, le=1, default=0.9)
    num_ctx: int = Field(gt=0, default=8192)
    seed: int = 42


class LLMSettings(BaseModel):
    """Configuração do LLM local (Ollama) para geração de resumos."""

    provider: Literal["ollama"] = "ollama"
    host: str = "http://127.0.0.1:11434"
    model: str = "llama3.1:8b"
    options: LLMOptions = LLMOptions()
    timeout: int = Field(gt=0, default=120)
    output_language: str = "pt-BR"

    model_config = {"protected_namespaces": ()}


class Settings(BaseSettings):
    """Configuração global do projeto, carregada de YAML + ``.env``, validada.

    Attributes
    ----------
    project_name : str
        Nome do projeto.
    random_seed : int
        Semente global de reprodutibilidade.
    language : str
        Idioma dos textos processados.
    sample_size : int | None
        Amostragem para execuções rápidas; ``None`` usa toda a base.
    log_level : str
        Nível de log padrão.
    model : ModelParams
        Hiperparâmetros dos modelos computacionais.
    llm : LLMSettings
        Configuração do LLM local (apenas resumo).

    Examples
    --------
    >>> settings = load_settings()
    >>> settings.random_seed
    42
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_nested_delimiter="__",
        extra="ignore",
        protected_namespaces=(),
    )

    project_name: str = "tweets-emocao-llm-local"
    random_seed: int = 42
    language: str = "pt"
    sample_size: int | None = None
    log_level: str = "INFO"

    model: ModelParams
    llm: LLMSettings


def _read_yaml(path: Path) -> dict[str, Any]:
    """Lê um YAML e retorna um dicionário; erro claro se ausente."""
    if not path.exists():
        raise FileNotFoundError(f"Arquivo de configuração não encontrado: {path}")
    with path.open(encoding="utf-8") as handler:
        return yaml.safe_load(handler) or {}


def load_settings(configs_dir: Path = CONFIGS_DIR) -> Settings:
    """Carrega e valida todas as configurações do projeto.

    Combina ``config.yaml``, ``model_params.yaml`` e ``llm.yaml`` em um único
    objeto :class:`Settings` validado por Pydantic.

    Parameters
    ----------
    configs_dir : Path, optional
        Diretório com os YAMLs, by default ``configs/``.

    Returns
    -------
    Settings
        Configuração validada e pronta para uso.

    Raises
    ------
    FileNotFoundError
        Se algum YAML obrigatório estiver ausente.
    pydantic.ValidationError
        Se algum valor violar o contrato de configuração.
    """
    general = _read_yaml(configs_dir / "config.yaml")
    model_params = _read_yaml(configs_dir / "model_params.yaml")
    llm_params = _read_yaml(configs_dir / "llm.yaml")

    return Settings(model=ModelParams(**model_params), llm=LLMSettings(**llm_params), **general)
