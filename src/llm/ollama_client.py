"""Cliente fino sobre o pacote ``ollama`` (execução 100% local).

Encapsula a chamada de chat e traduz erros de rede/pacote nas exceções do
domínio. Nenhum dado sai da máquina — a privacidade é garantida por construção.
"""

from __future__ import annotations

import logging
from typing import Any

from src.config.settings import LLMSettings
from src.exceptions.llm import LLMGenerationError, OllamaConnectionError

logger = logging.getLogger(__name__)


class OllamaClient:
    """Cliente de chat para um servidor Ollama local.

    Parameters
    ----------
    settings : LLMSettings
        Configuração validada do LLM (host, modelo, opções, timeout).

    Examples
    --------
    >>> client = OllamaClient(settings)
    >>> client.chat(messages)  # doctest: +SKIP
    'Resumo em pt-BR...'
    """

    def __init__(self, settings: LLMSettings) -> None:
        self.settings = settings
        self._client: Any | None = None

    def _ensure_client(self) -> Any:
        """Instancia o cliente ``ollama`` apontando para o host local.

        Returns
        -------
        Any
            Instância de ``ollama.Client``.

        Raises
        ------
        OllamaConnectionError
            Se o pacote ``ollama`` não estiver disponível.
        """
        if self._client is not None:
            return self._client
        try:
            from ollama import Client

            self._client = Client(host=self.settings.host, timeout=self.settings.timeout)
        except ImportError as error:  # pragma: no cover
            raise OllamaConnectionError("Pacote 'ollama' não instalado. Rode: uv sync") from error
        return self._client

    def chat(self, messages: list[dict[str, str]]) -> str:
        """Envia mensagens ao modelo local e retorna o texto da resposta.

        Parameters
        ----------
        messages : list[dict[str, str]]
            Mensagens no formato de chat (``role``/``content``).

        Returns
        -------
        str
            Conteúdo textual da resposta do modelo.

        Raises
        ------
        OllamaConnectionError
            Se não for possível falar com o servidor Ollama.
        LLMGenerationError
            Se a geração falhar ou a resposta vier vazia/malformada.
        """
        client = self._ensure_client()
        options = {
            "temperature": self.settings.options.temperature,
            "top_p": self.settings.options.top_p,
            "num_ctx": self.settings.options.num_ctx,
            "seed": self.settings.options.seed,
        }
        try:
            logger.info("Gerando resumo com o modelo local %s", self.settings.model)
            response = client.chat(
                model=self.settings.model,
                messages=messages,
                options=options,
            )
        except ConnectionError as error:
            logger.error("Sem conexão com o Ollama em %s", self.settings.host)
            raise OllamaConnectionError(
                f"Não foi possível conectar ao Ollama em {self.settings.host}: {error}"
            ) from error
        except Exception as error:
            logger.error("Falha na geração do LLM: %s", error)
            raise LLMGenerationError(f"Erro na geração do resumo: {error}") from error

        content = response.get("message", {}).get("content", "").strip()
        if not content:
            raise LLMGenerationError("O LLM retornou uma resposta vazia.")
        return content
