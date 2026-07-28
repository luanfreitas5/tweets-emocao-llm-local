# =============================================================================
# Dockerfile — imagem da API FastAPI (multi-stage, usuário não-root).
# O LLM roda no Ollama do host (ver docker-compose.yml). NÃO inclui GPU/torch
# pesado por padrão — ajuste a base se precisar de CUDA.
# =============================================================================

# --- Stage 1: build do ambiente ---------------------------------------------
FROM python:3.12-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

# Instala apenas as dependências primeiro (cache eficiente).
COPY pyproject.toml uv.lock* ./
RUN uv sync --extra api --no-install-project --no-dev || uv sync --extra api --no-dev

# Copia o código e instala o projeto.
COPY src ./src
COPY app ./app
COPY configs ./configs
RUN uv sync --extra api --no-dev

# --- Stage 2: runtime enxuto ------------------------------------------------
FROM python:3.12-slim AS runtime

# Usuário não-root.
RUN useradd --create-home --uid 1000 appuser
WORKDIR /app

COPY --from=builder --chown=appuser:appuser /app /app
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    OLLAMA_HOST="http://host.docker.internal:11434"

USER appuser
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
