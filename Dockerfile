# syntax=docker/dockerfile:1.7
# Imagem da API: código + dependências travadas. Os pesos do modelo NÃO entram aqui:
# o Transformer de sentimento é baixado do Hugging Face no primeiro startup e fica no
# volume `hf_cache` (HF_HOME=/hf), então trocar de modelo (configs/model_params.yaml)
# nunca exige uma nova imagem. O LLM roda no serviço Ollama (ver docker-compose.yml).
# Fixe as tags por patch; em produção acrescente @sha256:<digest>
# (obtido com `docker buildx imagetools inspect <imagem>:<tag>`). Mesma tag nos dois estágios.
#
# Tamanho: o torch resolvido pelo uv.lock vem do índice CUDA (pytorch-cu129) e deixa a
# imagem com vários GB. Para servir só em CPU, aponte `[tool.uv.sources]` para o índice
# CPU e rode `uv lock` antes do build. Meça com `docker images` e registre no README.

# ---------- Estágio 1: binário do uv, versão fixa ----------
FROM ghcr.io/astral-sh/uv:0.8.15 AS uv

# ---------- Estágio 2: resolve o ambiente a partir do uv.lock ----------
FROM python:3.12.11-slim-bookworm AS builder
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0
WORKDIR /app
# Só os manifestos: esta camada fica em cache até o lock mudar.
# --locked falha se o uv.lock estiver desatualizado em relação ao pyproject.toml.
# --no-dev mantém fora ferramentas de desenvolvimento; --extra api traz fastapi/uvicorn.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    uv sync --locked --no-dev --extra api --no-install-project

# ---------- Estágio 3: runtime enxuto, sem uv, sem compiladores ----------
FROM python:3.12.11-slim-bookworm AS runtime
ARG GIT_SHA=desconhecido
# libgomp1: runtime OpenMP exigido por torch/scikit-learn; ausente na imagem slim.
# hadolint ignore=DL3008
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid app --no-create-home --shell /usr/sbin/nologin app \
    # cache do Hugging Face: pertence ao usuário da app; o volume nomeado herda o dono
    && mkdir -p /hf \
    && chown app:app /hf
WORKDIR /app
# Código e venv pertencem ao root: o usuário da aplicação só lê
COPY --from=builder /app/.venv /app/.venv
COPY src/ src/
COPY app/ app/
COPY configs/ configs/
# O código importa `src.*` e `app.*` a partir da raiz do projeto.
ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONPATH="/app" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME="/hf" \
    GIT_SHA="${GIT_SHA}"
USER app:app
EXPOSE 8000
# /health responde 200 quando o processo está de pé; `model_loaded` indica o modelo em memória.
# start-period alto: o preload baixa/carrega o modelo no primeiro startup.
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
