# =============================================================================
# Makefile — atalhos coerentes com pyproject.toml / pre-commit / CI.
# Requer uv instalado. Recipes usam TAB (exigência do make).
# =============================================================================
.DEFAULT_GOAL := help
.PHONY: help install install-all hooks format lint typecheck security \
        deadcode complexity docstrings quality test smoke docs docs-serve \
        docs-deploy precommit profile train clean

help:  ## Lista os alvos disponíveis
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install:  ## Instala dependências do projeto (sem sincronizar grupos opcionais)
	uv sync

install-all:  ## Instala tudo (todos os grupos de dependências)
	uv sync --all-groups

hooks:  ## Instala os hooks do pre-commit
	uv run pre-commit install
	uv run pre-commit install --hook-type commit-msg

update-hooks:  ## Atualiza os hooks do pre-commit
	uv run pre-commit autoupdate

update:  ## Atualiza todas as dependências e sincroniza
	uv lock --upgrade
	uv sync --all-groups

# --- Qualidade -------------------------------------------------------------
check:  ## Checa formatação com ruff
	uv run ruff check .

format:  ## Formata o código com ruff
	uv run ruff format .

lint:  ## Lint com ruff
	uv run ruff format --check .

typecheck:  ## Type checking estático (basedPyright)
	uv run basedpyright

security:  ## Análise de segurança (bandit + pip-audit)
	uv run bandit -r src -c pyproject.toml
	uv run pip-audit

deadcode:  ## Detecta código morto (vulture)
	uv run vulture src

complexity:  ## Limites de complexidade (xenon)
	uv run xenon --max-absolute B --max-modules A --max-average A src

docstrings:  ## Cobertura de docstrings (interrogate)
	uv run interrogate -v src

quality: format lint typecheck security deadcode complexity docstrings  ## Roda toda a suíte de qualidade (espelha o CI)

# --- Testes ----------------------------------------------------------------
test:  ## Roda os testes com cobertura
	uv run pytest -m "not slow"

smoke:  ## Roda apenas os smoke tests
	uv run pytest -m smoke -q

precommit:  ## Roda todos os hooks do pre-commit em todos os arquivos
	uv run pre-commit run --all-files

# --- Documentação ----------------------------------------------------------
docs:  ## Constrói a documentação (modo estrito)
	uv run mkdocs build --strict

docs-serve:  ## Servidor local da documentação
	uv run mkdocs serve

docs-deploy:  ## Publica a documentação no GitHub Pages
	uv run mkdocs gh-deploy --force

# --- Utilitários -----------------------------------------------------------
profile:  ## Exemplo de profiling com scalene (ajuste o alvo)
	uv run scalene src/main.py

clean:  ## Remove caches e artefatos temporários
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov coverage.xml site
	find . -type d -name __pycache__ -exec rm -rf {} +

# --- Pipeline do projeto (Python calcula, LLM explica) ---------------------
preprocess:  ## Limpa os tweets brutos e grava o parquet processado (remove leakage de emoticons)
	uv run python -m src.cli.preprocess

classify:  ## Classifica o sentimento dos tweets com BERTimbau
	uv run python -m src.cli.classify

topics:  ## Extrai tópicos com embeddings + BERTopic
	uv run python -m src.cli.topics

summarize:  ## Gera resumos em linguagem simples via LLM local (Ollama) a partir do JSON estruturado
	uv run python -m src.cli.summarize

pipeline:  ## Executa o fluxo ponta a ponta (preprocess -> classify -> topics -> summarize)
	uv run python -m src.main

api:  ## Sobe a API FastAPI localmente (requer: make api-deps)
	uv run uvicorn app.main:app --reload --host 127.0.0.1 --port 8000

mlflow:  ## Sobe a UI do MLflow local (tracking)
	uv run mlflow ui --backend-store-uri ./mlruns