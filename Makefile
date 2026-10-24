# Atalhos reprodutíveis. `make check` replica o CI localmente.
# Dados: bronze (como recebido) -> silver (limpo/validado) -> gold (predições/tópicos).
UV := uv
RUN := $(UV) run python -m src.main
SAMPLE_SIZE ?= 5000

.DEFAULT_GOAL := help
.PHONY: help install ollama lint format typecheck security quality test smoke check hooks pre-commit \
	update lock release docs docs-serve docs-deploy profile clean \
	preprocess classify topics evaluate plot summarize pipeline sample train reproduce \
	api mlflow mlflow-ui docker-config docker-build docker-up docker-down

help:  ## Lista os alvos disponíveis
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

# --- Ambiente ---------------------------------------------------------------
install:  ## Instala todos os grupos + extra da API + hooks do git
	$(UV) sync --locked --all-groups --extra api
	$(UV) run pre-commit install --hook-type pre-commit --hook-type commit-msg

ollama:  ## Instala o llama3.1:8b
	$(UV) run ollama pull llama3.1:8b

update:  ## Atualiza dependências e o uv.lock (commite o lock)
	$(UV) lock --upgrade
	$(UV) sync --all-groups --extra api

lock:  ## Regenera o uv.lock sem atualizar versões
	$(UV) lock

# --- Qualidade --------------------------------------------------------------
lint:  ## Lint + verificação de formatação (ruff)
	$(UV) run ruff check src app tests
	$(UV) run ruff format --check src app tests

format:  ## Corrige lint e formata o código (ruff)
	$(UV) run ruff check --fix src app tests
	$(UV) run ruff format src app tests

typecheck:  ## Type checking estático (basedPyright) — CI
	$(UV) run basedpyright

security:  ## Segurança: bandit + pip-audit — CI
	$(UV) run bandit -c pyproject.toml -r src
	$(UV) run pip-audit

quality:  ## Código morto, modernização, complexidade e docstrings — CI
	$(UV) run vulture
	$(UV) run refurb src
	$(UV) run xenon --max-absolute B --max-modules A --max-average A src
	$(UV) run interrogate -v src

test:  ## Testes com cobertura (mínimo 80%)
	$(UV) run pytest -m "not slow" --cov --cov-report=term-missing --cov-report=xml

smoke:  ## Smoke tests (< 10 s), os mesmos do pre-commit
	$(UV) run pytest -m smoke -q --no-cov

check: lint typecheck security quality test  ## Replica o CI localmente

pre-commit:  ## Roda todos os hooks do pre-commit em todos os arquivos
	$(UV) run pre-commit run --all-files

release:  ## Nova versão (semver) + CHANGELOG + tag, via commitizen
	$(UV) run cz bump --changelog --yes

# --- Pipeline (Python calcula, LLM explica) ---------------------------------
preprocess:  ## bronze (CSV) -> silver (parquet validado)
	$(RUN) --stage preprocess

classify:  ## silver -> gold: sentimento (Transformer pt-BR)
	$(RUN) --stage classify

topics:  ## gold: tópicos (embeddings + BERTopic)
	$(RUN) --stage topics

evaluate:  ## F1-macro com IC 95% e por fatia -> reports/metrics/classification.json
	$(RUN) --stage evaluate

plot:  ## Figuras (png 300 dpi + svg) em reports/figures
	$(RUN) --stage plot

summarize:  ## insights.json + resumo em linguagem simples (requer Ollama)
	$(RUN) --stage summarize

pipeline:  ## Fluxo ponta a ponta (requer Ollama para o resumo)
	$(RUN) --stage all

sample:  ## Fluxo rápido sem LLM nem tópicos (amostra de SAMPLE_SIZE tweets)
	$(RUN) --stage preprocess --sample-size $(SAMPLE_SIZE)
	$(RUN) --stage classify
	$(RUN) --stage evaluate
	$(RUN) --stage plot

train: pipeline  ## Alias de `pipeline` (não há ajuste fino: o classificador é pré-treinado)

reproduce: sample  ## Reproduz os números do README em escala reduzida

# --- Serviços ---------------------------------------------------------------
api:  ## API FastAPI local em http://127.0.0.1:8000/docs
	$(UV) run uvicorn app.main:app --reload --host 127.0.0.1 --port 8000

mlflow:  ## Pipeline completo registrando o run no MLflow (exige árvore Git limpa)
	$(RUN) --stage all --track

mlflow-ui:  ## UI do MLflow (backend SQLite)
	$(UV) run mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db

docker-config:  ## Valida o docker-compose.yml
	docker compose config -q

docker-build:  ## Constrói a imagem da API (SHA do commit embutido em /version)
	GIT_SHA=$$(git rev-parse HEAD) docker compose build api

docker-up:  ## Sobe API + Ollama + MLflow
	docker compose up -d

docker-down:  ## Derruba os serviços
	docker compose down

# --- Documentação e utilitários ---------------------------------------------
docs:  ## Constrói a documentação (modo estrito)
	$(UV) run mkdocs build --strict

docs-serve:  ## Servidor local da documentação
	$(UV) run mkdocs serve

docs-deploy:  ## Publica a documentação no GitHub Pages
	$(UV) run mkdocs gh-deploy --force

profile:  ## Profiling de CPU/memória com scalene
	$(UV) run --group profiling scalene src/main.py

clean:  ## Remove caches e artefatos temporários
	rm -rf .pytest_cache .ruff_cache .coverage coverage.xml htmlcov site
	find . -type d -name __pycache__ -not -path "./.venv/*" -exec rm -rf {} +
