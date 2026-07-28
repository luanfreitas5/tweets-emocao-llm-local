# --- Configuração ----------------------------------------------------------
PYTHON := python
UV := uv
RUN := $(UV) run python src/main.py    # 'src' vira raiz do path ao rodar o script

.DEFAULT_GOAL := help
.PHONY: help init venv install install-all install-models  update lock export \
	check format lint typecheck security deadcode complexity docstrings refurb quality \
	test smoke hooks pre-commit update-hooks update-version docs docs-serve docs-deploy profile clean cache jupyter notebook add remove tree \
	clean-processed clean-reports clean-outputs clean-notebooks \
	preprocess classify topics summarize pipeline api mlflow

help:  ## Lista os alvos disponíveis
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

init:  ## Inicializa o projeto (instala dependências + hooks)
	$(MAKE) install
	$(MAKE) hooks

venv:  ## Cria o ambiente virtual (requer: uv)
	$(UV) venv

install:  ## Instala dependências (runtime + dev)
	$(UV) sync --dev

install-all:  ## Instala tudo (todos os extras + dev)
	$(UV) sync --all-extras --dev

install-models:  ## Instala os extras dos modelos Hugging Face (torch + transformers)
	$(UV) sync --extra models --dev

update:  ## Atualiza todas as dependências e sincroniza
	$(UV) lock --upgrade
	$(UV) sync --all-groups

lock:
	$(UV) lock

export:
	$(UV) export --no-hashes -o requirements.txt

# --- Qualidade -------------------------------------------------------------
check:  ## Checa formatação com ruff
	$(UV) run ruff check .

format:  ## Formata o código com ruff
	$(UV) run ruff format .

lint: ## Lint com ruff
	$(UV) run ruff check --fix .

typecheck:  ## Type checking estático (basedPyright)
	$(UV) run basedpyright

security:  ## Análise de segurança (bandit + pip-audit)
	$(UV) run bandit -r src -c pyproject.toml
	$(UV) run pip-audit

deadcode:  ## Detecta código morto (vulture)
	$(UV) run vulture src

complexity:  ## Limites de complexidade (xenon)
	$(UV) run xenon --max-absolute B --max-modules A --max-average A src

docstrings:  ## Cobertura de docstrings (interrogate)
	$(UV) run interrogate -v src

refurb:  ## Detecta código redundante (refurb)
	$(UV) run refurb src

quality: format lint typecheck security deadcode complexity docstrings refurb   ## Roda toda a suíte de qualidade (espelha o CI)

# --- Testes ----------------------------------------------------------------
test:  ## Roda os testes com cobertura
	$(UV) run pytest -m "not slow"

smoke:  ## Roda apenas os smoke tests
	$(UV) run pytest -m smoke -q

hooks:  ## Instala os hooks do pre-commit
	$(UV) run pre-commit install
	$(UV) run pre-commit install --hook-type commit-msg
	$(UV) run detect-secrets scan > .secrets.baseline

pre-commit:  ## Roda todos os hooks do pre-commit em todos os arquivos
	$(UV) run pre-commit run --all-files

update-hooks:  ## Atualiza os hooks do pre-commit
	$(UV) run pre-commit autoupdate

release:  ## Cria uma nova release (versão + changelog + tag)
	$(UV) run cz changelog
	$(UV) run cz bump --changelog --yes

# --- Limpeza de saídas do pipeline ------------------------------------------
clean-processed:  ## Remove os artefatos de dados processados
	rm -rf data/processed/*.parquet

clean-reports:  ## Remove os relatórios gerados (pastas por modelo + comparação)
	find reports -mindepth 1 -maxdepth 1 -type d -exec rm -rf {} +

clean-outputs: clean-processed clean-reports  ## Remove todas as saídas do pipeline

clean-notebooks:  ## Remove os notebooks com células vazias
	$(UV) run nbstripout notebooks

# --- Documentação ----------------------------------------------------------
docs:  ## Constrói a documentação (modo estrito)
	$(UV) run mkdocs build --strict

docs-serve:  ## Servidor local da documentação
	$(UV) run mkdocs serve

docs-deploy:  ## Publica a documentação no GitHub Pages
	$(UV) run mkdocs gh-deploy --force

# --- Utilitários -----------------------------------------------------------
profile:  ## Exemplo de profiling com scalene (ajuste o alvo)
	$(UV) run scalene src/main.py

clean:  ## Remove caches e artefatos temporários
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov coverage.xml site
	find . -type d -name __pycache__ -exec rm -rf {} +

cache:
	$(UV) cache clean

# --- Jupyter ----------------------------------------------------------------
jupyter:
	$(UV) run jupyter lab

notebook:
	$(UV) run jupyter notebook

# --- Gerenciamento de pacotes -----------------------------------------------
add:
	$(UV) add $(PKG)

remove:
	$(UV) remove $(PKG)

tree:
	$(UV) tree

# --- Pipeline do projeto (Python calcula, LLM explica) ---------------------
preprocess:  ## Limpa os tweets brutos e grava o parquet processado (remove leakage de emoticons)
	$(RUN) --stage preprocess

classify:  ## Classifica o sentimento dos tweets com BERTimbau
	$(RUN) --stage classify

topics:  ## Extrai tópicos com embeddings + BERTopic
	$(RUN) --stage topics

summarize:  ## Gera resumos em linguagem simples via LLM local (Ollama) a partir do JSON estruturado
	$(RUN) --stage summarize

pipeline:  ## Executa o fluxo ponta a ponta (preprocess -> classify -> topics -> summarize)
	$(RUN) --stage all

api:  ## Sobe a API FastAPI localmente (requer: make api-deps)
	$(UV) run uvicorn app.main:app --reload --host 127.0.0.1 --port 8000

mlflow:  ## Sobe a UI do MLflow local (tracking)
	$(UV) run mlflow ui --backend-store-uri ./mlruns
