# Uso

## Pipeline completo

```bash
# Fluxo ponta a ponta: limpeza → sentimento → tópicos → resumo
make pipeline

# Execução rápida em uma amostra (via argparse)
uv run python -m src.main --sample-size 5000

# Sem a etapa de tópicos (mais rápido)
uv run python -m src.main --no-topics
```

## Etapas isoladas

```bash
make preprocess   # bronze (data/bronze) → silver (data/silver/tweets_processed.parquet)
make classify     # → gold (data/gold/sentiment_predictions.parquet)
make topics       # → gold (data/gold/topic_assignments.parquet) + models/bertopic
make evaluate     # → reports/metrics/classification.json (F1-macro com IC 95% e por fatia)
make plot         # → reports/figures/*.png + *.svg
make summarize    # → reports/insights.json + reports/summary.md
```

## API local

```bash
make api          # http://127.0.0.1:8000/docs
```

Endpoints principais:

- `POST /sentiment` — classifica textos avulsos.
- `POST /summary` — gera resumo a partir de um `InsightsReport` estruturado.
- `GET /health` — processo ativo, modelo LLM e se o modelo de sentimento está carregado.
- `GET /version` — versão da API, dos modelos e SHA do commit.

## Rastreamento (MLflow)

```bash
make mlflow       # pipeline completo registrado no MLflow (exige árvore Git limpa)
make mlflow-ui    # UI sobre o backend SQLite (sqlite:///mlruns/mlflow.db)
```
