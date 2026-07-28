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
make preprocess   # data/raw → data/processed/tweets_processed.parquet
make classify     # → sentiment_predictions.parquet
make topics       # → topic_assignments.parquet + models/bertopic
make summarize    # → reports/insights.json + reports/summary.md
```

## API local

```bash
make api          # http://127.0.0.1:8000/docs
```

Endpoints principais:

- `POST /sentiment` — classifica textos avulsos.
- `POST /summary` — gera resumo a partir de um `InsightsReport` estruturado.
- `GET /health` — estado do serviço e modelo LLM.
