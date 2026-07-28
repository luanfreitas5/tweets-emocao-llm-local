<h1 align="center">🐦 Tweets Emoção & Tópicos — LLM Local</h1>

<p align="center">
  <em>Análise de emoção e tópicos em ~800 mil tweets em português.<br>
  <strong>Python calcula, LLM explica.</strong> Tudo roda localmente. 🔒</em>
</p>

<p align="center">
  <img alt="CI" src="https://img.shields.io/badge/CI-GitHub_Actions-2088FF?logo=githubactions&logoColor=white">
  <img alt="Python" src="https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white">
  <img alt="uv" src="https://img.shields.io/badge/env-uv-DE5FE9?logo=astral&logoColor=white">
  <img alt="Ruff" src="https://img.shields.io/badge/lint-ruff-D7FF64?logo=ruff&logoColor=black">
  <img alt="Coverage" src="https://img.shields.io/badge/coverage-%E2%89%A580%25-success">
  <img alt="License" src="https://img.shields.io/badge/license-MIT-green">
</p>

---

## 💡 A ideia

LLMs são ótimos para **escrever**, mas péssimos para **calcular** — inventam
números com confiança. Este projeto separa as responsabilidades:

| Camada | Faz o quê | Determinístico? |
|---|---|:---:|
| 🐍 **Python** | Limpa, classifica sentimento (BERTimbau), agrupa tópicos (BERTopic), calcula todas as métricas | ✅ |
| 🤖 **LLM local (Ollama)** | Recebe só o **JSON já calculado** e escreve um resumo em linguagem simples | ⛔ (restrito por prompt) |

O contrato formal entre as duas camadas é o
[`InsightsReport`](src/schemas/insights.py): **todo número que o LLM cita já foi
calculado em Python**. Resultado: um relatório *confiável* **e** *legível*.

## 🏗️ Fluxo

```
data/raw/*.csv
   └─▶ limpeza (anti-leakage)  ──▶ data/processed/tweets_processed.parquet
        └─▶ sentimento (BERTimbau)  ──▶ sentiment_predictions.parquet
             └─▶ tópicos (embeddings + BERTopic)  ──▶ topic_assignments.parquet
                  └─▶ InsightsReport (JSON)  ──▶  reports/insights.json
                       └─▶ resumo (Ollama)   ──▶  reports/summary.md
```

## 🚀 Início rápido

```bash
make install          # ambiente + dependências (uv)
ollama pull llama3.1:8b
# coloque os CSVs da base em data/raw/

make pipeline                             # fluxo completo
uv run python -m src.main --sample-size 5000   # execução rápida
make api                                  # API local em http://127.0.0.1:8000/docs
```

> 📥 **Base:** [Portuguese Tweets for Sentiment Analysis](https://www.kaggle.com/datasets/augustop/portuguese-tweets-for-sentiment-analysis) (~800k tweets, pt-BR).

## 🧠 Decisões de engenharia (senior bar)

- **Anti-leakage:** os rótulos vêm de *supervisão distante* (emoticons `:)`/`:(`,
  hashtags `#fato`) que aparecem no próprio texto. A limpeza os **remove** para o
  modelo aprender conteúdo, não o atalho. → [`remove_label_leakage`](src/preprocessing/cleaning.py)
- **Validação de dados:** contratos [`pandera`](src/schemas/) na entrada e saída de
  cada etapa — falha cedo, com erro claro.
- **Avaliação rigorosa:** F1-macro **com intervalo de confiança** (bootstrap) e
  **por fatia** — nunca um número solto. → [`src/evaluation`](src/evaluation/)
- **Reprodutibilidade:** `seed_everything`, hash do dataset, `uv.lock` commitado.
- **Privacidade / LGPD:** base pública anônima; processamento 100% local; nenhum
  `@usuário`/link nos relatórios.

## 🧰 Stack

`polars` · `transformers` (BERTimbau) · `sentence-transformers` · `BERTopic` ·
`ollama` · `pydantic v2` · `pandera` · `FastAPI` · `mlflow` · `pytest` + `hypothesis`

## 📁 Estrutura

```
configs/     YAML validado por Pydantic (config, paths, logging, model_params, llm, deploy)
src/
  preprocessing/  limpeza + anti-leakage        models/     BERTimbau, embeddings, BERTopic
  analysis/       InsightsReport (Python calcula) llm/       Ollama, prompts, summarizer
  evaluation/     métricas com IC e por fatia     pipelines/ etapas + fluxo ponta a ponta
  schemas/        contratos pandera + Pydantic    cli/       comandos por etapa
app/          API FastAPI (/sentiment, /summary, /health)
tests/        unit + property-based + smoke
docs/         MkDocs Material  ·  reports/  Model Card + Datasheet
```

## 🛠️ Comandos úteis

```bash
make quality   # lint + type + segurança + complexidade + docstrings (espelha o CI)
make test      # pytest com cobertura
make docs-serve
make help      # todos os alvos
```

## 📄 Licença

[MIT](LICENSE) © 2026 Luan Freitas
