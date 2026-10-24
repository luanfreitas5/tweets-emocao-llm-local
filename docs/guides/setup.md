# Setup

## Pré-requisitos

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) para gerenciamento de ambiente
- [Ollama](https://ollama.com/) instalado e rodando localmente
- (Opcional) GPU NVIDIA com CUDA para acelerar BERTimbau/embeddings

## Instalação

```bash
# Ambiente (todos os grupos de dependências) + hooks do git (pre-commit e commit-msg)
make install
```

## Modelo local (Ollama)

```bash
ollama pull llama3.1:8b   # ou o modelo definido em configs/llm.yaml
```

## Dados

Baixe a base [Portuguese Tweets for Sentiment Analysis](https://www.kaggle.com/datasets/augustop/portuguese-tweets-for-sentiment-analysis)
e coloque os CSVs em `data/bronze/` (camada bronze: dados como recebidos, imutáveis e ignorados pelo Git).
As camadas seguintes são geradas pelo pipeline: `data/silver/` (limpos e validados) e `data/gold/` (predições e tópicos).
