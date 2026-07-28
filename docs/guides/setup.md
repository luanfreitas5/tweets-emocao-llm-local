# Setup

## Pré-requisitos

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) para gerenciamento de ambiente
- [Ollama](https://ollama.com/) instalado e rodando localmente
- (Opcional) GPU NVIDIA com CUDA para acelerar BERTimbau/embeddings

## Instalação

```bash
# Ambiente + dependências (runtime + dev)
make install

# Dependências da API (opcional)
make api-deps

# Hooks de qualidade
make hooks
```

## Modelo local (Ollama)

```bash
ollama pull llama3.1:8b   # ou o modelo definido em configs/llm.yaml
```

## Dados

Baixe a base [Portuguese Tweets for Sentiment Analysis](https://www.kaggle.com/datasets/augustop/portuguese-tweets-for-sentiment-analysis)
e coloque os CSVs em `data/raw/` (já ignorados pelo Git).
