# Tweets Emoção & Tópicos — LLM Local

Pipeline de análise de **emoção** e **tópicos** em ~800 mil tweets em português.

A arquitetura central é **"Python calcula, LLM explica"**:

- **Python calcula** (determinístico, auditável): limpeza dos tweets,
  classificação de sentimento com BERTimbau, clusterização de tópicos com
  embeddings + BERTopic e todas as métricas.
- **LLM explica** (Ollama local): recebe apenas o JSON estruturado já computado
  e gera um resumo em linguagem simples — **sem inventar números**.

!!! tip "Privacidade por construção"
    Todo o processamento — inclusive o LLM — roda localmente. Nenhum dado sai
    da máquina.

## Fluxo

```
raw CSV → limpeza → sentimento (BERTimbau) → tópicos (BERTopic)
        → InsightsReport (JSON) → resumo (Ollama)
```

Veja o [guia de setup](guides/setup.md), o [guia de uso](guides/usage.md) e as
[decisões de arquitetura](guides/architecture.md).
