# Changelog

Todas as mudanças notáveis deste projeto são documentadas aqui.

O formato segue [Keep a Changelog](https://keepachangelog.com/pt-BR/1.1.0/) e o
projeto adota [Versionamento Semântico](https://semver.org/lang/pt-BR/).

## [Não lançado]

### Adicionado
- Estrutura inicial do projeto (configs, `src/`, `app/`, testes, docs).
- Pipeline "Python calcula, LLM explica":
  - Limpeza de tweets com remoção de *leakage* de emoticons/hashtags.
  - Classificação de sentimento com Transformer pt-BR (BERTimbau/afins).
  - Modelagem de tópicos com embeddings + BERTopic.
  - Agregação de insights em `InsightsReport` (JSON) e resumo via Ollama local.
- API FastAPI local (`/sentiment`, `/summary`, `/health`).
- Contratos de dados (pandera), avaliação com IC 95% e por fatia.
- Model Card e Datasheet.

## [0.1.0] - 2026-07-05

### Adicionado
- Versão inicial do esqueleto do projeto.
