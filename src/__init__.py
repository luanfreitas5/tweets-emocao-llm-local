"""Pacote principal do projeto *tweets-emocao-llm-local*.

Arquitetura central: **"Python calcula, LLM explica"**. Todo o cálculo
(limpeza, classificação de sentimento com BERTimbau, clusterização de tópicos
com embeddings + BERTopic e as métricas) é feito em Python de forma
determinística. Um LLM local (via Ollama) recebe apenas JSON estruturado já
computado e gera resumos em linguagem simples, sem inventar números.

Subpacotes principais
---------------------
config
    Carregamento e validação de configuração, logging, caminhos e sementes.
constants
    Constantes de colunas, rótulos e padrões regex de limpeza de tweets.
data
    Ingestão, escrita e particionamento dos datasets de tweets.
preprocessing
    Limpeza de texto e remoção de vazamento (emoticons/hashtags rotuladores).
models
    Classificador de sentimento, codificador de embeddings e modelo de tópicos.
llm
    Cliente Ollama, prompts e geração de resumos a partir de JSON.
evaluation
    Métricas rigorosas (incerteza, avaliação por fatia).
visualization
    Gráficos com paleta consistente.
pipelines
    Orquestração de cada etapa e do fluxo ponta a ponta.
cli
    Comandos de linha de comando para cada etapa.
schemas
    Contratos de dados (pandera) e modelos Pydantic de I/O do LLM.
utils
    Utilitários compartilhados (I/O, hashing, timing).
"""

__version__ = "0.1.0"
