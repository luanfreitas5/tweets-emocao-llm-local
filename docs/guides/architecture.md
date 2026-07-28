# Decisões de Arquitetura

## "Python calcula, LLM explica"

O LLM é uma fonte notória de alucinação numérica. Para obter um relatório
confiável **e** legível, separamos responsabilidades:

| Camada | Responsabilidade | Determinístico? |
|---|---|---|
| Python | Limpeza, sentimento, tópicos, métricas | Sim |
| LLM (Ollama) | Traduzir o JSON em texto simples | Não (mas restrito) |

O contrato formal é o [`InsightsReport`](../reference.md): **todo número que o
LLM menciona já foi calculado em Python**. O prompt de sistema proíbe inventar
dados, e a temperatura baixa reduz a variação.

## Anti-leakage (decisão crítica)

Os rótulos da base vêm de **supervisão distante**: emoticons (`:)`/`:(`) e
hashtags (`#fato`) na coluna `query_used` definiram o sentimento — e esses mesmos
símbolos aparecem no texto. Deixá-los faria o classificador aprender o atalho, não
o conteúdo. A função `remove_label_leakage` os elimina antes da modelagem.

## Avaliação rigorosa

- Métrica principal: **F1-macro** (classes desbalanceadas, todas importam).
- Sempre com **intervalo de confiança** (bootstrap), nunca um número solto.
- **Avaliação por fatia** para revelar falhas ocultas em subgrupos.

## Privacidade

Base pública e anônima; o pipeline (inclusive o LLM) roda 100% local. Exemplos
nos relatórios são limitados e sem `@usuários`/links.
