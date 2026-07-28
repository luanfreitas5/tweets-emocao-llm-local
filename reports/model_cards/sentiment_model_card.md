# Model Card — Classificador de Sentimento (pt-BR)

> Template. Preencha as seções com números após treinar/avaliar (ver
> `src/evaluation`). Não reporte métrica pontual sem intervalo de confiança.

## Detalhes do modelo

- **Tarefa:** classificação de sentimento em 3 classes (Positivo/Negativo/Neutro).
- **Base do modelo:** Transformer pt-BR (BERTimbau/afins) definido em
  `configs/model_params.yaml`.
- **Entrada:** texto de tweet **limpo** (sem emoticons/hashtags rotuladores).
- **Saída:** rótulo canônico + score de confiança.

## Uso pretendido

- Análise agregada de opinião em tweets em português (portfólio/pesquisa).
- **Fora de escopo:** decisões individuais sobre pessoas; moderação automática
  sem revisão humana; domínios fora de redes sociais em pt-BR.

## Dados de treino

- [Portuguese Tweets for Sentiment Analysis](https://www.kaggle.com/datasets/augustop/portuguese-tweets-for-sentiment-analysis).
- Rótulos por **supervisão distante** (emoticons/hashtags) — ver *limitações*.

## Avaliação

| Métrica | Valor | IC 95% |
|---|---|---|
| F1-macro | _a preencher_ | _a preencher_ |

### Por classe

_A preencher (precisão/recall/F1 por classe)._

### Por fatia

_A preencher (ex.: por faixa de comprimento do tweet)._

## Limitações e riscos

- **Ruído de rótulo:** a supervisão distante por emoticons é imperfeita; textos
  irônicos/ambíguos podem estar mal rotulados.
- **Leakage:** mitigado removendo emoticons/hashtags antes da modelagem
  (`remove_label_leakage`).
- **Viés:** linguagem de redes sociais pode conter gírias/viés regional; avaliar
  por fatia antes de qualquer uso.

## Considerações éticas

- Base pública e anônima; nenhum dado pessoal é exposto nos relatórios.
- Processamento 100% local (privacidade).
