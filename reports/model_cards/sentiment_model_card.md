# Model Card — Classificador de Sentimento de Tweets (pt-BR) v0.2.0

**Status:** DRAFT — **não aprovado para classificação individual**; utilizável apenas para análise agregada exploratória.
**Bloqueios pendentes:**
- Baseline treinado (regressão logística TF-IDF) e teste de significância (McNemar) — bloqueia qualquer afirmação de que
  o Transformer é "melhor"; bloqueia `APPROVED`.
- Métrica no limiar do projeto: F1-macro 0,362 < `min_f1_macro` 0,70 (`configs/model_params.yaml`) — bloqueia uso em
  decisões; é preciso ajustar o modelo, trocar de modelo ou rever a métrica/referência.
- Calibração e limiar de decisão (seção 5) — bloqueia usar o *score* como probabilidade.
- Rastreabilidade do run (seção 1): execução sem MLflow — bloqueia reprodução exata dos números publicados.

## 1. Detalhes do modelo

- **Responsável / contato:** Luan Freitas (<https://github.com/luanfreitas5/tweets-emocao-llm-local>)
- **Data:** 2026-10-24 (execução que gerou `reports/metrics/classification.json`, `reports/insights.json` e as figuras)
- **Algoritmo:** Transformer pré-treinado `lxyuan/distilbert-base-multilingual-cased-sentiments-student`
  (DistilBERT multilíngue, 3 classes), via `transformers.pipeline`. **Sem ajuste fino neste projeto**; os rótulos do modelo
  são mapeados para `Positivo`/`Negativo`/`Neutro` (`configs/model_params.yaml`).
- **Rastreabilidade:** Git SHA `TODO(luan): a execução de 2026-10-24 não foi rastreada; rode make mlflow` ·
  MLflow run `TODO(luan): idem` · SHA-256 do CSV bronze
  `85313df6a14fbb6829d15362f5cd20ac2f027ec454e9f1009532151b0da63a5b` · hash do DataFrame gold (`hash_dataframe`)
  `56a98c1c6a620275a792047fed4648b585a2dc325de00f25a61ed935cdabc7f0`
- **Ambiente da execução:** Linux/WSL, Python 3.13 (caminho do venv nos logs); semente 42.
- **Licença:** código MIT; licença dos pesos: `TODO(luan): conferir o model card do modelo no Hugging Face Hub`.

## 2. Uso pretendido

- **Uso primário:** estimar a distribuição **agregada** de sentimento e comparar o clima entre grupos de tweets em
  português (portfólio/pesquisa exploratória), sempre com a ressalva de que o modelo erra muito (seção 8).
- **Usuários pretendidos:** analistas e pesquisadores que consomem o `InsightsReport` e o resumo gerado pelo LLM.
- **Humano no circuito:** sim — a saída é lida como tendência, sem decisão automática.

## 3. Usos fora do escopo

- Decisões ou classificação sobre **indivíduos** (perfilar, avaliar ou punir autores de tweets).
- Moderação automática de conteúdo sem revisão humana.
- Domínios que não sejam tweets em português do Brasil (atendimento, saúde, textos formais).
- Conclusões sobre a classe **Neutro**: não há exemplos Neutro na referência; todas as 52.167 previsões Neutro contam como
  erro na avaliação.
- Interpretar o *score* como probabilidade calibrada (calibração não avaliada).
- Estimar a prevalência real de sentimento: o modelo prevê 57,4% de Negativo contra 66,2% na referência.

## 4. Fatores

- **Grupos relevantes:** comprimento do tweet (curto 1–5, médio 6–15, longo 16+ palavras; `src/pipelines/evaluation.py`);
  ironia/gírias. Atributos demográficos **não existem** na base.
- **Ambientes:** tweets públicos coletados em 2018; a linguagem de redes sociais muda com o tempo.

## 5. Métricas

- **Métrica principal:** F1-macro — as classes importam igualmente e a referência é desbalanceada (66% / 34%). Confundir
  Positivo com Negativo é o erro mais grave (inverte o clima agregado); `TODO(luan): quantificar custos relativos se
  houver uso a jusante`.
- **Atenção à definição:** o F1-macro de 3 classes inclui Neutro, com suporte 0 na referência (F1 = 0 por construção);
  isso reduz o valor em 1/3. Ver seção 8 para o valor com as duas classes que existem.
- **Limiar de decisão:** argmax das probabilidades do modelo; nenhum limiar ajustado em validação.
- **Calibração:** `TODO(luan): Brier score e curva de confiabilidade (reports/figures/calibracao.png)`.

## 6. Dados de avaliação

Os 713.518 tweets da base `NoThemeTweets.csv` após a limpeza (camada silver), com o emoticon como rótulo de referência
(**supervisão distante**, removido do texto antes da previsão): Negativo 472.462 (66,2%) e Positivo 241.056 (33,8%); sem
Neutro. Como o modelo **não foi treinado** nesses dados, não há *split* treino/teste; a base inteira é avaliação. A
referência é ruidosa (seção 10): a métrica mede **concordância com o emoticon**, não com anotação humana.

## 7. Dados de treino

Não há treino neste projeto: o modelo é usado como recebido do Hugging Face. Os dados do modelo de origem estão descritos
no model card dele (`TODO(luan): linkar`). Datasheet da base de avaliação:
[`reports/datasheets/tweets_datasheet.md`](../datasheets/tweets_datasheet.md).

## 8. Análise quantitativa

Fonte: [`reports/metrics/classification.json`](../metrics/classification.json) (bootstrap, 1.000 reamostragens, semente 42).

| Recorte | n | F1-macro (3 classes) ± IC 95% | Referência trivial ("sempre Negativo") |
|---|---|---|---|
| Geral | 713.518 | 0,362 (0,361–0,363) | 0,266 |
| Curto (1–5 palavras) | `TODO(luan)` | 0,334 (sem IC) | — |
| Médio (6–15 palavras) | `TODO(luan)` | 0,366 (sem IC) | — |
| Longo (16+ palavras) | `TODO(luan)` | 0,374 (sem IC) | — |
| Baseline: regressão logística TF-IDF | — | `TODO(luan)` | — |

A referência trivial é calculada da prevalência (F1 Negativo = 0,797; demais classes 0), não de um modelo treinado.

| Classe | Precisão | Recall | F1 | Suporte |
|---|---|---|---|---|
| Negativo | 0,704 | 0,610 | 0,653 | 472.462 |
| Positivo | 0,423 | 0,442 | 0,432 | 241.056 |
| Neutro | 0,000 | 0,000 | 0,000 | 0 |

- **Acurácia:** 0,553 — **menor** que a de sempre prever "Negativo" (0,662).
- **F1-macro só com as duas classes com suporte** (média dos F1 de Negativo e Positivo, derivada da tabela acima): 0,543;
  também abaixo de 0,70.
- **Erros dominantes** ([matriz de confusão](../figures/matriz_confusao.png), normalizada por linha): 50% dos Positivos
  foram previstos como Negativo; 31% dos Negativos, como Positivo; 8% dos Negativos e 5% dos Positivos, como Neutro.
- **IC estreito (±0,001):** reflete n grande, não a qualidade da referência; não demonstra que o modelo é bom.
- Distribuição prevista: Negativo 57,4%, Positivo 35,3%, Neutro 7,3% (409.390 / 251.961 / 52.167).

## 9. Considerações éticas e fairness

- **Atributos sensíveis auditados:** nenhum — a base não traz sexo, idade, região ou raça. Isto **não** significa ausência
  de viés: dialetos e gírias regionais podem ser classificados de forma desigual (`TODO(luan): auditar por proxy
  linguístico se houver uso além do agregado`).
- **Disparidades encontradas:** não medidas.
- **Mitigação aplicada:** nenhuma.
- **Base legal LGPD:** interesse legítimo para pesquisa, com minimização; `@menções` e links são removidos na limpeza e
  não aparecem nos relatórios (ver Datasheet, seção 8). Atenção: `reports/insights.json` traz `example_texts` com trechos
  de tweets reais e não é versionado.

## 10. Limitações, riscos e monitoramento

- **Limitações conhecidas:**
  - Desempenho fraco frente à referência (F1-macro 0,362; acurácia abaixo da classe majoritária).
  - Referência por emoticon é ruidosa (ironia, sarcasmo) e remover o emoticon altera o texto; parte do "erro" pode ser
    ruído da referência, e isso **não foi quantificado** (sem anotação humana).
  - Sem exemplos Neutro na referência; a classe Neutro prevista (7,3%) é não validada.
  - Sem baseline treinado e sem teste de significância.
  - Tópicos (BERTopic): 98,5% dos tweets em *outliers* (270.279) ou no tópico 0 (432.803); só 1,5% distribuído em 30
    tópicos pequenos, com termos principais vazios. Os insights por tópico são pouco informativos.
  - O resumo do LLM (`llama3.1:8b`) não sinaliza nenhuma dessas limitações.
- **Mudanças de distribuição previstas:** base de 2018; vocabulário, memes e eventos posteriores não estão representados.
- **Monitoramento:** não há serviço em produção contínua. Se a API for exposta, registrar scores e proporção por classe
  para checar *drift* (`evidently`), sem registrar o texto dos tweets.
- **Critério de retreino / descontinuação:** o limiar `min_f1_macro = 0,70` do projeto **já está violado**; antes de
  qualquer uso além do exploratório, ajustar/trocar o modelo, treinar o baseline e reavaliar com a métrica restrita às
  classes com suporte.
