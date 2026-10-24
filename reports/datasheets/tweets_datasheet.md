# Datasheet — Portuguese Tweets for Sentiment Analysis (`NoThemeTweets.csv`)

## 1. Motivação
- Por que foi criado? Para qual tarefa? Base pública de análise de sentimento em português, com ~800 mil tweets
  rotulados como positivo, negativo ou neutro. Neste projeto serve para demonstrar a arquitetura
  "Python calcula, LLM explica" e para avaliar um classificador pré-treinado.
- Quem criou e quem financiou? Publicada no Kaggle pelo autor *augustop*; `TODO(luan): confirmar autoria
  acadêmica/financiamento na página do dataset`.

## 2. Composição
- O que cada instância representa? Um tweet público em português.
- Número de instâncias e período coberto: **785.814** tweets brutos em `NoThemeTweets.csv`; **713.518** após a limpeza
  (silver). Período: coleta em 2018 (`tweet_date` é texto, ex.: `Tue Aug 21 04:35:39 +0000 2018`;
  `TODO(luan): converter e registrar o intervalo exato`).
- Campos, tipos e contrato: `id`, `tweet_text`, `tweet_date`, `sentiment`, `query_used` (bronze);
  contrato de entrada em `src/schemas/raw.py` e de saída (silver) em `src/schemas/processed.py`.
- Rótulo: `sentiment` (Positivo/Negativo/Neutro) derivado por **supervisão distante** — o emoticon/hashtag registrado em
  `query_used` (ex.: `:)`, `:(`, `#fato`) define o rótulo; **não há anotação humana**. Arquivos "sem tema" usam vírgula e
  rótulo textual; os de treino/teste usam ponto e vírgula e rótulo numérico (0/1/2).
- Dados ausentes (taxa por coluna): 0 nulos nas 5 colunas do CSV bronze (`id`, `tweet_text`, `tweet_date`,
  `sentiment`, `query_used`), medido ao ler o arquivo com `polars`.
- Campos sensíveis / PII e tratamento: o texto pode conter `@menções` e links (identificáveis); são removidos na limpeza
  (`src/preprocessing/cleaning.py`) e não aparecem em relatórios.
- É amostra de uma população maior? Sim: tweets que continham os emoticons/hashtags de busca, não uma amostra aleatória
  do Twitter/X. Prevalência do **rótulo de referência**: bronze 522.707 Negativo (66,5%) e 263.107 Positivo (33,5%);
  silver 472.462 Negativo (66,2%) e 241.056 Positivo (33,8%). **Este arquivo não tem exemplos Neutro** (outros CSVs da
  pasta, como `TweetsNeutralHash.csv`, não são usados no pipeline). Previsão do modelo no silver: Negativo 57,4%,
  Positivo 35,3%, Neutro 7,3% (saída do modelo, não prevalência real).

## 3. Processo de coleta
- Fonte(s) e sistema(s) de origem: API pública do Twitter/X, buscas por emoticons e hashtags; redistribuição via
  Kaggle (<https://www.kaggle.com/datasets/augustop/portuguese-tweets-for-sentiment-analysis>).
- Período e frequência de coleta: 2018; coleta única (snapshot).
- Consentimento / aviso aos titulares: não há consentimento individual; os dados são públicos e anonimizados na
  publicação original (`TODO(luan): confirmar na página do dataset`).

## 4. Pré-processamento
- Limpeza, filtragem e linhas removidas por regra: remoção de emoticons/hashtags rotuladores (**anti-*leakage***),
  URLs, menções e números; normalização de espaços e caixa; deduplicação por texto limpo
  (`src/preprocessing/`). Total removido: 785.814 → 713.518 (**72.296 linhas, 9,2%**);
  `TODO(luan): registrar a contagem por regra (o log da etapa preprocess da execução de 2026-10-24 não foi preservado)`.
- Anonimização / pseudonimização: remoção de `@menções` e links; não há segredo/salt (nenhum identificador é mantido).
- Hash SHA-256 dos dados brutos (`data/bronze/NoThemeTweets.csv`):
  `85313df6a14fbb6829d15362f5cd20ac2f027ec454e9f1009532151b0da63a5b`. Hash do DataFrame gold da execução de 2026-10-24
  (`hash_dataframe`, ordem-independente): `56a98c1c6a620275a792047fed4648b585a2dc325de00f25a61ed935cdabc7f0`.
- Camadas: **bronze** `data/bronze/` (CSVs como recebidos, imutáveis) → **silver** `data/silver/tweets_processed.parquet`
  (limpo, tipado, validado por `pandera`) → **gold** `data/gold/` (`sentiment_predictions.parquet`,
  `topic_assignments.parquet`). Nenhuma camada é versionada no Git.

## 5. Usos
- Usos pretendidos: benchmark de NLP em português, análise agregada de opinião, demonstração de pipeline.
- Usos desaconselhados: inferência sobre indivíduos; qualquer uso que reidentifique autores; tratar o rótulo por emoticon
  como verdade absoluta; generalizar para o público brasileiro em geral (amostra enviesada por quem usa emoticons).
- Riscos de viés conhecidos: ruído de rótulo (ironia, sarcasmo), viés de seleção (só tweets com emoticon/hashtag),
  período único (2018), linguagem regional e gírias; ausência total da classe Neutro neste arquivo, o que impede avaliar
  classificadores de 3 classes (ver Model Card).

## 6. Distribuição
- Licença e restrições: base pública no Kaggle; `TODO(luan): registrar a licença exibida na página do dataset` antes
  de redistribuir qualquer derivado. O repositório **não** redistribui os dados.
- Onde está armazenado (DVC remote, bucket): localmente em `data/bronze/` (ignorado pelo Git);
  `TODO(luan): configurar DVC remote se for compartilhar`.

## 7. Manutenção
- Responsável: Luan Freitas.
- Frequência de atualização: nenhuma — snapshot estático de 2018.
- Como atender pedidos de exclusão de titulares: o dataset é de terceiros; não há chave de titular. Pedidos devem ser
  encaminhados à fonte (Kaggle/autor); localmente, remover o tweet do bronze por `id` e reexecutar o pipeline.

## 8. LGPD
- Base legal (art. 7 / art. 11): interesse legítimo para pesquisa (art. 7º, IX), com minimização e remoção de
  identificadores diretos. Dados de opinião podem revelar categorias sensíveis (art. 11) — por isso só são usados de forma
  agregada. `TODO(luan): validar a base legal com o encarregado/orientador antes de publicar artigo`.
- Prazo de retenção: apenas durante o projeto de pesquisa; sem retenção em serviços externos (processamento 100% local,
  inclusive o LLM via Ollama).
- Relatório de impacto (RIPD) necessário? Não para o uso agregado de portfólio; `TODO(luan): reavaliar se houver
  publicação de exemplos de tweets` (os `example_texts` em `reports/insights.json` mostram trechos de tweets reais).
