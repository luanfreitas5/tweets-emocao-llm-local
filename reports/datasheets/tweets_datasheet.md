# Datasheet — Portuguese Tweets for Sentiment Analysis

## Motivação

Base para análise de sentimento em português, com ~800 mil tweets rotulados como
positivo, negativo ou neutro. Usada aqui para demonstrar a arquitetura
"Python calcula, LLM explica".

## Composição

- **Unidade:** um tweet por linha.
- **Colunas:** `id`, `tweet_text`, `tweet_date`, `sentiment`, `query_used`.
- **Rótulos:** derivados por **supervisão distante** — o emoticon/hashtag em
  `query_used` (ex.: `:)`, `:(`, `#fato`) define o `sentiment`.
- **Formatos:** arquivos "sem tema" usam vírgula e rótulo textual; os de
  treino/teste usam ponto e vírgula e rótulo numérico (0/1/2).

## Coleta

- Coletados da API pública do Twitter/X em 2018, filtrando por emoticons e
  hashtags. Fonte: Kaggle (autor: *augustop*).

## Pré-processamento

- Remoção de emoticons/hashtags rotuladores (**anti-leakage**), URLs, menções e
  números; normalização de espaços e caixa; deduplicação por texto limpo.
- Ver `src/preprocessing/`.

## Licenciamento e privacidade

- Base pública no Kaggle; verificar os termos da licença do dataset antes de
  redistribuir.
- Conteúdo potencialmente identificável (`@usuários`, links) é **removido** na
  limpeza e não aparece nos relatórios.
- Base de tratamento (LGPD): interesse legítimo para pesquisa, com minimização e
  anonimização.

## Usos recomendados / não recomendados

- **Recomendado:** benchmark de NLP em pt-BR, análise agregada de opinião.
- **Não recomendado:** inferência sobre indivíduos; qualquer uso que reidentifique
  autores.
