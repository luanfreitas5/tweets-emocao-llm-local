# syntax=docker/dockerfile:1.7
# Servidor MLflow (tracking + registry) com backend SQLite em volume nomeado.
# A imagem oficial roda como root: aqui criamos um usuário sem privilégios.
# Mantenha a versão igual à do mlflow no uv.lock (cliente e servidor compatíveis; lock: 3.17.2).
# SQLite serve para uso individual/local; para servidor compartilhado, troque o backend por
# Postgres (instale psycopg2-binary nesta imagem e aponte --backend-store-uri para ele).
FROM ghcr.io/mlflow/mlflow:v3.17.2
# /mlflow (banco) e /mlartifacts são criados com o dono certo ANTES do volume:
# o volume nomeado herda a permissão do diretório da imagem.
RUN groupadd --system --gid 10001 mlflow \
    && useradd --system --uid 10001 --gid mlflow --no-create-home mlflow \
    && mkdir -p /mlflow /mlartifacts \
    && chown mlflow:mlflow /mlflow /mlartifacts
USER mlflow:mlflow
EXPOSE 5000
