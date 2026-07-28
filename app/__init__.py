"""Aplicação FastAPI — serving local do fluxo "Python calcula, LLM explica".

Expõe endpoints para classificar sentimento de textos avulsos e para gerar um
resumo em linguagem simples a partir de um ``InsightsReport`` estruturado. Roda
100% local (privacidade).

Módulos
-------
schemas
    Modelos de request/response da API.
dependencies
    Injeção de settings e componentes (classificador, summarizer).
main
    Instância ``app`` e registro dos roteadores.
routers
    Roteadores por recurso (sentiment, summary, health).
"""
