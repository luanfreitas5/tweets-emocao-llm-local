## v0.2.0 (2026-07-28)

### Feat

- adiciona workflows e configurações para CI/CD
- adiciona configuração do ambiente e documentação
- adiciona arquivos de configuração para o projeto - Cria config.yaml com configurações gerais. - Cria deploy.yaml para configuração da API FastAPI. - Cria llm.yaml para configuração do LLM local. - Cria logging.yaml para configuração do logger. - Cria model_params.yaml com hiperparâmetros dos modelos. - Cria paths.yaml para caminhos de dados e saídas.
- adiciona nova funcionalidade de autenticação Implementa sistema de autenticação de usuários com suporte a login e registro. Melhora a segurança das credenciais armazenadas.
- adiciona testes para validação e processamento de dados
- adiciona estrutura inicial da API com roteadores e injeção de dependências
- adiciona pipelines para processamento de tweets
- adiciona arquivos de configuração (.env, Makefile, mkdocs.yml)
- adiciona Dockerfile e docker-compose.yml para orquestração da API
- adiciona arquivo pyproject.toml com configuração do projeto
- adiciona arquivos de configuração e documentação inicial

### Fix

- corrige exibição da distribuição de sentimentos
- corrige a leitura e processamento de tweets

### Refactor

- atualiza a orquestração do pipeline e remove comandos CLI
- melhora a normalização de sentimentos e limpeza de texto
- melhora a legibilidade da função encode
- remove resultados desnecessários do baseline
- simplifica chamada ao modelo de embeddings
- melhora a legibilidade e tratamento de exceções
- reorganiza e melhora a configuração do Makefile
- melhora a legibilidade dos testes e renomeia variáveis
