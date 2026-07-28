"""Templates de prompt para o resumo dos insights.

O prompt de sistema impõe a regra central anti-alucinação: **usar apenas os
números do JSON, nunca inventar dados**. O prompt de usuário injeta o JSON
estruturado (``InsightsReport``) já computado em Python.
"""

from __future__ import annotations

from src.schemas.insights import InsightsReport

SYSTEM_PROMPT = (
    "Você é um analista de dados que escreve resumos claros em português do Brasil. "
    "Regras invioláveis:\n"
    "1. Use SOMENTE os números e fatos presentes no JSON fornecido.\n"
    "2. NUNCA invente estatísticas, percentuais, tópicos ou exemplos.\n"
    "3. Se um dado não estiver no JSON, não o mencione.\n"
    "4. Escreva em linguagem simples, acessível a um público não técnico.\n"
    "5. Não reproduza @usuários, links ou conteúdo sensível dos exemplos.\n"
    "Seu papel é apenas explicar, em texto corrido, o que os números já dizem."
)

USER_TEMPLATE = (
    "A partir do relatório estruturado abaixo (em JSON), escreva um resumo em "
    "Markdown com: (a) um panorama geral do sentimento, (b) os principais tópicos "
    "e o clima predominante em cada um, e (c) uma conclusão curta. "
    "Não acrescente nenhum número que não esteja no JSON.\n\n"
    "```json\n{report_json}\n```"
)


def build_messages(report: InsightsReport) -> list[dict[str, str]]:
    """Monta a lista de mensagens (sistema + usuário) para o chat do Ollama.

    Parameters
    ----------
    report : InsightsReport
        Relatório estruturado computado em Python.

    Returns
    -------
    list[dict[str, str]]
        Mensagens no formato ``[{"role": ..., "content": ...}, ...]``.

    Examples
    --------
    >>> messages = build_messages(report)  # doctest: +SKIP
    >>> messages[0]["role"]  # doctest: +SKIP
    'system'
    """
    report_json = report.model_dump_json(indent=2)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(report_json=report_json)},
    ]
