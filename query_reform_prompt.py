PROMPT_TEMPLATE = """You are a biomedical text simplifier.

Rewrite the health claim below into one simplified, direct, search-ready sentence.

Rules:
- Keep only the core factual medical claim.
- Remove source attributions such as "scientists say", "researchers found", or "studies show".
- Remove certainty or exaggeration such as "proven", "definitely", "always", or "miracle".
- Use clear clinical/scientific wording when helpful.
- Do not add new information.
- Output only the rewritten sentence.
- No JSON. No explanation. No markdown.

CLAIM: {claim}

SIMPLIFIED CLAIM:"""
