from src import prompts
from src.config import CFG
from src.llm import generate
from src.retriever import search


_PREFIXES = (
    "query:", "search query:", "search:", "follow-up query:",
    "followup query:", "follow-up:", "followup:", "q:",
)


def _clean_followup(raw: str) -> str:
    s = raw.strip()
    # Peel common prefixes the model may add
    for prefix in _PREFIXES:
        if s.lower().startswith(prefix):
            s = s[len(prefix):].strip()
    # Drop wrapping quotes and take only the first non-empty line
    s = s.strip().strip('"').strip("'")
    lines = [ln.strip() for ln in s.split("\n") if ln.strip()]
    return lines[0] if lines else ""


def multihop_rag(question: str) -> dict:
    hop1 = search(question, k=CFG["top_k_first"])
    hop1_ids = {h["id"] for h in hop1}

    raw_followup = generate(
        prompts.FOLLOWUP_PROMPT.format(
            passages=prompts.format_passages(hop1),
            question=question,
        ),
        system=prompts.SYSTEM,
    )
    followup = _clean_followup(raw_followup)

    if len(followup.split()) < 3:
        followup = question

    hop2 = search(followup, k=CFG["top_k_second"], exclude_ids=hop1_ids)

    all_passages = hop1 + hop2
    pred = generate(
        prompts.ANSWER_PROMPT.format(
            passages=prompts.format_passages(all_passages),
            question=question,
        ),
        system=prompts.SYSTEM,
    )

    return {
        "pred": pred,
        "followup_query": followup,
        "retrieved": [p["id"] for p in all_passages],
        "hop1": [h["id"] for h in hop1],
        "hop2": [h["id"] for h in hop2],
    }