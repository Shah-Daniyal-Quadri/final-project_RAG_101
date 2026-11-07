from src import prompts
from src.llm import generate
from src.retriever import search
from src.config import CFG

def multihop_rag(question: str) -> dict:
    hop1 = search(question, k=CFG["top_k_first"])
    hop1_ids = {h["id"] for h in hop1}

    followup = generate(
        prompts.FOLLOWUP_PROMPT.format(
            passages=prompts.format_passages(hop1), question=question),
        system=prompts.SYSTEM,
    ).strip().strip('"').split("\n")[0]

    hop2 = search(followup, k=CFG["top_k_second"], exclude_ids=hop1_ids)

    all_passages = hop1 + hop2
    pred = generate(
        prompts.ANSWER_PROMPT.format(
            passages=prompts.format_passages(all_passages), question=question),
        system=prompts.SYSTEM,
    )
    return {
        "pred": pred,
        "followup_query": followup,
        "retrieved": [p["id"] for p in all_passages],
        "hop1": [h["id"] for h in hop1],
        "hop2": [h["id"] for h in hop2],
    }