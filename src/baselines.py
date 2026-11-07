from src import prompts
from src.llm import generate
from src.retriever import search
from src.config import CFG

def llm_only(question: str) -> dict:
    pred = generate(prompts.LLM_ONLY_PROMPT.format(question=question),
                    system=prompts.SYSTEM)
    return {"pred": pred, "retrieved": []}

def single_shot_rag(question: str) -> dict:
    hits = search(question, k=CFG["top_k_single"])
    pred = generate(
        prompts.ANSWER_PROMPT.format(
            passages=prompts.format_passages(hits), question=question),
        system=prompts.SYSTEM,
    )
    return {"pred": pred, "retrieved": [h["id"] for h in hits]}