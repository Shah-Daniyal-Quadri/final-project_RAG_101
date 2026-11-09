from functools import lru_cache
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from src.config import CFG

@lru_cache
def _llm() -> ChatOllama:
    return ChatOllama(
        model=CFG["model"],
        temperature=CFG["temperature"],
        seed=CFG["seed"],
        base_url=CFG["ollama_host"],
        num_gpu=0,
    )

def generate(prompt: str, system: str | None = None) -> str:
    msgs = []
    if system:
        msgs.append(SystemMessage(content=system))
    msgs.append(HumanMessage(content=prompt))
    resp = _llm().invoke(msgs)
    # ChatOllama returns an AIMessage; .content may be a str or list of parts
    content = resp.content
    if isinstance(content, list):
        content = " ".join(part.get("text", "") if isinstance(part, dict) else str(part)
                           for part in content)
    return content.strip()