
from functools import lru_cache

from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

from src.config import CFG


def _detect_device() -> dict:
    preference = CFG.get("device", "auto").lower()

    if preference == "cpu":
        return {"available": False, "name": None, "num_gpu": 0}

    try:
        import torch
        if torch.cuda.is_available():
            return {
                "available": True,
                "name": torch.cuda.get_device_name(0),
                "num_gpu": None,
            }
    except Exception:
        pass

    return {"available": False, "name": None, "num_gpu": 0}


@lru_cache
def _llm() -> ChatOllama:
    dev = _detect_device()
    kwargs = {
        "model": CFG["model"],
        "temperature": CFG["temperature"],
        "seed": CFG["seed"],
        "base_url": CFG["ollama_host"],
    }
    if dev["num_gpu"] is not None:
        kwargs["num_gpu"] = dev["num_gpu"]
    num_ctx = CFG.get("num_ctx")
    if num_ctx:
        kwargs["num_ctx"] = num_ctx

    print(f"[llm] backend: {'GPU (' + dev['name'] + ')' if dev['available'] else 'CPU'}"
          f" | model={CFG['model']} | num_gpu={dev['num_gpu']}")
    return ChatOllama(**kwargs)


def generate(prompt: str, system: str | None = None) -> str:
    msgs = []
    if system:
        msgs.append(SystemMessage(content=system))
    msgs.append(HumanMessage(content=prompt))
    resp = _llm().invoke(msgs)
    content = resp.content
    if isinstance(content, list):
        content = " ".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
    return content.strip()


def llm_backend_info() -> dict:
    """For Checkpoint 3 report: which backend actually ran?"""
    dev = _detect_device()
    return {
        "model": CFG["model"],
        "gpu_available": dev["available"],
        "gpu_name": dev["name"],
        "backend": "GPU" if dev["available"] else "CPU",
    }