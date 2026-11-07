import json, numpy as np
from functools import lru_cache
from langchain_huggingface import HuggingFaceEmbeddings
from src.config import CFG, path

@lru_cache
def _load():
    idx = np.load(path("index"), allow_pickle=True)
    chunks = {c["id"]: c for c in
              (json.loads(l) for l in open(path("chunks"), encoding="utf-8"))}
    model = HuggingFaceEmbeddings(
        model_name=CFG["embedding_model"],
        encode_kwargs={"normalize_embeddings": True},
    )
    return idx["embeddings"], idx["ids"], chunks, model

def search(query: str, k: int, exclude_ids: set[str] | None = None) -> list[dict]:
    emb, ids, chunks, model = _load()
    q = np.array(model.embed_query(query), dtype="float32")
    scores = emb @ q
    order = np.argsort(-scores)
    out = []
    for i in order:
        cid = str(ids[i])
        if exclude_ids and cid in exclude_ids:
            continue
        out.append({"id": cid, "score": float(scores[i]), **chunks[cid]})
        if len(out) == k:
            break
    return out