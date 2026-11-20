import json

import numpy as np
import torch
from langchain_huggingface import HuggingFaceEmbeddings

from src.config import CFG, path


def _device() -> str:
    pref = CFG.get("device", "auto").lower()
    if pref in ("cpu", "cuda"):
        return pref
    return "cuda" if torch.cuda.is_available() else "cpu"


def _embedder() -> HuggingFaceEmbeddings:
    dev = _device()
    print(f"[embed] device={dev} | model={CFG['embedding_model']}")
    return HuggingFaceEmbeddings(
        model_name=CFG["embedding_model"],
        model_kwargs={"device": dev},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 64},
    )


def main():
    chunks = [json.loads(l) for l in open(path("chunks"), encoding="utf-8")]
    model = _embedder()
    texts = [c["text"] for c in chunks]
    emb = np.array(model.embed_documents(texts), dtype="float32")
    ids = np.array([c["id"] for c in chunks])

    path("index").parent.mkdir(parents=True, exist_ok=True)
    np.savez(path("index"), embeddings=emb, ids=ids)
    print(f"Indexed {len(chunks)} chunks -> {path('index')} "
          f"(dim={emb.shape[1]}, size={emb.nbytes / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()