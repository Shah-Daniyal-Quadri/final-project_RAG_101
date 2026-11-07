import json, numpy as np
from langchain_huggingface import HuggingFaceEmbeddings
from src.config import CFG, path

def _embedder() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=CFG["embedding_model"],
        encode_kwargs={"normalize_embeddings": True},
    )

def main():
    chunks = [json.loads(l) for l in open(path("chunks"), encoding="utf-8")]
    model = _embedder()
    texts = [c["text"] for c in chunks]
    emb = np.array(model.embed_documents(texts), dtype="float32")
    ids = np.array([c["id"] for c in chunks])
    path("index").parent.mkdir(parents=True, exist_ok=True)
    np.savez(path("index"), embeddings=emb, ids=ids)
    print(f"Indexed {len(chunks)} chunks → {path('index')}")

if __name__ == "__main__":
    main()