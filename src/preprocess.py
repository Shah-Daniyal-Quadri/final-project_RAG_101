import json, re, hashlib
from src.config import CFG, path

def clean(text: str) -> str:
    text = re.sub(r"\[\d+\]", "", text)          # remove [1] citations
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()

def chunk_article(article: dict) -> list[dict]:
    paragraphs = [p.strip() for p in clean(article["text"]).split("\n") if p.strip()]
    chunks, buf = [], []
    for p in paragraphs:
        buf.append(p)
        if sum(len(x.split()) for x in buf) >= CFG["chunk_min_words"]:
            text = " ".join(buf)
            if len(text.split()) > CFG["chunk_max_words"]:
                words = text.split()
                text = " ".join(words[: CFG["chunk_max_words"]])
            chunks.append(text)
            buf = []
    if buf:
        chunks.append(" ".join(buf))

    out = []
    for i, text in enumerate(chunks):
        cid = f"{article['title']}#{i:03d}"
        out.append({"id": cid, "title": article["title"],
                    "url": article["url"], "text": text})
    return out

def main():
    articles = json.load(open(path("articles"), encoding="utf-8"))
    all_chunks = []
    for a in articles:
        all_chunks.extend(chunk_article(a))
    path("chunks").parent.mkdir(parents=True, exist_ok=True)
    with open(path("chunks"), "w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"Wrote {len(all_chunks)} chunks → {path('chunks')}")

if __name__ == "__main__":
    main()