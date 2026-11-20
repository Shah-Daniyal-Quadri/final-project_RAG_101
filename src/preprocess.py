"""
Article cleaning + chunking.

Chunking rules
--------------
1. Paragraphs that fit within `chunk_max_words` are atomic — never split.
2. Longer paragraphs are split at sentence boundaries only — never mid-sentence.
3. Units (paragraph or sentence) are packed greedily up to `chunk_max_words`.
4. Overlap = the last full sentence of the previous chunk, carried into the
   next chunk ONLY if it fits within `chunk_overlap_words` AND leaves room
   for the first unit of the next chunk. If it would not fit, overlap is
   skipped for that boundary — we never cut a sentence just to hit a word count.
5. Nothing is ever truncated. If a single unit exceeds `chunk_max_words`,
   it gets its own chunk.
"""
import json
import re

from src.config import CFG, path


_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\"'])")


def clean(text: str) -> str:
    text = re.sub(r"\[\d+\]", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def _split_sentences(paragraph: str) -> list[str]:
    return [s.strip() for s in _SENT_SPLIT.split(paragraph) if s.strip()]


def _overlap_seed(prev_chunk: str, overlap_words: int, room: int) -> tuple[list[str], int]:
    """Return (seed_units, seed_word_count) for the next chunk.

    The seed is the last *full sentence* of `prev_chunk`, included only if it
    fits within both `overlap_words` and `room` (words left before max).
    """
    if overlap_words <= 0:
        return [], 0
    sents = _split_sentences(prev_chunk)
    if not sents:
        return [], 0
    last = sents[-1]
    lw = len(last.split())
    if lw <= overlap_words and lw <= room:
        return [last], lw
    return [], 0


def chunk_article(article: dict) -> list[dict]:
    max_w = int(CFG["chunk_max_words"])
    min_w = int(CFG["chunk_min_words"])
    overlap = int(CFG.get("chunk_overlap_words", 0))

    text = clean(article["text"])
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]

    # ---- Build atomic units -------------------------------------------------
    units: list[str] = []
    for p in paragraphs:
        if len(p.split()) <= max_w:
            units.append(p)                     # short paragraph: keep whole
        else:
            units.extend(_split_sentences(p))   # long paragraph: split by sentence

    # ---- Pack units into chunks --------------------------------------------
    chunks: list[str] = []
    buf: list[str] = []
    buf_w = 0

    for u in units:
        uw = len(u.split())

        # Oversized unit: place alone, never truncate.
        if uw > max_w:
            if buf:
                chunks.append(" ".join(buf))
                buf, buf_w = [], 0
            chunks.append(u)
            continue

        # Would overflow? Flush buffer, seed next chunk with sentence overlap.
        if buf and buf_w + uw > max_w:
            chunks.append(" ".join(buf))
            buf, buf_w = _overlap_seed(chunks[-1], overlap, max_w - uw)

        buf.append(u)
        buf_w += uw

    if buf:
        chunks.append(" ".join(buf))

    # Merge a tiny trailing chunk into its predecessor (never splits anything).
    if len(chunks) > 1 and len(chunks[-1].split()) < min_w:
        chunks[-2] = chunks[-2] + " " + chunks[-1]
        chunks.pop()

    return [
        {"id": f"{article['title']}#{i:03d}",
         "title": article["title"],
         "url": article["url"],
         "text": c}
        for i, c in enumerate(chunks)
    ]


def main():
    articles = json.load(open(path("articles"), encoding="utf-8"))
    all_chunks = []
    for a in articles:
        all_chunks.extend(chunk_article(a))

    path("chunks").parent.mkdir(parents=True, exist_ok=True)
    with open(path("chunks"), "w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    sizes = sorted(len(c["text"].split()) for c in all_chunks)
    n = len(sizes)
    print(f"Wrote {n} chunks -> {path('chunks')}")
    print(f"  size words: min={sizes[0]} "
          f"median={sizes[n // 2]} "
          f"p90={sizes[int(n * 0.9)]} max={sizes[-1]}")


if __name__ == "__main__":
    main()