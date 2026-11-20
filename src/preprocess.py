"""
Article cleaning + chunking.

Chunking rules
--------------
1. Section headings (== ... ==, === ... ===, etc.) are HARD boundaries.
   Chunks never cross a section boundary — same behavior as article
   boundaries.
2. Boilerplate sections (References, See also, External links, ...) are
   skipped entirely.
3. Paragraphs that fit within chunk_max_words are atomic — kept whole.
   Longer paragraphs are split at sentence boundaries only.
4. Within a section, atomic units are packed until the next unit would
   exceed chunk_max_words, then flush. Last chunk of a section may be
   smaller than the target.
5. Overlap: every new chunk (except the first in its section) starts
   with at least ONE full sentence from the tail of the previous chunk
   — always. Additional trailing sentences are prepended only while the
   running total stays <= chunk_overlap_words. Sentences are never cut.
   Oversized single-sentence chunks do not seed the next chunk.
6. Nothing is ever truncated.
7. Section / subsection titles are stored as metadata; they do NOT
   appear in the chunk body text. format_passages in prompts.py
   surfaces them to the LLM.
"""
import json
import re

from src.config import CFG, path

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\"'])")
_HEADING_RE = re.compile(r"^\s*(=+)\s*(.+?)\s*\1\s*$")

_SKIP_SECTIONS = {
    "see also", "references", "external links", "further reading",
    "notes", "bibliography", "citations", "sources", "footnotes",
    "notes and references", "explanatory notes", "general references",
    "works cited", "external links modified",
}


def clean(text: str) -> str:
    text = re.sub(r"\[\d+\]", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def _split_sentences(paragraph: str) -> list[str]:
    return [s.strip() for s in _SENT_SPLIT.split(paragraph) if s.strip()]


def _parse_sections(text: str) -> list[dict]:
    blocks: list[dict] = []
    cur_section = ""
    sub_stack: list[str] = []
    cur_lines: list[str] = []

    def flush():
        if not cur_lines:
            return
        body = "\n".join(cur_lines).strip()
        cur_lines.clear()
        if body:
            blocks.append({
                "section": cur_section,
                "subsection": " > ".join(sub_stack),
                "body": body,
            })

    for line in text.split("\n"):
        m = _HEADING_RE.match(line)
        if m:
            flush()
            level = len(m.group(1))
            title = m.group(2).strip()
            if level <= 2:
                cur_section = title
                sub_stack = []
            else:
                depth = level - 3
                sub_stack = sub_stack[:depth] + [title]
        else:
            cur_lines.append(line)
    flush()
    return blocks


def _overlap_seed(prev_chunk: str, overlap_words: int) -> tuple[list[str], int]:
    sents = _split_sentences(prev_chunk)
    if not sents:
        return [], 0
    seed = [sents[-1]]
    total = len(sents[-1].split())
    for s in reversed(sents[:-1]):
        sw = len(s.split())
        if total + sw > overlap_words:
            break
        seed.insert(0, s)
        total += sw
    return seed, total


def _pack_units(units: list[str], max_w: int, overlap_words: int) -> list[str]:
    chunks: list[str] = []
    buf: list[str] = []
    buf_w = 0

    def reseed():
        nonlocal buf, buf_w
        seed, seed_w = _overlap_seed(chunks[-1], overlap_words)
        buf, buf_w = list(seed), seed_w

    for u in units:
        uw = len(u.split())

        if uw > max_w:
            if buf:
                chunks.append(" ".join(buf))
            chunks.append(u)
            buf, buf_w = [], 0
            continue

        if buf and buf_w + uw > max_w:
            chunks.append(" ".join(buf))
            reseed()

        buf.append(u)
        buf_w += uw

    if buf:
        chunks.append(" ".join(buf))
    return chunks


def chunk_article(article: dict) -> list[dict]:
    max_w = int(CFG["chunk_max_words"])
    overlap = int(CFG.get("chunk_overlap_words", 0))

    text = clean(article["text"])
    blocks = _parse_sections(text)

    out: list[dict] = []
    counter = 0
    for block in blocks:
        if block["section"].lower() in _SKIP_SECTIONS:
            continue

        paragraphs = [p.strip() for p in block["body"].split("\n") if p.strip()]
        units: list[str] = []
        for p in paragraphs:
            if len(p.split()) <= max_w:
                units.append(p)
            else:
                units.extend(_split_sentences(p))

        for c in _pack_units(units, max_w, overlap):
            out.append({
                "id": f"{article['title']}#{counter:04d}",
                "title": article["title"],
                "section": block["section"],
                "subsection": block["subsection"],
                "url": article["url"],
                "text": c,
            })
            counter += 1

    return out


def main():
    articles = json.load(open(path("articles"), encoding="utf-8"))
    all_chunks: list[dict] = []
    for a in articles:
        all_chunks.extend(chunk_article(a))

    path("chunks").parent.mkdir(parents=True, exist_ok=True)
    with open(path("chunks"), "w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    sizes = sorted(len(c["text"].split()) for c in all_chunks)
    n = len(sizes)
    n_sections = sum(1 for c in all_chunks if c["section"])
    print(f"Wrote {n} chunks from {len(articles)} articles -> {path('chunks')}")
    print(f"  size words: min={sizes[0]} "
          f"median={sizes[n // 2]} p90={sizes[int(n * 0.9)]} max={sizes[-1]}")
    print(f"  sections labelled: {n_sections}/{n} chunks "
          f"({100 * n_sections / n:.1f}%)")


if __name__ == "__main__":
    main()
