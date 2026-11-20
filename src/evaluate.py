import re, string

def _norm(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(ch for ch in s if ch not in string.punctuation)
    return " ".join(s.split())

def exact_match(pred: str, gold: str) -> int:
    return int(_norm(pred) == _norm(gold))

def f1(pred: str, gold: str) -> float:
    p, g = _norm(pred).split(), _norm(gold).split()
    if not p or not g:
        return 0.0
    counts = {}
    for w in p:
        counts[w] = counts.get(w, 0) + 1
    overlap = 0
    for w in g:
        if counts.get(w, 0) > 0:
            overlap += 1
            counts[w] -= 1
    if overlap == 0:
        return 0.0
    prec, rec = overlap / len(p), overlap / len(g)
    return 2 * prec * rec / (prec + rec)

def recall_at_k(retrieved: list[str], gold: list[str], k: int) -> float:
    """Chunk-level recall: fraction of gold chunk IDs found in the top-k."""
    if not gold:
        return float("nan")
    return len(set(retrieved[:k]) & set(gold)) / len(set(gold))

def article_of(chunk_id: str) -> str:
    """Chunk IDs look like 'Babur#002' -> article title 'Babur'."""
    return chunk_id.rsplit("#", 1)[0]

def article_recall_at_k(retrieved: list[str], gold_articles: list[str], k: int) -> float:
    """Article-level recall: fraction of gold articles that have at least one
    chunk in the top-k retrieved. NaN if unlabeled or nothing was retrieved
    (llm_only), so those rows are skipped when averaging."""
    if not gold_articles or not retrieved:
        return float("nan")
    hit = {article_of(c) for c in retrieved[:k]}
    gold = set(gold_articles)
    return len(gold & hit) / len(gold)