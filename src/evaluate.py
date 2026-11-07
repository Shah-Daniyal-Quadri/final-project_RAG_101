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
    if not gold:
        return float("nan")
    return len(set(retrieved[:k]) & set(gold)) / len(set(gold))