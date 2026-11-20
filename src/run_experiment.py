import argparse, json, time, csv
import pandas as pd
from pathlib import Path

from src.config import CFG, path
from src.baselines import llm_only, single_shot_rag
from src.multihop import multihop_rag
from src.evaluate import exact_match, f1, recall_at_k, article_recall_at_k

SYSTEMS = {"llm_only": llm_only, "rag": single_shot_rag, "multihop": multihop_rag}

def load_supporting_articles() -> dict[str, list[str]]:
    """Gold articles per question, e.g. {"q001": ["First Battle of Panipat", "Babur"]}."""
    p = path("supporting_articles")
    return json.load(open(p, encoding="utf-8")) if p.exists() else {}

def recall(retrieved: list[str], q: dict, gold_articles: list[str], k: int) -> float:
    """Chunk-level recall if the question has supporting_chunks, else article-level."""
    if q.get("supporting_chunks"):
        return recall_at_k(retrieved, q["supporting_chunks"], k)
    return article_recall_at_k(retrieved, gold_articles, k)

def load_questions(split: str) -> list[dict]:
    qs = [json.loads(l) for l in open(path("questions"), encoding="utf-8")]
    if split == "all":
        return qs
    split_ids = set(json.load(open(path("splits") / f"{split}.json")))
    return [q for q in qs if q["qid"] in split_ids]

def _mean_valid(xs: list[float]) -> float | None:
    vals = [x for x in xs if x == x]  # filter NaN
    return sum(vals) / len(vals) if vals else None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", choices=SYSTEMS, required=True)
    ap.add_argument("--split", choices=["all", "dev", "test"], default="all")
    args = ap.parse_args()

    if args.split != "all" and not (path("splits") / f"{args.split}.json").exists():
        print(f"[note] no split file for '{args.split}' -> using all questions")
        args.split = "all"

    fn = SYSTEMS[args.system]
    questions = load_questions(args.split)
    support = load_supporting_articles()
    rows = []

    for q in questions:
        t0 = time.time()
        out = fn(q["question"])
        latency = time.time() - t0
        gold = q["answer"]
        gold_articles = support.get(q["qid"], [])

        row = {
            "qid": q["qid"],
            "question": q["question"],
            "gold": gold,
            "pred": out["pred"],
            "em": exact_match(out["pred"], gold),
            "f1": f1(out["pred"], gold),
            "recall_at_2": recall(out["retrieved"], q, gold_articles, 2),
            "recall_at_4": recall(out["retrieved"], q, gold_articles, 4),
            "recall_at_10": recall(out["retrieved"], q, gold_articles, 10),
            "gold_articles": "|".join(gold_articles),
            "retrieved": "|".join(out["retrieved"]),
            "followup_query": out.get("followup_query", ""),
            "latency_s": round(latency, 2),
        }
        rows.append(row)
        print(f"[{q['qid']}] {out['pred'][:60]!r} | EM={row['em']} F1={row['f1']:.2f}")

    out_dir = path("results")
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{args.system}_{args.split}.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    def _mean_valid(xs):
        vals = [x for x in xs if x == x]
        return sum(vals) / len(vals) if vals else None

    agg = {
        "system": args.system,
        "split": args.split,
        "n": len(rows),
        "EM": _mean_valid([r["em"] for r in rows]),
        "F1": _mean_valid([r["f1"] for r in rows]),
        "Recall@2": _mean_valid([r["recall_at_2"] for r in rows]),
        "Recall@4": _mean_valid([r["recall_at_4"] for r in rows]),
        "Recall@10": _mean_valid([r["recall_at_10"] for r in rows]),
        "avg_latency_s": _mean_valid([r["latency_s"] for r in rows]),
    }
    print("\n" + json.dumps(agg, indent=2))

if __name__ == "__main__":
    main()
