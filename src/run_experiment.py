import argparse, json, time, csv
import pandas as pd
from pathlib import Path

from src.config import CFG, path
from src.baselines import llm_only, single_shot_rag
from src.multihop import multihop_rag
from src.evaluate import exact_match, f1, recall_at_k

SYSTEMS = {"llm_only": llm_only, "rag": single_shot_rag, "multihop": multihop_rag}

def load_questions(split: str) -> list[dict]:
    qs = [json.loads(l) for l in open(path("questions"), encoding="utf-8")]
    split_ids = set(json.load(open(path("splits") / f"{split}.json")))
    return [q for q in qs if q["qid"] in split_ids]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", choices=SYSTEMS, required=True)
    ap.add_argument("--split", choices=["dev", "test"], default="dev")
    args = ap.parse_args()

    fn = SYSTEMS[args.system]
    questions = load_questions(args.split)
    rows = []

    for q in questions:
        t0 = time.time()
        out = fn(q["question"])
        latency = time.time() - t0
        gold = q["answer"]
        gold_ids = q.get("supporting_chunks", [])

        row = {
            "qid": q["qid"],
            "question": q["question"],
            "gold": gold,
            "pred": out["pred"],
            "em": exact_match(out["pred"], gold),
            "f1": f1(out["pred"], gold),
            "recall_at_4": recall_at_k(out["retrieved"], gold_ids, 4),
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

    agg = {
        "system": args.system, "split": args.split, "n": len(rows),
        "EM": sum(r["em"] for r in rows) / len(rows),
        "F1": sum(r["f1"] for r in rows) / len(rows),
        "Recall@4": sum(r["recall_at_4"] for r in rows
                        if r["recall_at_4"] == r["recall_at_4"]) / len(rows),
        "avg_latency_s": sum(r["latency_s"] for r in rows) / len(rows),
    }
    print("\n" + json.dumps(agg, indent=2))

if __name__ == "__main__":
    main()