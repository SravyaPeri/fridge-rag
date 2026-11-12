from __future__ import annotations

import argparse
import hashlib
import json

from fridge.core import ROOT, Index, LocalEncoder
from fridge.generation import DEFAULT_MODEL, SAFETY, recommend


def main() -> None:
    parser = argparse.ArgumentParser(description="Fridge Notes local RAG demo")
    parser.add_argument("command", choices=["index", "query", "evaluate"])
    parser.add_argument("--query", default="Half a can of black beans, stale tortillas, and a sad lime")
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()
    encoder = LocalEncoder()
    index = Index(ROOT / "data/recipes.sqlite3")
    if args.command == "index":
        print(json.dumps(index.sync(ROOT / "recipes", ROOT / "data/uploads", encoder), indent=2))
    elif args.command == "query":
        results = index.search(args.query, encoder)
        print(json.dumps([{key: r[key] for key in
                           ("id", "title", "source", "score", "matched", "missing")} for r in results], indent=2))
        if args.generate:
            answer = recommend(args.query, results, args.model)
            if not answer:
                print("\nNo supported recommendation. Add a relevant recipe or try another query.")
            for item in answer:
                print(f"\n{item['recipe']['title']} {item['citation']}\n{item['reason']}\n"
                      f"Not mentioned (excluding pantry): {', '.join(item['recipe']['missing']) or 'None'}\n"
                      f"{item['recipe']['steps']}")
            print(f"\nSafety: {SAFETY}")
    else:
        cases = json.loads((ROOT / "evaluation.json").read_text())
        hits, top1, reciprocal, details = 0, 0, 0.0, []
        for case in cases:
            results = index.search(case["query"], encoder, limit=3)
            ranks = [i + 1 for i, r in enumerate(results) if r["source"] == case["expected_source"]]
            rank = ranks[0] if ranks else None
            hits += rank is not None
            top1 += rank == 1
            reciprocal += 1 / rank if rank else 0
            details.append({**case, "expected_id": hashlib.sha256(case["expected_source"].encode()).hexdigest()[:12],
                            "rank": rank, "retrieved_ids": [r["id"] for r in results]})
        report = {"cases": len(cases), "hit_rate_at_3": hits / len(cases),
                  "top_1_accuracy": top1 / len(cases), "mrr_at_3": reciprocal / len(cases),
                  "model": encoder.fingerprint, "details": details}
        output = ROOT / "data/evaluation-report.json"
        output.write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
