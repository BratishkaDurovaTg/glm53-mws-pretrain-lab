"""Offline comparison of two run dirs (per-record metrics + paired bootstrap).

Reads generations.jsonl + the suite files, recomputes per-record scores purely,
then reports deltas with paired bootstrap CIs.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from compactex.evaluation.metrics import (  # noqa: E402
    paired_bootstrap_delta,
    parse_prediction,
    score_record,
)
from compactex.suites import load_suite  # noqa: E402

SUITES = {
    "dev": ROOT / "data" / "dev" / "examples.jsonl",
    "t1_literal_dev": ROOT / "benchmarks" / "literal" / "dev" / "examples.jsonl",
    "t2_unseen_schema_dev": ROOT / "benchmarks" / "unseen_schema" / "dev" / "examples.jsonl",
    "t4_adversarial_dev": ROOT / "benchmarks" / "adversarial" / "dev" / "examples.jsonl",
    "t3_technical": ROOT / "benchmarks" / "technical" / "examples.jsonl",
    "t1_literal_sealed": ROOT / "benchmarks" / "literal" / "sealed" / "examples.jsonl",
    "t2_unseen_schema_sealed": ROOT / "benchmarks" / "unseen_schema" / "sealed" / "examples.jsonl",
    "t4_adversarial_sealed": ROOT / "benchmarks" / "adversarial" / "sealed" / "examples.jsonl",
    "sealed_final": ROOT / "benchmarks" / "sealed_final" / "examples.jsonl",
}


def per_record_f1(run_dir: Path, suite_name: str) -> dict[str, float]:
    gens = {}
    gen_path = run_dir / suite_name / "generations.jsonl"
    for line in gen_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            gens[row["id"]] = row["raw"]
    examples = load_suite(SUITES[suite_name])
    out = {}
    for ex in examples:
        payload, flags = parse_prediction(gens.get(ex.id, ""))
        rec = score_record(ex, payload)
        tp, fp, fn = rec["tp"], rec["fp"], rec["fn"]
        out[ex.id] = (2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else 1.0
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True, help="run dir of baseline")
    ap.add_argument("--tuned", required=True, help="run dir of tuned model")
    ap.add_argument("--suites", nargs="+", default=["dev", "t1_literal_dev",
                    "t2_unseen_schema_dev", "t4_adversarial_dev", "t3_technical"])
    args = ap.parse_args()

    for suite in args.suites:
        base = per_record_f1(Path(args.baseline), suite)
        tuned = per_record_f1(Path(args.tuned), suite)
        ids = sorted(set(base) & set(tuned))
        b = [base[i] for i in ids]
        t = [tuned[i] for i in ids]
        lo, hi, mean = paired_bootstrap_delta(t, b, B=2000, seed=7)
        print(f"{suite}: base_f1={sum(b)/len(b):.3f} tuned_f1={sum(t)/len(t):.3f} "
              f"delta={mean:+.3f} 95%CI[{lo:+.3f},{hi:+.3f}] n={len(ids)} "
              f"{'SIGNIFICANT' if lo > 0 or hi < 0 else 'inconclusive'}")


if __name__ == "__main__":
    main()
