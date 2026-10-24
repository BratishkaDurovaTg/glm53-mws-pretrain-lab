"""Offline comparison of two run dirs with CORRECTED metric semantics.

Three grounding levels, never mixed in one delta:
- lenient (type, value) via the SAME harvester on BOTH runs (apples-to-apples
  semantics, works for non-canonical base outputs);
- message-grounded (type, value, message_id) — requires canonical parse;
- exact-grounded (type, value, message_id, start, end) — gold span offsets.

Paired bootstrap CIs per suite. Dev-side suites only: sealed per-record files
were destroyed by the sealed boundary policy (aggregate-only there).
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from compactex.evaluation.metrics import (  # noqa: E402
    lenient_tv_counts,
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
}

DEV_DEFAULT = ["t2_unseen_schema_dev", "t3_technical", "t4_adversarial_dev",
               "t1_literal_dev", "dev"]


def _f1(tp: int, fp: int, fn: int) -> float:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0


def per_record(run_dir: Path, suite_name: str) -> dict[str, dict[str, float]]:
    gens = {}
    for line in (run_dir / suite_name / "generations.jsonl").read_text(
            encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            gens[row["id"]] = row["raw"]
    out = {}
    for ex in load_suite(SUITES[suite_name]):
        raw = gens.get(ex.id, "")
        payload, _flags = parse_prediction(raw)
        rec = score_record(ex, payload)
        lenient = lenient_tv_counts(ex, raw)
        out[ex.id] = {
            "lenient_tv": _f1(lenient["tv_tp"], lenient["tv_fp"], lenient["tv_fn"]),
            "message": _f1(rec["tp"], rec["fp"], rec["fn"]),
            "exact": _f1(rec["exact_tp"], rec["exact_fp"], rec["exact_fn"]),
            "lenient_tp": lenient["tv_tp"], "lenient_fp": lenient["tv_fp"],
            "lenient_fn": lenient["tv_fn"],
            "msg_tp": rec["tp"], "msg_fp": rec["fp"], "msg_fn": rec["fn"],
            "exact_tp": rec["exact_tp"], "exact_fp": rec["exact_fp"],
            "exact_fn": rec["exact_fn"],
        }
    return out


def paired_ci(base: dict, tuned: dict, ids: list[str], field: str):
    b = [base[i][field] for i in ids]
    t = [tuned[i][field] for i in ids]
    return paired_bootstrap_delta(t, b, B=2000, seed=7)


def micro_f1(records: dict, ids: list[str], prefix: str) -> float:
    tp = sum(records[i][f"{prefix}_tp"] for i in ids)
    fp = sum(records[i][f"{prefix}_fp"] for i in ids)
    fn = sum(records[i][f"{prefix}_fn"] for i in ids)
    return _f1(tp, fp, fn)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--tuned", required=True)
    ap.add_argument("--suites", nargs="+", default=DEV_DEFAULT)
    args = ap.parse_args()

    for suite in args.suites:
        base = per_record(Path(args.baseline), suite)
        tuned = per_record(Path(args.tuned), suite)
        ids = sorted(set(base) & set(tuned))
        lines = [f"{suite} (n={len(ids)}):"]
        for field, label, prefix in [
                ("lenient_tv", "semantic tv (same harvester)", "lenient"),
                ("message", "message-grounded", "msg"),
                ("exact", "exact-grounded", "exact")]:
            lo, hi, mean = paired_ci(base, tuned, ids, field)
            b_micro = micro_f1(base, ids, prefix)
            t_micro = micro_f1(tuned, ids, prefix)
            sig = "SIG" if (lo > 0 or hi < 0) else "ns"
            lines.append(
                f"  {label:32s} micro base={b_micro:.3f} tuned={t_micro:.3f} "
                f"delta={mean:+.3f} CI[{lo:+.3f},{hi:+.3f}] {sig}")
        print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
