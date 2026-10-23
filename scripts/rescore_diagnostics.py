"""Rescore stored raw generations with the diagnostic metric decomposition.

No inference rerun: reads <run>/<suite>/generations.jsonl + the frozen suite,
recomputes per-record scores, writes AGGREGATE diagnostics only (sealed suites
never yield per-example artifacts). Strict metric meaning is unchanged; the
historical metrics.json stays untouched.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from compactex.evaluation.metrics import (  # noqa: E402
    aggregate,
    lenient_tv_counts,
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

KEYS = ["f1", "type_value_f1", "type_f1", "evidence_accuracy_given_tv",
        "value_exact_match_given_type", "schema_valid_rate", "json_valid_rate",
        "unsupported_fact_rate", "no_fact_accuracy"]


def rescore(run_dir: Path) -> dict:
    out = {}
    for suite, path in SUITES.items():
        gen_path = run_dir / suite / "generations.jsonl"
        if not gen_path.exists():
            continue
        gens = {}
        for line in gen_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                gens[row["id"]] = row["raw"]
        recs = []
        lenient = []
        for ex in load_suite(path):
            raw = gens.get(ex.id, "")
            payload, flags = parse_prediction(raw)
            rec = score_record(ex, payload)
            rec |= flags
            recs.append(rec)
            lenient.append(lenient_tv_counts(ex, raw))
        agg = aggregate(recs)
        ltp = sum(r["tv_tp"] for r in lenient)
        lfp = sum(r["tv_fp"] for r in lenient)
        lfn = sum(r["tv_fn"] for r in lenient)
        lp = ltp / (ltp + lfp) if ltp + lfp else 0.0
        lr = ltp / (ltp + lfn) if ltp + lfn else 0.0
        agg["lenient_type_value_f1"] = (
            2 * lp * lr / (lp + lr) if lp + lr else 0.0)
        out[suite] = {k: agg[k] for k in KEYS + ["lenient_type_value_f1"]}
    return out


if __name__ == "__main__":
    default_dir = ROOT / "experiments" / "baseline_base_qwen"
    run_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else default_dir
    result = rescore(run_dir)
    dest = run_dir / "diagnostics.json"
    dest.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    for suite, m in result.items():
        print(f"{suite}: strict={m['f1']:.3f} tv={m['type_value_f1']:.3f} "
              f"lenient_tv={m['lenient_type_value_f1']:.3f} "
              f"type_f1={m['type_f1']:.3f} schema={m['schema_valid_rate']:.2f}")
    print("->", dest)
