"""Milestone-only SEALED evaluation. Aggregate metrics only — no per-example dumps.

Usage: python scripts/run_sealed.py --adapter <path|base> --label <milestone-name>
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

SEALED = [
    ("t1_literal_sealed", ROOT / "benchmarks" / "literal" / "sealed" / "examples.jsonl"),
    ("t2_unseen_schema_sealed", ROOT / "benchmarks" / "unseen_schema" / "sealed" / "examples.jsonl"),
    ("t4_adversarial_sealed", ROOT / "benchmarks" / "adversarial" / "sealed" / "examples.jsonl"),
    ("sealed_final", ROOT / "benchmarks" / "sealed_final" / "examples.jsonl"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="base")
    ap.add_argument("--label", required=True)
    args = ap.parse_args()

    import torch

    from compactex.inference import LocalRunner, run_suite

    runner = LocalRunner()
    if args.adapter != "base":
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        mp = str(ROOT / "models" / "Qwen2.5-0.5B-Instruct"
                 / "7ae557604adf67be50417f59c2c2f167def9a775")
        tok = AutoTokenizer.from_pretrained(mp, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            mp, local_files_only=True, dtype=torch.bfloat16)
        model = PeftModel.from_pretrained(model, args.adapter)
        model.to("mps").eval()
        runner._model, runner._tok, runner._torch = model, tok, torch
    else:
        runner.load()

    out = ROOT / "experiments" / "sealed_runs" / args.label
    results = {}
    for name, path in SEALED:
        m = run_suite(runner, path, out / name)
        results[name] = m
        print(f"[sealed:{args.label}] {name}: f1={m['f1']:.3f} P={m['precision']:.3f} "
              f"R={m['recall']:.3f} schema={m['schema_valid_rate']:.2f} "
              f"copy={m['value_exact_copy_rate']:.2f} unsup={m['unsupported_fact_rate']:.2f}",
              flush=True)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                            capture_output=True, text=True).stdout.strip()
    (out / "metrics.json").write_text(json.dumps({
        "adapter": args.adapter, "label": args.label, "git_commit": commit,
        "suites": results}, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"[sealed:{args.label}] written -> {out / 'metrics.json'}", flush=True)


if __name__ == "__main__":
    main()
