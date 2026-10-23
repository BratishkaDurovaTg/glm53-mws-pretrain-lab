"""Milestone-only SEALED evaluation. Aggregate metrics only — no per-example dumps.

Boundary policy (owner directive): the iterative research loop must never see
sealed raw texts, gold, per-example predictions or individual failures. This
script scores in memory, persists ONLY aggregate metrics, and deletes the
per-example generation files immediately. Sealed examples.jsonl suites stay
frozen and hash-manifested; nothing here regenerates or modifies them.
Documented limitation: the boundary is procedural/tooling, not physical.

Usage: python scripts/run_sealed.py --adapter <path|base> --label <milestone-name>
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

SEALED = [
    ("t1_literal_sealed", ROOT / "benchmarks" / "literal" / "sealed" / "examples.jsonl"),
    ("t2_unseen_schema_sealed",
     ROOT / "benchmarks" / "unseen_schema" / "sealed" / "examples.jsonl"),
    ("t4_adversarial_sealed",
     ROOT / "benchmarks" / "adversarial" / "sealed" / "examples.jsonl"),
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
        suite_out = out / name
        m = run_suite(runner, path, suite_out)
        results[name] = m
        # aggregate-only boundary: destroy per-example artifacts immediately
        shutil.rmtree(suite_out, ignore_errors=True)
        print(f"[sealed:{args.label}] {name}: f1={m['f1']:.3f} P={m['precision']:.3f} "
              f"R={m['recall']:.3f} tv_f1={m['type_value_f1']:.3f} "
              f"schema={m['schema_valid_rate']:.2f} "
              f"unsup={m['unsupported_fact_rate']:.2f}",
              flush=True)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                            capture_output=True, text=True).stdout.strip()
    (out / "metrics.json").write_text(json.dumps({
        "adapter": args.adapter, "label": args.label, "git_commit": commit,
        "suites": results}, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"[sealed:{args.label}] written -> {out / 'metrics.json'}", flush=True)


if __name__ == "__main__":
    main()
