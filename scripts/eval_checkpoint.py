"""Evaluate a LoRA checkpoint (base+adapter) on DEV-side suites + T3.

Sealed suites are NOT touched here — separate milestone-only runner.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from compactex.inference import LocalRunner, run_suite  # noqa: E402

SUITES = [
    ("dev", ROOT / "data" / "dev" / "examples.jsonl"),
    ("t1_literal_dev", ROOT / "benchmarks" / "literal" / "dev" / "examples.jsonl"),
    ("t2_unseen_schema_dev", ROOT / "benchmarks" / "unseen_schema" / "dev" / "examples.jsonl"),
    ("t4_adversarial_dev", ROOT / "benchmarks" / "adversarial" / "dev" / "examples.jsonl"),
    ("t3_technical", ROOT / "benchmarks" / "technical" / "examples.jsonl"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default=None)
    args = ap.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    mp = str(ROOT / "models" / "Qwen2.5-0.5B-Instruct" / "7ae557604adf67be50417f59c2c2f167def9a775")
    tok = AutoTokenizer.from_pretrained(mp, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(mp, local_files_only=True, dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, args.adapter)
    model.to("mps").eval()

    runner = LocalRunner()
    runner._model, runner._tok, runner._torch = model, tok, torch

    out = Path(args.out)
    results = {}
    for name, path in SUITES:
        t0 = time.perf_counter()
        m = run_suite(runner, path, out / name)
        m["wall_seconds"] = round(time.perf_counter() - t0, 1)
        results[name] = m
        print(f"[{args.label or 'ckpt'}] {name}: f1={m['f1']:.3f} "
              f"P={m['precision']:.3f} R={m['recall']:.3f} "
              f"schema={m['schema_valid_rate']:.2f} copy={m['value_exact_copy_rate']:.2f} "
              f"unsup={m['unsupported_fact_rate']:.2f} nofact={m['no_fact_accuracy']}",
              flush=True)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                            capture_output=True, text=True).stdout.strip()
    (out / "metrics.json").write_text(json.dumps({
        "adapter": args.adapter, "label": args.label, "git_commit": commit,
        "suites": results}, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
