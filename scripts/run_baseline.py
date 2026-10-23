"""Run the BASE model over every frozen suite. Saves metrics + full provenance."""

import json
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from compactex.inference import DEFAULT_MODEL_PATH, DEFAULT_REVISION, LocalRunner, run_suite  # noqa: E402

SUITES = [
    ("dev", ROOT / "data" / "dev" / "examples.jsonl"),
    ("t1_literal_dev", ROOT / "benchmarks" / "literal" / "dev" / "examples.jsonl"),
    ("t2_unseen_schema_dev",
     ROOT / "benchmarks" / "unseen_schema" / "dev" / "examples.jsonl"),
    ("t4_adversarial_dev",
     ROOT / "benchmarks" / "adversarial" / "dev" / "examples.jsonl"),
    ("t1_literal_sealed", ROOT / "benchmarks" / "literal" / "sealed" / "examples.jsonl"),
    ("t2_unseen_schema_sealed",
     ROOT / "benchmarks" / "unseen_schema" / "sealed" / "examples.jsonl"),
    ("t4_adversarial_sealed", ROOT / "benchmarks" / "adversarial" / "sealed" / "examples.jsonl"),
    ("sealed_final", ROOT / "benchmarks" / "sealed_final" / "examples.jsonl"),
]


def main() -> None:
    out_root = ROOT / "experiments" / "baseline_base_qwen"
    out_root.mkdir(parents=True, exist_ok=True)
    runner = LocalRunner().load()
    all_metrics = {}
    for name, path in SUITES:
        t0 = time.perf_counter()
        print(f"[baseline] {name} ...", flush=True)
        metrics = run_suite(runner, path, out_root / name)
        metrics["wall_seconds"] = round(time.perf_counter() - t0, 1)
        all_metrics[name] = metrics
        print(f"[baseline] {name}: f1={metrics['f1']:.3f} "
              f"schema_valid={metrics['schema_valid_rate']:.2f} "
              f"json_valid={metrics['json_valid_rate']:.2f} "
              f"({metrics['wall_seconds']}s)", flush=True)
    provenance = {
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "revision": DEFAULT_REVISION,
        "model_path": str(DEFAULT_MODEL_PATH),
        "decoding": "greedy, max_new_tokens=256, enable_thinking=False, seed=42",
        "device": "mps", "dtype": "bfloat16",
        "python": platform.python_version(),
        "machine": platform.machine(),
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True
        ).stdout.strip(),
    }
    (out_root / "metrics.json").write_text(
        json.dumps({"provenance": provenance, "suites": all_metrics}, ensure_ascii=False,
                   indent=2, default=str),
        encoding="utf-8",
    )
    print("[baseline] done ->", out_root / "metrics.json", flush=True)


if __name__ == "__main__":
    main()
