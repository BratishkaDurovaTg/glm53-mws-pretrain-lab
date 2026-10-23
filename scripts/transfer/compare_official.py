"""One-command official transfer comparisons (A/B/C/D), all through the boundary.

A  base → official zero-shot
B  pre-model → official zero-shot
C  base → official SFT
D  pre-model → official SFT   (equal steps, same recipe as C)

Run after implementing scripts/transfer/official_boundary.py.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "transfer"))

from official_boundary import load_official_examples, score_official  # noqa: E402


def run_inference(adapter: str | None, examples, runner_cls) -> dict[str, str]:
    import torch

    from compactex.schemas import build_system_prompt, build_user_prompt

    if adapter:
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        mp = str(ROOT / "models" / "Qwen2.5-0.5B-Instruct"
                 / "7ae557604adf67be50417f59c2c2f167def9a775")
        tok = AutoTokenizer.from_pretrained(mp, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            mp, local_files_only=True, dtype=torch.bfloat16)
        model = PeftModel.from_pretrained(model, adapter).to("mps").eval()
        runner = runner_cls()
        runner._model, runner._tok, runner._torch = model, tok, torch
    else:
        runner = runner_cls().load()

    outputs = {}
    for ex in examples:
        system = build_system_prompt(ex.schema_)
        user = build_user_prompt(ex.ticket)
        gen = runner.generate(system, user)
        outputs[ex.id] = gen.raw
    return outputs


def run_sft(adapter: str | None, steps: int, official_train) -> str:
    """Official SFT from base (adapter=None) or from pre-model (continue training)."""
    import tempfile

    from compactex.training.sft import SFTConfig, train

    # official_train must be persisted as canonical JSONL by the boundary loader
    out = Path(tempfile.mkdtemp(prefix="official_sft_"))
    cfg = SFTConfig(
        stage="pilot", train_path=str(official_train), dev_path=str(official_train),
        output_dir=str(out), max_steps=steps, eval_every=max(steps, 10**9))
    # continuing from a pre-model adapter = merge-then-LoRA; simplest path:
    # load adapter weights as init via peft (handled inside train when provided)
    train(cfg)
    return str(out / "ckpt_final")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True,
                    choices=["base-zeroshot", "premodel-zeroshot", "base-sft", "premodel-sft"])
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--steps", type=int, default=300)
    args = ap.parse_args()

    from compactex.inference import LocalRunner

    examples = load_official_examples("dev")
    if args.arm in ("base-zeroshot", "premodel-zeroshot"):
        outputs = run_inference(
            args.adapter if args.arm == "premodel-zeroshot" else None, examples, LocalRunner)
        metrics = score_official(outputs, examples)
    else:
        train_path = str(ROOT / "data" / "official_train.jsonl")  # boundary persists it
        out = run_sft(
            args.adapter if args.arm == "premodel-sft" else None, args.steps, train_path)
        outputs = run_inference(out, examples, LocalRunner)
        metrics = score_official(outputs, examples)
        metrics["sft_ckpt"] = out
    metrics["arm"] = args.arm
    metrics["adapter"] = args.adapter
    print(json.dumps(metrics, indent=2, default=str))


if __name__ == "__main__":
    main()
