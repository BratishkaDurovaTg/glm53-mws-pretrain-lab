"""LoRA SFT with completion-only loss, staged gates, task-level checkpoint selection.

Hand-rolled loop (AdamW + cosine + grad clip) — full control over masking and
checkpoint evaluation on MPS without Trainer-version churn.
"""

import json
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

from compactex.inference import DEFAULT_MODEL_PATH, DEFAULT_REVISION
from compactex.schemas import TaskExample, build_system_prompt, build_user_prompt, target_json
from compactex.suites import load_suite


@dataclass
class SFTConfig:
    stage: str = "pilot"  # smoke | tiny_overfit | pilot
    train_path: str = "data/train/examples.jsonl"
    dev_path: str = "data/dev/examples.jsonl"
    output_dir: str = "experiments/runs/pilot1"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj"])
    lr: float = 1e-4
    warmup_steps: int = 20
    max_steps: int = 600
    batch_size: int = 1
    grad_accum: int = 8
    max_length: int = 2048
    seed: int = 42
    eval_every: int = 200
    eval_subset: int = 60
    tiny_examples: int = 2
    tiny_steps: int = 120
    smoke_steps: int = 2
    save_limit: int = 3  # early / best / final
    init_adapter: str | None = None  # continuation: resume this LoRA, trainable


def encode_example(tok, ex: TaskExample, cfg: SFTConfig) -> dict | None:
    system = build_system_prompt(ex.schema_)
    user = build_user_prompt(ex.ticket)
    assistant = target_json(ex.gold)
    full = tok.apply_chat_template(
        [{"role": "system", "content": system}, {"role": "user", "content": user},
         {"role": "assistant", "content": assistant}],
        tokenize=True, truncation=False, padding=False,
    )
    prompt = tok.apply_chat_template(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        tokenize=True, add_generation_prompt=True, truncation=False, padding=False,
    )
    full_ids = full["input_ids"] if hasattr(full, "keys") else full
    prompt_ids = prompt["input_ids"] if hasattr(prompt, "keys") else prompt
    if hasattr(full_ids, "tolist"):
        full_ids = full_ids.tolist()
    if hasattr(prompt_ids, "tolist"):
        prompt_ids = prompt_ids.tolist()
    if len(full_ids) > cfg.max_length:
        return None
    n_prompt = len(prompt_ids)
    labels = [-100] * n_prompt + full_ids[n_prompt:]
    if all(label == -100 for label in labels):
        return None
    return {"input_ids": full_ids, "labels": labels}


def collate(batch: list[dict], pad_id: int):
    import torch

    maxlen = max(len(b["input_ids"]) for b in batch)
    pad = pad_id if pad_id is not None else 0
    input_ids, labels, attn = [], [], []
    for b in batch:
        n = len(b["input_ids"])
        input_ids.append(b["input_ids"] + [pad] * (maxlen - n))
        labels.append(b["labels"] + [-100] * (maxlen - n))
        attn.append([1] * n + [0] * (maxlen - n))
    return (
        torch.tensor(input_ids, dtype=torch.long),
        torch.tensor(labels, dtype=torch.long),
        torch.tensor(attn, dtype=torch.long),
    )


def build_model(cfg: SFTConfig, model_path: Path = DEFAULT_MODEL_PATH):
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    tok = AutoTokenizer.from_pretrained(
        str(model_path), local_files_only=True, revision=DEFAULT_REVISION,
        trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path), local_files_only=True, revision=DEFAULT_REVISION,
        use_safetensors=True, dtype=torch.bfloat16)
    model.to("mps")
    model.config.use_cache = False
    if cfg.init_adapter:
        # continuation: resume the previous LoRA weights, keep them trainable
        model = PeftModel.from_pretrained(
            model, cfg.init_adapter, is_trainable=True)
    else:
        lora = LoraConfig(
            r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
            target_modules=cfg.target_modules, bias="none", task_type="CAUSAL_LM")
        model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    return model, tok


def train(cfg: SFTConfig, model_path: Path = DEFAULT_MODEL_PATH) -> dict:
    import torch

    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    model, tok = build_model(cfg, model_path)
    examples = load_suite(Path(cfg.train_path))
    rng = random.Random(cfg.seed)
    rng.shuffle(examples)

    if cfg.stage == "smoke":
        examples, steps = examples[:8], cfg.smoke_steps
    elif cfg.stage == "tiny_overfit":
        examples, steps = examples[:cfg.tiny_examples], cfg.tiny_steps
    else:
        steps = min(cfg.max_steps, math.ceil(len(examples) / cfg.grad_accum))

    feats = [f for f in (encode_example(tok, ex, cfg) for ex in examples) if f]
    (out / "config.json").write_text(json.dumps(
        {**cfg.__dict__, "n_train_examples": len(feats), "steps": steps}, indent=2))

    model.train()
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.lr, weight_decay=0.0, fused=False)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda s: min((s + 1) / max(cfg.warmup_steps, 1),
                      0.5 * (1 + math.cos(math.pi * min(s / max(steps, 1), 1.0)))),
    )
    pad_id = tok.pad_token_id or tok.eos_token_id
    losses: list[float] = []
    checkpoints: dict[str, dict] = {}
    step, t0 = 0, time.perf_counter()
    eval_result: dict = {}
    running = 0.0
    while step < steps:
        for micro in range(cfg.grad_accum):
            idx = (step * cfg.grad_accum + micro) % len(feats)
            ids, labels, attn = collate([feats[idx]], pad_id)
            ids, labels, attn = ids.to("mps"), labels.to("mps"), attn.to("mps")
            out_logits = model(input_ids=ids, attention_mask=attn).logits
            shift_logits = out_logits[:, :-1, :]
            shift_labels = labels[:, 1:]
            # bf16 CE (no fp32 upcast of the 152k-vocab logits), sync only once per step
            loss = torch.nn.functional.cross_entropy(
                shift_logits.reshape(-1, shift_logits.size(-1)),
                shift_labels.reshape(-1), ignore_index=-100)
            (loss / cfg.grad_accum).backward()
            running = running + loss.detach()
        losses.append(float(running) / cfg.grad_accum)
        running = 0.0
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()
        sched.step()
        opt.zero_grad()
        step += 1
        del out_logits, shift_logits, loss
        if step % 25 == 0:
            torch.mps.empty_cache()
        if step % 10 == 0 or step == steps:
            recent = losses[-10:]
            print(f"step {step}/{steps} loss {sum(recent)/len(recent):.4f} "
                  f"({(time.perf_counter()-t0)/step:.2f}s/step)", flush=True)
        # checkpoints: early (~25%), periodic-eval on dev, final
        if cfg.stage == "pilot" and step == max(steps // 4, 1):
            model.save_pretrained(out / "ckpt_early")
        if cfg.stage == "pilot" and (step % cfg.eval_every == 0 or step == steps):
            f1 = quick_eval(model, tok, cfg)
            checkpoints[f"ckpt_{step}"] = {"dev_f1": f1, "step": step}
            print(f"[eval] step {step} dev extraction-F1 {f1:.3f}", flush=True)
            best = max(checkpoints.values(), key=lambda c: c["dev_f1"])["dev_f1"]
            if f1 >= best:
                model.save_pretrained(out / "ckpt_best")
    if cfg.stage == "pilot" and not checkpoints:
        model.save_pretrained(out / "ckpt_best")
    model.save_pretrained(out / "ckpt_final")
    train_seconds = time.perf_counter() - t0
    manifest = {
        "stage": cfg.stage, "steps": steps, "n_examples": len(feats),
        "loss_first": losses[0] if losses else None,
        "loss_last": sum(losses[-cfg.grad_accum:]) / max(min(len(losses), cfg.grad_accum), 1),
        "train_seconds": round(train_seconds, 1),
        "checkpoints": checkpoints or {"final_only": True},
        "eval": eval_result,
    }
    (out / "train_manifest.json").write_text(json.dumps(manifest, indent=2))
    # gate checks
    if cfg.stage == "smoke":
        assert all(math.isfinite(x) for x in losses), "smoke: non-finite loss"
        print("[gate] smoke PASS: finite updates", flush=True)
    if cfg.stage == "tiny_overfit":
        final_loss = manifest["loss_last"]
        assert final_loss < 0.05, f"tiny_overfit: loss {final_loss:.4f} did not collapse"
        print(f"[gate] tiny_overfit PASS: loss {final_loss:.5f}", flush=True)
    return manifest


def quick_eval(model, tok, cfg: SFTConfig) -> float:
    """Extraction-F1 (exact facts) on a dev subset, generation-based."""
    import torch

    from compactex.evaluation.metrics import aggregate, parse_prediction, score_record

    dev = load_suite(Path(cfg.dev_path))[: cfg.eval_subset]
    model.eval()
    recs = []
    for ex in dev:
        system = build_system_prompt(ex.schema_)
        user = build_user_prompt(ex.ticket)
        enc = tok.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            tokenize=True, add_generation_prompt=True, enable_thinking=False,
            truncation=False, padding=False)
        ids = enc["input_ids"] if hasattr(enc, "keys") else enc
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        with torch.inference_mode():
            out = model.generate(
                input_ids=torch.tensor([ids]).to("mps"),
                attention_mask=torch.ones(1, len(ids), dtype=torch.long).to("mps"),
                max_new_tokens=256, do_sample=False, num_beams=1,
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
                eos_token_id=tok.eos_token_id, use_cache=True)
        text = tok.decode(out[0].tolist()[len(ids):], skip_special_tokens=True)
        payload, flags = parse_prediction(text)
        rec = score_record(ex, payload)
        rec |= flags
        recs.append(rec)
    model.train()
    return float(aggregate(recs)["f1"])


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="pilot",
                    choices=["smoke", "tiny_overfit", "pilot"])
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()
    cfg = SFTConfig(stage=args.stage)
    if args.output_dir:
        cfg.output_dir = args.output_dir
    if args.stage == "smoke":
        cfg.output_dir = args.output_dir or "experiments/runs/smoke"
    if args.stage == "tiny_overfit":
        cfg.output_dir = args.output_dir or "experiments/runs/tiny_overfit"
    print(train(cfg))
