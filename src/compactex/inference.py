"""Local-only HF inference over frozen suites. Greedy, deterministic, no downloads."""

import json
import time
from dataclasses import dataclass
from pathlib import Path

from compactex.evaluation.metrics import aggregate, parse_prediction, score_record
from compactex.schemas import build_system_prompt, build_user_prompt
from compactex.suites import load_suite

DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parents[2]
    / "models" / "Qwen2.5-0.5B-Instruct"
    / "7ae557604adf67be50417f59c2c2f167def9a775"
)
DEFAULT_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"


@dataclass
class GenResult:
    raw: str
    generated_tokens: int
    seconds: float


class LocalRunner:
    def __init__(self, model_path: Path | str = DEFAULT_MODEL_PATH,
                 revision: str = DEFAULT_REVISION, device: str = "mps",
                 dtype: str = "bfloat16", max_new_tokens: int = 256,
                 max_input_tokens: int = 4096, seed: int = 42):
        self.model_path = str(model_path)
        self.revision = revision
        self.device = device
        self.dtype = dtype
        self.max_new_tokens = max_new_tokens
        self.max_input_tokens = max_input_tokens
        self.seed = seed
        self._model = None
        self._tok = None
        self._torch = None

    def load(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        torch.manual_seed(self.seed)
        self._tok = AutoTokenizer.from_pretrained(
            self.model_path, local_files_only=True, trust_remote_code=False,
            revision=self.revision,
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_path, local_files_only=True, trust_remote_code=False,
            revision=self.revision, use_safetensors=True,
            dtype=getattr(torch, self.dtype),
        )
        self._model.to(self.device)
        self._model.eval()
        return self

    def generate(self, system: str, user: str) -> GenResult:
        import torch
        from transformers import GenerationConfig

        tok = self._tok
        encoded = tok.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            tokenize=True, add_generation_prompt=True, enable_thinking=False,
            truncation=False, padding=False,
        )
        if hasattr(encoded, "keys"):  # BatchEncoding/dict (transformers 5.x)
            ids = encoded["input_ids"]
        else:
            ids = encoded
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        if len(ids) > self.max_input_tokens:
            raise ValueError("input_too_long")
        torch.manual_seed(self.seed)
        inputs = torch.tensor([ids], dtype=torch.long).to(self.device)
        attention = torch.ones_like(inputs)
        cfg = GenerationConfig(
            do_sample=False, num_beams=1, max_new_tokens=self.max_new_tokens,
            pad_token_id=tok.pad_token_id or tok.eos_token_id,
            eos_token_id=tok.eos_token_id, use_cache=True,
        )
        t0 = time.perf_counter()
        with torch.inference_mode():
            out = self._model.generate(
                input_ids=inputs, attention_mask=attention, generation_config=cfg
            )
        seconds = time.perf_counter() - t0
        new_tokens = out[0].tolist()[len(ids):]
        text = tok.decode(new_tokens, skip_special_tokens=True)
        return GenResult(raw=text, generated_tokens=len(new_tokens), seconds=seconds)


def run_suite(runner: LocalRunner, suite_path: Path, out_dir: Path,
              limit: int | None = None, skip_examples: int = 0) -> dict:
    examples = load_suite(suite_path)
    if skip_examples:
        examples = examples[skip_examples:]
    if limit:
        examples = examples[:limit]
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    per_record = []
    raw_lines = []
    for ex in examples:
        system = build_system_prompt(ex.schema_)
        user = build_user_prompt(ex.ticket)
        try:
            gen = runner.generate(system, user)
            raw, gen_tokens, secs = gen.raw, gen.generated_tokens, gen.seconds
        except ValueError:  # input_too_long
            raw, gen_tokens, secs = "", 0, 0.0
        payload, flags = parse_prediction(raw)
        rec = score_record(ex, payload)
        rec |= flags
        rec["generated_tokens"] = gen_tokens
        rec["seconds"] = secs
        per_record.append(rec)
        raw_lines.append(json.dumps({
            "id": ex.id, "raw": raw, "generated_tokens": gen_tokens, "seconds": secs,
        }, ensure_ascii=False))
        records.append(rec)
    metrics = aggregate(per_record)
    (out_dir / "generations.jsonl").write_text(
        "\n".join(raw_lines) + "\n", encoding="utf-8")
    return metrics
