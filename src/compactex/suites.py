"""Suite generation and freezing: splits + T1/T2/T4 with manifests and dedup."""

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from compactex.factory import build_example
from compactex.families import families_for_pool
from compactex.literals import pool_of_value
from compactex.schemas import TaskExample

LITERAL_FOCUS = {
    "error_code": 3.0, "version": 3.0, "uuid": 2.0, "ip": 2.0, "url": 1.5,
    "host": 1.5, "path": 1.5, "resource_id": 1.5, "ticket_ref": 1.0, "datetime": 1.0,
    "port": 0.5,
}


@dataclass
class SuiteConfig:
    name: str
    pool: str
    n: int
    seed_base: int
    empty_rate: float = 0.3
    negative_rate: float = 0.3
    kind_weights: dict[str, float] | None = None
    status_weights: dict[str, float] | None = None
    n_mentions: tuple[int, int] = (2, 7)
    family_filter: list[str] | None = None
    note: str = ""


@dataclass
class SuiteResult:
    config: SuiteConfig
    examples: list[TaskExample] = field(default_factory=list)
    quarantined: dict[str, int] = field(default_factory=dict)
    dropped_dupe: int = 0


def _canonical_key(ex: TaskExample) -> str:
    payload = {
        "ticket": [m.content for m in ex.ticket.messages],
        "fields": sorted(f.name for f in ex.schema_.fields),
        "gold": sorted((f.type, f.value) for f in ex.gold),
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()


def _pool_leaks(examples: list[TaskExample], pool: str) -> list[str]:
    leaks = []
    for ex in examples:
        for fact in ex.gold:
            p = pool_of_value(fact.value)
            if p is not None and p != pool:
                leaks.append(f"{ex.id}:{fact.value}:{p}")
    return leaks


def generate_suite(cfg: SuiteConfig) -> SuiteResult:
    fams = families_for_pool(cfg.pool)
    if cfg.family_filter:
        fams = [f for f in fams if f.name in cfg.family_filter]
    result = SuiteResult(config=cfg)
    seen: set[str] = set()
    seed = cfg.seed_base
    attempts = 0
    while len(result.examples) < cfg.n and attempts < cfg.n * 20:
        attempts += 1
        fam = fams[seed % len(fams)]
        built = build_example(
            seed, cfg.pool, fam,
            kind_weights=cfg.kind_weights,
            status_weights=cfg.status_weights,
            n_mentions=cfg.n_mentions,
            negative_rate=cfg.negative_rate,
            empty_rate=cfg.empty_rate,
            example_id=f"{cfg.name}-{len(result.examples):05d}",
        )
        seed += 1
        if built.example is None:
            reason = built.quarantine_reason.split(":")[0]
            result.quarantined[reason] = result.quarantined.get(reason, 0) + 1
            continue
        key = _canonical_key(built.example)
        if key in seen:
            result.dropped_dupe += 1
            continue
        seen.add(key)
        result.examples.append(built.example)
    return result


def freeze_suite(result: SuiteResult, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    records = [
        ex.model_dump(mode="json", by_alias=True) | {"gold_n": len(ex.gold)}
        | {"meta": ex.meta}
        for ex in result.examples
    ]
    jsonl = out_dir / "examples.jsonl"
    jsonl.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n",
        encoding="utf-8",
    )
    leaks = _pool_leaks(result.examples, result.config.pool)
    manifest = {
        "name": result.config.name,
        "pool": result.config.pool,
        "n": len(result.examples),
        "config": {k: v for k, v in asdict(result.config).items()},
        "quarantined": result.quarantined,
        "dropped_duplicates": result.dropped_dupe,
        "empty_gold_n": sum(1 for ex in result.examples if not ex.gold),
        "pool_leaks": leaks,
        "sha256": hashlib.sha256(jsonl.read_bytes()).hexdigest(),
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return jsonl


def load_suite(path: Path) -> list[TaskExample]:
    examples = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            data = json.loads(line)
            data.pop("gold_n", None)
            examples.append(TaskExample.model_validate(data))
    return examples
