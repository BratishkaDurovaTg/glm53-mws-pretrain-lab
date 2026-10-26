"""Exp3 corpus: 2500 natural-breadth examples + 1000 Pilot1 replay anchor.

Single-variable discipline: same world-state machinery, same S_train schema
bank, same tech literal kinds (consumer kinds stay evaluation-only — T5 tests
zero-shot transfer to NEW literal kinds). Changed variable = surface language.
Mix documented pre-training: 2500 new (25% empty-gold, standard near-pilot1
semantic distribution) + 1000 replay from data/train (seed 42 sample).
"""

import hashlib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from compactex.factory import FactoryResult  # noqa: E402
from compactex.gold import derive_facts, sample_schema  # noqa: E402
from compactex.natural_families import NATURAL_FAMILIES, render_natural  # noqa: E402
from compactex.render import RenderAmbiguous, substitute_and_locate  # noqa: E402
from compactex.schemas import Message, TaskExample, Ticket  # noqa: E402
from compactex.suites import load_suite  # noqa: E402
from compactex.worldstate import build_world_state  # noqa: E402


def build_breadth(rng: random.Random, seed: int, fam) -> FactoryResult:
    ws = build_world_state(
        rng, ticket_id=f"b3-{seed:08x}",
        n_messages=rng.randint(*fam.n_messages),
        n_mentions=rng.randint(2, 7),
        negative_rate=0.3,
    )
    want_empty = rng.random() < 0.25
    schema = sample_schema("train", rng, ws, want_match=not want_empty)
    if schema is None:
        return FactoryResult(None, "no_negative_schema")
    try:
        rendered = render_natural(ws, fam, rng)
        roles = [r for r, _ in rendered]
        contents = [c for _, c in rendered]
        final, spans = substitute_and_locate(contents, ws, "train", rng)
        if any(not m.strip() for m in final):
            return FactoryResult(None, "empty_message")
        ticket = Ticket(
            ticket_id=f"b3-{seed:08x}",
            messages=[Message(id=f"m{i + 1}", role=roles[i], content=final[i])
                      for i in range(len(final))],
        )
        gold = derive_facts(schema, ws, spans)
        ex = TaskExample(
            id=f"b3-{seed:08d}", ticket=ticket, schema=schema, gold=gold,
            meta={"pool": "train", "family": f"natural:{fam.name}", "batch": "batch3",
                  "renderer": "glm53_natural_bank", "gold_n": len(gold),
                  "n_mentions": len(ws.mentions)},
        )
        return FactoryResult(ex)
    except RenderAmbiguous as exc:
        return FactoryResult(None, f"render_ambiguous:{exc}")
    except Exception as exc:
        return FactoryResult(None, f"{type(exc).__name__}:{exc}")


def main() -> None:
    out_dir = ROOT / "data" / "batch3"
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(31337)
    seen: set[str] = set()
    quarantined: dict[str, int] = {}
    records = []
    seed = 900_000
    attempts = 0
    while len(records) < 2500 and attempts < 60_000:
        attempts += 1
        fam = NATURAL_FAMILIES[seed % len(NATURAL_FAMILIES)]
        r = build_breadth(random.Random(seed), seed, fam)
        seed += 1
        if r.example is None:
            key = r.quarantine_reason.split(":")[0]
            quarantined[key] = quarantined.get(key, 0) + 1
            continue
        ex = r.example
        canon = hashlib.sha256(json.dumps(
            {"t": [m.content for m in ex.ticket.messages],
             "f": sorted(x.name for x in ex.schema_.fields)},
            ensure_ascii=False).encode()).hexdigest()
        if canon in seen:
            quarantined["dupe"] = quarantined.get("dupe", 0) + 1
            continue
        seen.add(canon)
        records.append(ex.model_dump(mode="json", by_alias=True))
    # replay anchor: 1000 pilot1 train examples, seed-42 sample, provenance kept
    pilot = load_suite(ROOT / "data" / "train" / "examples.jsonl")
    replay_rng = random.Random(42)
    replay = replay_rng.sample(pilot, 1000)
    for ex in replay:
        rec = ex.model_dump(mode="json", by_alias=True)
        rec["meta"] = rec.get("meta", {}) | {"batch": "batch3_replay"}
        records.append(rec)
    jsonl = out_dir / "examples.jsonl"
    jsonl.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n",
                     encoding="utf-8")
    new = [r for r in records if r.get("meta", {}).get("batch") == "batch3"]
    manifest = {
        "name": "batch3_natural_breadth",
        "n_total": len(records),
        "n_new": len(new),
        "n_replay": len(records) - len(new),
        "empty_gold_n_new": sum(1 for r in new if not r["gold"]),
        "families": sorted({r["meta"]["family"] for r in new if "family" in r["meta"]}),
        "quarantined": quarantined,
        "seed_base": 900_000,
        "sha256": hashlib.sha256(jsonl.read_bytes()).hexdigest(),
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
