"""Freeze a hand-authored suite (T3/T5 style) via the mechanical pipeline."""

import hashlib
import importlib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def build(entry: dict, rng: random.Random, pool: str):
    from compactex.gold import build_field, derive_facts
    from compactex.render import substitute_and_locate
    from compactex.schemas import (
        Evidence, Message, Selector, TaskExample, TaskSchema, Ticket,
    )
    from compactex.worldstate import Mention, MessageSpec, WorldState

    msgs = [MessageSpec(index=i + 1, role=role, mention_tokens=[])
            for i, role in enumerate(entry["roles"])]
    mentions = []
    for token, kind, status, component in entry["mentions"]:
        found = [i + 1 for i, text in enumerate(entry["texts"]) if token in text]
        if len(found) != 1:
            raise ValueError(f"{entry['id']}: token {token} in messages {found}")
        m = Mention(token=token, kind=kind, status=status,
                    message_index=found[0], reported_by=msgs[found[0] - 1].role,
                    of_component=component)
        mentions.append(m)
        msgs[found[0] - 1].mention_tokens.append(token)
    ws = WorldState(ticket_id=entry["id"], messages=msgs, mentions=mentions)
    schema = TaskSchema(fields=[build_field(Selector(kind=k, status=s))
                                for k, s in entry["selectors"]])
    final, spans = substitute_and_locate(entry["texts"], ws, pool, rng)
    ticket = Ticket(ticket_id=entry["id"],
                    messages=[Message(id=f"m{i + 1}", role=entry["roles"][i],
                                      content=final[i])
                              for i in range(len(final))])
    gold = derive_facts(schema, ws, spans)
    return TaskExample(
        id=entry["id"], ticket=ticket, schema=schema, gold=gold,
        meta={"pool": pool, "family": "handwritten", "gold_n": len(gold)},
    )


def main() -> None:
    module_name, out_dir, pool, seed = sys.argv[1], Path(sys.argv[2]), sys.argv[3], int(sys.argv[4])
    sys.path.insert(0, str(out_dir))
    sys.path.insert(0, str(out_dir.parent))
    entries = importlib.import_module(module_name).T5 if "t5" in module_name \
        else importlib.import_module(module_name).HANDWRITTEN
    rng = random.Random(seed)
    examples = [build(e, rng, pool) for e in entries]
    out_dir.mkdir(parents=True, exist_ok=True)
    records = [ex.model_dump(mode="json", by_alias=True) for ex in examples]
    jsonl = out_dir / "examples.jsonl"
    jsonl.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n",
                     encoding="utf-8")
    manifest = {
        "name": out_dir.name,
        "origin": f"hand-authored ({module_name}), gold mechanical, pool={pool}",
        "n": len(examples),
        "empty_gold_n": sum(1 for ex in examples if not ex.gold),
        "total_gold_facts": sum(len(ex.gold) for ex in examples),
        "sha256": hashlib.sha256(jsonl.read_bytes()).hexdigest(),
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
