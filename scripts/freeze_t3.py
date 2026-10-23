"""Freeze T3 (hand-written technical holdout) via the same mechanical pipeline."""

import hashlib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "benchmarks" / "technical"))

from compactex.gold import build_field, derive_facts  # noqa: E402
from compactex.render import substitute_and_locate  # noqa: E402
from compactex.schemas import Message, TaskExample, TaskSchema, Ticket  # noqa: E402
from compactex.worldstate import Mention, MessageSpec, WorldState  # noqa: E402
from handwritten import HANDWRITTEN  # noqa: E402


def build(entry: dict, rng: random.Random) -> TaskExample:
    msgs = [
        MessageSpec(index=i + 1, role=role, mention_tokens=[])
        for i, role in enumerate(entry["roles"])
    ]
    mentions = []
    for token, kind, status, component in entry["mentions"]:
        found = [i + 1 for i, text in enumerate(entry["texts"]) if token in text]
        if len(found) != 1:
            raise ValueError(f"{entry['id']}: token {token} appears in {found} messages")
        m = Mention(
            token=token, kind=kind, status=status,
            message_index=found[0], reported_by=msgs[found[0] - 1].role,
            of_component=component,
        )
        mentions.append(m)
        msgs[found[0] - 1].mention_tokens.append(token)
    ws = WorldState(ticket_id=entry["id"], messages=msgs, mentions=mentions)
    schema = TaskSchema(fields=[
        build_field(__import__("compactex.schemas", fromlist=["Selector"]).Selector(
            kind=kind, status=status))
        for kind, status in entry["selectors"]
    ])
    final, spans = substitute_and_locate(entry["texts"], ws, "dev", rng)
    ticket = Ticket(
        ticket_id=entry["id"],
        messages=[Message(id=f"m{i + 1}", role=entry["roles"][i], content=final[i])
                  for i in range(len(final))],
    )
    gold = derive_facts(schema, ws, spans)
    return TaskExample(
        id=entry["id"], ticket=ticket, schema=schema, gold=gold,
        meta={"pool": "dev", "family": "handwritten", "origin": "glm_hand_rendered",
              "gold_n": len(gold)},
    )


def main() -> None:
    out = ROOT / "benchmarks" / "technical"
    rng = random.Random(20261023)
    examples = []
    for entry in HANDWRITTEN:
        examples.append(build(entry, rng))
    records = [ex.model_dump(mode="json", by_alias=True) for ex in examples]
    jsonl = out / "examples.jsonl"
    jsonl.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n",
        encoding="utf-8")
    manifest = {
        "name": "t3_technical",
        "origin": "GLM hand-written tickets (non-template), gold mechanical",
        "n": len(examples),
        "empty_gold_n": sum(1 for ex in examples if not ex.gold),
        "total_gold_facts": sum(len(ex.gold) for ex in examples),
        "sha256": hashlib.sha256(jsonl.read_bytes()).hexdigest(),
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print("t3:", manifest)


if __name__ == "__main__":
    main()
