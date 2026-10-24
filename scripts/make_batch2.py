"""Experiment 2 data: abstention-targeted batch2 (NEW corpus, pilot1 untouched).

Mix (documented before training):
- 45%  empty-gold hard negatives: literal-rich scenes where the requested kind is
  present but the schema definition filters everything out (wrong status/speaker/
  component/message). Rich text, not trivial empties.
- 27.5% near-miss disambiguation: 3-6 same-kind candidates, filtered selector
  picks exactly a subset.
- 27.5% mixed positive/negative: one field with gold, another field empty.
Fresh seeds (700_000+), train pool for schemas/families/literals, aggregate
failure taxonomy only (no DEV/T2/T4 example reuse, nothing sealed).
"""

import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from compactex.factory import FactoryResult  # noqa: E402
from compactex.families import render_ticket  # noqa: E402
from compactex.gold import SCHEMA_BANKS, build_field, selector_matches  # noqa: E402
from compactex.render import RenderAmbiguous, substitute_and_locate  # noqa: E402
from compactex.schemas import (  # noqa: E402
    Message, TaskExample, TaskSchema, Ticket,
)
from compactex.worldstate import KINDS, build_world_state  # noqa: E402

RICH_KIND_SETS = [
    {"error_code": 4, "version": 3, "uuid": 2, "host": 2},
    {"version": 4, "host": 2, "ip": 2, "resource_id": 2, "path": 1},
    {"error_code": 3, "ip": 3, "url": 2, "ticket_ref": 2, "port": 1},
    {"uuid": 3, "error_code": 3, "path": 2, "datetime": 1, "version": 2},
]
MIXED_STATUS = {"current": 0.3, "historical": 0.3, "hypothetical": 0.2, "resolved": 0.2}


def _assemble(ws, schema, fam, rng, pool, example_id) -> FactoryResult:
    try:
        rendered = render_ticket(ws, fam, rng)
        roles = [r for r, _ in rendered]
        contents = [c for _, c in rendered]
        from compactex.gold import derive_facts

        final, spans = substitute_and_locate(contents, ws, pool, rng)
        if any(not msg.strip() for msg in final):
            return FactoryResult(None, "empty_message")
        ticket = Ticket(
            ticket_id=example_id,
            messages=[Message(id=f"m{i + 1}", role=roles[i], content=final[i])
                      for i in range(len(final))],
        )
        gold = derive_facts(schema, ws, spans)
        ex = TaskExample(
            id=example_id, ticket=ticket, schema=schema, gold=gold,
            meta={"pool": pool, "family": fam.name, "batch": "batch2",
                  "gold_n": len(gold), "n_mentions": len(ws.mentions)},
        )
        return FactoryResult(ex)
    except RenderAmbiguous as exc:
        return FactoryResult(None, f"render_ambiguous:{exc}")
    except Exception as exc:
        return FactoryResult(None, f"{type(exc).__name__}:{exc}")


def _scene(rng, seed, n_messages, n_mentions, kind_weights, negative_rate):
    return build_world_state(
        rng, ticket_id=f"b2-{seed:08x}", n_messages=n_messages, n_mentions=n_mentions,
        kind_weights=kind_weights, status_weights=MIXED_STATUS,
        negative_rate=negative_rate)


def make_empty_hard_negative(rng, seed, fam, bank):
    """Kind present in text, but every requested filter excludes it."""
    kinds = rng.choice(RICH_KIND_SETS)
    ws = _scene(rng, seed, rng.randint(2, 4), rng.randint(4, 8), kinds, 0.35)
    present = {m.kind for m in ws.mentions}
    # hardest: selectors whose KIND is present but whose filters exclude all mentions
    hard = [s for s in bank
            if s.kind in present and not selector_matches(s, ws)
            and (s.status != "any" or s.speaker != "any"
                 or s.in_message is not None or s.of_component is not None)]
    soft = [s for s in bank if not selector_matches(s, ws)]
    pool_sel = hard or soft
    if not pool_sel:
        return FactoryResult(None, "no_nonmatching_selector")
    first = rng.choice(pool_sel)
    fields = [build_field(first)]
    rest = [s for s in pool_sel if s != first]
    if rest and rng.random() < 0.5:
        fields.append(build_field(rng.choice(rest)))
    return _assemble(ws, TaskSchema(fields=fields), fam, rng, "train", f"b2e-{seed:08d}")


def make_near_miss(rng, seed, fam, bank):
    """3-6 same-kind candidates; a filtered selector picks a strict subset."""
    kind = rng.choice([k for k in KINDS if k != "port"])
    ws = _scene(rng, seed, rng.randint(1, 3), rng.randint(3, 6), {kind: 1.0}, 0.4)
    of_kind = [m for m in ws.mentions if m.kind == kind]
    if len(of_kind) < 2:
        return FactoryResult(None, "too_few_candidates")
    matching_filtered = [
        s for s in bank
        if s.kind == kind and 0 < len(selector_matches(s, ws)) < len(of_kind)
        and (s.status != "any" or s.speaker != "any" or s.in_message is not None
             or s.of_component is not None or s.scope != "all")
    ]
    if not matching_filtered:
        return FactoryResult(None, "no_subset_selector")
    sel = rng.choice(matching_filtered)
    fields = [build_field(sel)]
    extra = [s for s in bank if s != sel and selector_matches(s, ws)]
    if extra and rng.random() < 0.5:
        fields.append(build_field(rng.choice(extra)))
    return _assemble(ws, TaskSchema(fields=fields), fam, rng, "train", f"b2n-{seed:08d}")


def make_mixed(rng, seed, fam, bank):
    """At least one gold-producing field AND one empty field in one schema."""
    kinds = rng.choice(RICH_KIND_SETS)
    ws = _scene(rng, seed, rng.randint(2, 4), rng.randint(3, 7), kinds, 0.3)
    matching = [s for s in bank if selector_matches(s, ws)]
    non_matching = [s for s in bank if not selector_matches(s, ws)]
    if not matching or not non_matching:
        return FactoryResult(None, "no_mixed_pair")
    pos = build_field(rng.choice(matching))
    neg = build_field(rng.choice(non_matching))
    return _assemble(ws, TaskSchema(fields=[pos, neg]), fam, rng, "train",
                     f"b2m-{seed:08d}")


BUILDERS = [(make_empty_hard_negative, 0.45), (make_near_miss, 0.275), (make_mixed, 0.275)]


def main() -> None:
    import hashlib
    import json

    from compactex.families import families_for_pool

    out_dir = ROOT / "data" / "batch2"
    out_dir.mkdir(parents=True, exist_ok=True)
    fams = families_for_pool("train")
    bank = SCHEMA_BANKS["train"]()
    rng = random.Random(20261023)
    target = {"make_empty_hard_negative": 1125, "make_near_miss": 690, "make_mixed": 685}
    counts = dict.fromkeys(target, 0)
    quarantined: dict[str, int] = {}
    seen: set[str] = set()
    records = []
    seed = 700_000
    attempts = 0
    while any(counts[b.__name__] < target[b.__name__] for b, _ in BUILDERS) \
            and attempts < 60_000:
        attempts += 1
        builder = rng.choices([b for b, _ in BUILDERS],
                              weights=[target[b.__name__] - counts[b.__name__]
                                       for b, _ in BUILDERS])[0]
        fam = fams[seed % len(fams)]
        r = builder(random.Random(seed), seed, fam, bank)
        seed += 1
        if r.example is None:
            key = r.quarantine_reason.split(":")[0]
            quarantined[key] = quarantined.get(key, 0) + 1
            continue
        ex = r.example
        canon = hashlib.sha256(json.dumps(
            {"t": [m.content for m in ex.ticket.messages],
             "f": sorted(x.name for x in ex.schema_.fields)}, ensure_ascii=False)
            .encode()).hexdigest()
        if canon in seen:
            quarantined["dupe"] = quarantined.get("dupe", 0) + 1
            continue
        seen.add(canon)
        records.append(ex.model_dump(mode="json", by_alias=True))
        counts[builder.__name__] += 1
    jsonl = out_dir / "examples.jsonl"
    jsonl.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n",
        encoding="utf-8")
    empty = sum(1 for r in records if not r["gold"])
    manifest = {
        "name": "batch2_abstention",
        "n": len(records),
        "target_mix": {"empty_hard_negative": 1125, "near_miss": 690, "mixed": 685},
        "achieved": counts,
        "empty_gold_n": empty,
        "empty_gold_share": round(empty / max(len(records), 1), 3),
        "quarantined": quarantined,
        "seed_base": 700_000,
        "sha256": hashlib.sha256(jsonl.read_bytes()).hexdigest(),
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
