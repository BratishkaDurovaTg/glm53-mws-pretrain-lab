"""End-to-end example factory: world-state → render → substitute → gold, with quarantine."""

import random
from dataclasses import dataclass

from compactex.families import Family, render_ticket
from compactex.gold import derive_facts, sample_schema
from compactex.render import RenderAmbiguous, substitute_and_locate
from compactex.schemas import Message, TaskExample, TaskSchema, Ticket
from compactex.worldstate import build_world_state

LITERAL_KINDS = ["error_code", "version", "uuid", "ip", "url", "host", "port",
                 "path", "resource_id", "ticket_ref", "datetime"]


@dataclass
class FactoryResult:
    example: TaskExample | None
    quarantine_reason: str | None = None


def build_example(
    seed: int,
    pool: str,
    family: Family,
    kind_weights: dict[str, float] | None = None,
    status_weights: dict[str, float] | None = None,
    n_mentions: tuple[int, int] = (2, 7),
    negative_rate: float = 0.3,
    empty_rate: float = 0.0,
    example_id: str | None = None,
) -> FactoryResult:
    rng = random.Random(seed)
    n_msgs = rng.randint(*family.n_messages)
    n_ment = rng.randint(*n_mentions)
    ticket_seed = f"{pool[:2]}-{seed:08x}"
    want_empty = rng.random() < empty_rate
    try:
        ws = build_world_state(
            rng, ticket_id=ticket_seed, n_messages=n_msgs, n_mentions=n_ment,
            kind_weights=kind_weights, status_weights=status_weights,
            negative_rate=negative_rate,
        )
        schema: TaskSchema | None = sample_schema(pool, rng, ws, want_match=not want_empty)
        if schema is None:
            return FactoryResult(None, "no_negative_schema")
        if not want_empty and not any(sel_matches_stub(f.selector, ws) for f in schema.fields):
            return FactoryResult(None, "no_matching_field")
        rendered = render_ticket(ws, family, rng)
        roles = [r for r, _ in rendered]
        contents = [c for _, c in rendered]
        final, spans = substitute_and_locate(contents, ws, pool, rng)
        ticket = Ticket(
            ticket_id=ticket_seed,
            messages=[Message(id=f"m{i + 1}", role=roles[i], content=final[i])
                      for i in range(len(final))],
        )
        gold = derive_facts(schema, ws, spans)
        ex = TaskExample(
            id=example_id or f"{pool}-{seed:08d}",
            ticket=ticket,
            schema=schema,
            gold=gold,
            meta={
                "pool": pool,
                "family": family.name,
                "seed": seed,
                "n_mentions": len(ws.mentions),
                "lang_style": family.langs,
                "gold_n": len(gold),
            },
        )
        return FactoryResult(ex)
    except RenderAmbiguous as exc:
        return FactoryResult(None, f"render_ambiguous:{exc}")
    except Exception as exc:  # pydantic validation, program errors → quarantine bucket
        return FactoryResult(None, f"{type(exc).__name__}:{exc}")


def sel_matches_stub(selector, ws) -> bool:
    from compactex.gold import selector_matches

    return bool(selector_matches(selector, ws))
