"""Placeholder → literal substitution and mechanical span location."""

import random
import re

from compactex.literals import GENERATORS
from compactex.worldstate import WorldState


class RenderAmbiguous(Exception):
    """Literal collision or lost span — caller must quarantine the example."""


def substitute_and_locate(
    contents: list[str], ws: WorldState, pool: str, rng: random.Random
) -> tuple[list[str], dict[str, tuple[int, str, int, int]]]:
    """Replace every mention token with a pool literal; locate first span per token.

    Returns (final_contents, {token: (message_index(1-based), value, start, end)}).
    """
    values: dict[str, str] = {}
    used_per_message: dict[int, set[str]] = {}
    for mention in ws.mentions:
        gen = GENERATORS[mention.kind]
        for _ in range(5):
            candidate = gen(pool, rng)
            if candidate not in used_per_message.get(mention.message_index, set()):
                break
        else:
            raise RenderAmbiguous(f"literal collision for {mention.token}")
        used_per_message.setdefault(mention.message_index, set()).add(candidate)
        values[mention.token] = candidate

    final: list[str] = []
    spans: dict[str, tuple[int, str, int, int]] = {}
    # longest tokens first: ERR_11 must not be corrupted by replacing ERR_1
    ordered = sorted(values.items(), key=lambda kv: -len(kv[0]))
    for i, content in enumerate(contents, start=1):  # noqa: B007
        text = content
        for token, value in ordered:
            text = text.replace(token, value)
        final.append(text)
    for mention in ws.mentions:
        value = values[mention.token]
        text = final[mention.message_index - 1]
        hits = [m.start() for m in re.finditer(re.escape(value), text)]
        if not hits:
            raise RenderAmbiguous(f"value {value} not found after substitution")
        start = hits[0]
        spans[mention.token] = (mention.message_index, value, start, start + len(value))
    # placeholder leakage check: no leftover tokens like ERR_9 in final text
    joined = "\n".join(final)
    leftover = re.search(r"\b(?:ERR|VER|UID|IP|URL|HST|PRT|PTH|RES|TKT|DTE|AMT|PHN|ACC|ORD|APP|TXN|TAR|SRV|DEV)_\d+\b", joined)
    if leftover:
        raise RenderAmbiguous(f"unsubstituted token {leftover.group(0)!r}")
    return final, spans
