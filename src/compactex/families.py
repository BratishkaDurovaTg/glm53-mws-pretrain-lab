"""Template families = structural assembly patterns over the shared frame bank.

Train families differ from dev/sealed structurally (message count, ordering, noise
density, log/letter/table shapes). The frame bank is shared — family isolation is
structural at pilot scale (documented limitation, see DATA_DESIGN).
"""

import random
from dataclasses import dataclass

from compactex.frames import NOISE, OPENERS, fill, pick_frame
from compactex.worldstate import MessageSpec, WorldState


@dataclass(frozen=True)
class Family:
    name: str
    pool: str
    n_messages: tuple[int, int]
    noise_per_msg: tuple[int, int]
    opener: bool
    joiner: str
    log_style_first: bool
    langs: str  # ru | en | mixed
    bullet_lines: bool = False
    signature: bool = False
    agent_first: bool = False


FAMILIES: list[Family] = [
    # ---- train (16) ----
    Family("short_single", "train", (1, 1), (0, 1), True, " ", False, "ru"),
    Family("short_single_en", "train", (1, 1), (0, 1), True, " ", False, "en"),
    Family("short_double", "train", (2, 2), (1, 2), True, " ", False, "ru"),
    Family("dialogue_full", "train", (3, 4), (1, 2), True, "\n", False, "ru"),
    Family("dialogue_en", "train", (3, 4), (1, 2), True, "\n", False, "en"),
    Family("log_first", "train", (2, 3), (0, 1), False, "\n", True, "mixed"),
    Family("mixed_lang", "train", (2, 4), (0, 2), True, "\n", False, "mixed"),
    Family("terse", "train", (1, 2), (0, 0), False, "; ", False, "ru"),
    Family("verbose_noise", "train", (2, 3), (2, 3), True, " ", False, "ru"),
    Family("dense_multifact", "train", (1, 2), (0, 0), False, "\n", False, "mixed"),
    Family("bullets", "train", (1, 2), (0, 1), False, "\n", False, "ru", bullet_lines=True),
    Family("signature_thread", "train", (2, 3), (0, 1), True, "\n", False, "ru", signature=True),
    Family("short_mixed_en_ru", "train", (1, 2), (0, 1), True, " ", False, "mixed"),
    Family("dialog_noise_en", "train", (3, 3), (2, 3), True, "\n", False, "en"),
    Family("log_comment", "train", (2, 2), (1, 1), False, "\n", True, "ru"),
    Family("plain_flow", "train", (2, 3), (1, 1), False, " ", False, "ru"),
    # ---- dev (4): structures unseen in train ----
    Family("agent_first", "dev", (2, 3), (0, 1), True, "\n", False, "ru", agent_first=True),
    Family("table_lines", "dev", (1, 2), (0, 0), False, "\n", False, "en", bullet_lines=True),
    Family("kv_lines", "dev", (1, 1), (0, 0), False, "\n", False, "ru", bullet_lines=True),
    Family("long_letter", "dev", (1, 1), (2, 3), True, " ", False, "ru"),
    Family("echo_dialogue", "dev", (3, 4), (1, 2), True, "\n", False, "mixed"),
    # ---- sealed (4): structures unseen everywhere ----
    Family("stacked_logs", "sealed", (2, 2), (0, 0), False, "\n", True, "mixed", bullet_lines=True),
    Family("polite_formal", "sealed", (1, 2), (2, 3), True, " ", False, "ru", signature=True),
    Family("rapid_chat", "sealed", (4, 5), (0, 1), False, " ", False, "mixed"),
    Family("report_style", "sealed", (1, 1), (1, 2), False, "\n", False, "en", bullet_lines=True),
]


def families_for_pool(pool: str) -> list[Family]:
    return [f for f in FAMILIES if f.pool == pool]


def _lang_for(family: Family, rng: random.Random) -> str:
    if family.langs == "mixed":
        return rng.choice(["ru", "en"])
    return family.langs


def _render_lines(spec: MessageSpec, ws: WorldState, fam: Family, rng: random.Random) -> str:
    by_token = {m.token: m for m in ws.mentions}
    lang = _lang_for(fam, rng)
    lines: list[str] = []
    if fam.opener and spec.index == 1:
        lines.append(rng.choice(OPENERS[lang]))
    n_noise = rng.randint(*fam.noise_per_msg)
    noise = [rng.choice(NOISE[lang]) for _ in range(n_noise)]
    body: list[str] = []
    for token in spec.mention_tokens:
        mention = by_token[token]
        if fam.log_style_first and spec.index == 1:
            body.append(f"lvl=err src=api detail={token}")
        else:
            frame = pick_frame(rng, mention.kind, mention.status, lang,
                               mention.of_component is not None)
            body.append(fill(frame, token, mention.of_component))
    if fam.bullet_lines:
        body = [f"- {b}" if not fam.log_style_first else b for b in body]
    merged: list[str] = []
    for b in body:
        merged.append(b)
        if noise and rng.random() < 0.5:
            merged.append(noise.pop())
    merged.extend(noise)
    lines.extend(merged)
    if fam.signature and spec.index == len(ws.messages):
        lines.append("-- " + (rng.choice(["support", "дежурный", "ops"])) )
    return fam.joiner.join(lines)


def render_ticket(ws: WorldState, fam: Family, rng: random.Random) -> list[tuple[str, str]]:
    """Returns [(role, content)] with {TOKEN} placeholders still embedded."""
    specs = list(ws.messages)
    if fam.agent_first and len(specs) >= 2:
        specs[0], specs[1] = specs[1], specs[0]
        roles = [s.role for s in specs]
        specs = [
            MessageSpec(index=i + 1, role=roles[i], mention_tokens=s.mention_tokens)
            for i, s in enumerate(specs)
        ]
        # keep mention.message_index consistent with the new order
        remap = {i + 1: specs[i].mention_tokens for i in range(len(specs))}
        for m in ws.mentions:
            for idx, tokens in remap.items():
                if m.token in tokens:
                    m.message_index = idx
    out = [(s.role, _render_lines(s, ws, fam, rng)) for s in specs]
    return out
