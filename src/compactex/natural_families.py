"""Natural structural families (Exp3): messy surfaces over the natural frame bank.

Decor styles: plain chat, logs-in-prose, terminal, stack-trace, config paste,
incident report, terse, verbose, quoted history. Typo-noise corrupts only the
frame text (applied BEFORE literal substitution), never the literals.
Single-variable principle: same world-state machinery, same statuses, new surface.
"""

import random
import re
from dataclasses import dataclass

from compactex.frames import fill
from compactex.natural_frames import BANK, NOISE_NATURAL
from compactex.worldstate import WorldState


@dataclass(frozen=True)
class NaturalFamily:
    name: str
    pool: str  # train only for Exp3
    n_messages: tuple[int, int]
    noise_per_msg: tuple[int, int]
    decor: str  # plain | log | term | stack | config | report | quote
    langs: str  # ru | en | mixed
    typo_rate: float = 0.35


NATURAL_FAMILIES: list[NaturalFamily] = [
    NaturalFamily("casual_chat", "train", (1, 2), (0, 2), "plain", "ru"),
    NaturalFamily("casual_chat_long", "train", (2, 4), (1, 3), "plain", "ru", 0.5),
    NaturalFamily("eng_slack", "train", (1, 3), (0, 2), "plain", "en", 0.25),
    NaturalFamily("ruen_mix", "train", (1, 3), (0, 2), "plain", "mixed"),
    NaturalFamily("logs_in_prose", "train", (2, 3), (0, 1), "log", "ru", 0.15),
    NaturalFamily("logs_in_prose_en", "train", (2, 3), (0, 1), "log", "en", 0.15),
    NaturalFamily("terminal_user", "train", (1, 2), (0, 1), "term", "mixed", 0.1),
    NaturalFamily("stack_trace", "train", (1, 2), (0, 1), "stack", "mixed", 0.1),
    NaturalFamily("config_paste", "train", (1, 2), (0, 1), "config", "mixed", 0.1),
    NaturalFamily("incident_report", "train", (1, 1), (1, 2), "report", "ru", 0.1),
    NaturalFamily("incident_report_en", "train", (1, 1), (1, 2), "report", "en", 0.1),
    NaturalFamily("terse_msg", "train", (1, 2), (0, 0), "plain", "mixed", 0.5),
    NaturalFamily("verbose_noisy", "train", (1, 3), (2, 3), "plain", "ru", 0.45),
    NaturalFamily("quoted_history", "train", (2, 3), (0, 1), "quote", "ru", 0.2),
]


def _lang(fam: NaturalFamily, rng: random.Random) -> str:
    return rng.choice(["ru", "en"]) if fam.langs == "mixed" else fam.langs


def _mess(text: str, rng: random.Random) -> str:
    """Controlled surface corruption on frame text only ({T}/{C} intact)."""
    if rng.random() < 0.45:
        text = text.lower()
    if rng.random() < 0.3:
        text = re.sub(r"([.!,?;])", "", text, count=1)
    if rng.random() < 0.2:
        pos = rng.randint(0, max(len(text) - 1, 0))
        text = text[:pos] + " " + text[pos:]
    if rng.random() < 0.15:
        text = text.replace(" ", "  ", 1)
    return text.strip()


def _pick_frame(kind: str, status: str, lang: str, rng: random.Random,
                has_component: bool) -> str:
    entry = BANK.get((kind, status)) or BANK[("_any", status)]

    def usable_from(e: dict) -> list[str]:
        bank = e.get(lang) or e.get("ru") or []
        return [f for f in bank if has_component or "{C}" not in f]

    usable = usable_from(entry) or usable_from(BANK[("_any", status)])
    if not usable:  # last resort: any frame regardless of {C}
        usable = (entry.get(lang) or entry.get("ru") or [])[:]
    return rng.choice(usable)


def _decorate(decor: str, frame_text: str, token: str, rng: random.Random) -> str:
    if decor == "log":
        return f"lvl=err src=api detail={token}  # {frame_text}"
    if decor == "term":
        return f"$ check --id {token}\n{frame_text}"
    if decor == "stack":
        return f"\tat handler({token})\n{frame_text}"
    if decor == "config":
        return f"  value: {token}\n{frame_text}"
    if decor == "quote":
        return f"> {frame_text}"
    return frame_text


def render_natural(ws: WorldState, fam: NaturalFamily, rng: random.Random) -> list[tuple[str, str]]:
    by_token = {m.token: m for m in ws.mentions}
    out: list[tuple[str, str]] = []
    for spec in ws.messages:  # all messages: world-state sized to family beforehand
        lang = _lang(fam, rng)
        lines: list[str] = []
        for token in spec.mention_tokens:
            mention = by_token[token]
            frame = _pick_frame(mention.kind, mention.status, lang, rng,
                                mention.of_component is not None)
            if fam.typo_rate and rng.random() < fam.typo_rate:
                frame = _mess(frame, rng)
            filled = fill(frame, token, mention.of_component)
            lines.append(_decorate(fam.decor, filled, token, rng))
        n_noise = rng.randint(*fam.noise_per_msg)
        noise = [rng.choice(NOISE_NATURAL[lang]) for _ in range(n_noise)]
        merged = lines + noise
        rng.shuffle(merged)
        content = "\n".join(merged) if fam.decor in ("log", "stack", "config", "report") \
            else " ".join(merged)
        out.append((spec.role, content if content.strip() else "проверьте пожалуйста"))
    return out
