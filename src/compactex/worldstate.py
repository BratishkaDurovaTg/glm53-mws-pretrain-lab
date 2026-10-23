"""Latent world-state: the ground truth from which gold is derived mechanically."""

import random
from typing import Literal

from pydantic import Field, model_validator

from compactex.schemas import LiteralKind, SpeakerFilter, StatusFilter, StrictModel

MentionStatus = Literal["current", "historical", "hypothetical", "resolved", "mentioned_only"]

COMPONENT_WORDS = ["billing", "auth", "search", "delivery", "storage", "media", "payments", "sso"]

KINDS: list[LiteralKind] = [
    "error_code", "version", "uuid", "ip", "url", "host",
    "port", "path", "resource_id", "ticket_ref", "datetime",
]

# statuses an actual event can have; "mentioned_only" = literal present but no event asserted
EVENT_STATUSES: list[MentionStatus] = ["current", "historical", "hypothetical", "resolved"]


class Mention(StrictModel):
    token: str  # placeholder id, e.g. ERR_A; unique per scene
    kind: LiteralKind
    status: MentionStatus
    message_index: int = Field(ge=1)  # 1-based
    reported_by: SpeakerFilter  # role of the message it lives in
    of_component: str | None = None  # COMPONENT_WORDS entry, plain word in text


class MessageSpec(StrictModel):
    index: int  # 1-based
    role: SpeakerFilter
    mention_tokens: list[str] = Field(default_factory=list)


class WorldState(StrictModel):
    ticket_id: str
    messages: list[MessageSpec]
    mentions: list[Mention]

    @model_validator(mode="after")
    def consistent(self):
        idx = {m.index for m in self.messages}
        if idx != set(range(1, len(self.messages) + 1)):
            raise ValueError("message indices must be 1..N")
        tokens = [m.token for m in self.mentions]
        if len(tokens) != len(set(tokens)):
            raise ValueError("duplicate mention token")
        by_token = {m.token: m for m in self.mentions}
        for msg in self.messages:
            for t in msg.mention_tokens:
                if t not in by_token:
                    raise ValueError(f"unknown mention token {t}")
                if by_token[t].message_index != msg.index:
                    raise ValueError(f"token {t} listed in wrong message")
        return self


def _token(kind: LiteralKind, seq: int) -> str:
    from compactex.literals import PLACEHOLDER_BY_KIND

    return f"{PLACEHOLDER_BY_KIND[kind]}_{seq}"


def build_world_state(
    rng: random.Random,
    ticket_id: str,
    n_messages: int,
    n_mentions: int,
    kind_weights: dict[str, float] | None = None,
    status_weights: dict[str, float] | None = None,
    negative_rate: float = 0.25,
) -> WorldState:
    """Generate one scene. Distractor (mentioned_only) mentions carry `negative_rate` share."""
    roles: list[SpeakerFilter] = []
    for i in range(n_messages):
        if i == 0:
            roles.append("user")
        elif i == 1:
            roles.append("agent")
        else:
            roles.append(rng.choice(["user", "agent"]))
    messages = [MessageSpec(index=i + 1, role=r, mention_tokens=[]) for i, r in enumerate(roles)]

    kind_w = kind_weights or {k: 1.0 for k in KINDS}
    event_w = status_weights or {
        "current": 0.4, "historical": 0.2, "hypothetical": 0.1, "resolved": 0.05,
    }
    event_total = sum(event_w.values()) or 1.0
    status_w = {k: v / event_total * (1.0 - negative_rate) for k, v in event_w.items()}
    status_w["mentioned_only"] = negative_rate
    mention_status_pool: list[StatusFilter] = EVENT_STATUSES + ["mentioned_only"]
    mentions: list[Mention] = []
    used_seq: dict[str, int] = {}
    for _ in range(n_mentions):
        kind = rng.choices(list(kind_w), weights=list(kind_w.values()))[0]
        used_seq[kind] = used_seq.get(kind, 0) + 1
        token = _token(kind, used_seq[kind])
        status = rng.choices(
            mention_status_pool,
            weights=[status_w.get(s, 0.05) for s in mention_status_pool],
        )[0]
        msg_i = rng.randint(1, n_messages)
        of_component = rng.choice(COMPONENT_WORDS) if rng.random() < 0.35 else None
        mention = Mention(
            token=token,
            kind=kind,
            status=status,
            message_index=msg_i,
            reported_by=messages[msg_i - 1].role,
            of_component=of_component,
        )
        mentions.append(mention)
        messages[msg_i - 1].mention_tokens.append(token)
    return WorldState(ticket_id=ticket_id, messages=messages, mentions=mentions)
