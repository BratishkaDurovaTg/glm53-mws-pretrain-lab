"""Selector application (mechanical gold) and schema banks with pool separation."""

import random

from compactex.schemas import (
    Evidence,
    Fact,
    SchemaField,
    Selector,
    TaskSchema,
)
from compactex.worldstate import COMPONENT_WORDS, WorldState

KIND_NAME = {
    "error_code": "ERROR_CODE",
    "version": "VERSION",
    "uuid": "UUID",
    "ip": "IP_ADDRESS",
    "url": "URL",
    "host": "HOSTNAME",
    "port": "PORT",
    "path": "FILE_PATH",
    "resource_id": "RESOURCE_ID",
    "ticket_ref": "TICKET_REF",
    "datetime": "DATE",
}

KIND_PLURAL = {
    "error_code": "error codes",
    "version": "version strings",
    "uuid": "UUID values",
    "ip": "IP addresses",
    "url": "URLs",
    "host": "hostnames",
    "port": "port numbers",
    "path": "file paths",
    "resource_id": "resource identifiers",
    "ticket_ref": "ticket references",
    "datetime": "dates",
}

STATUS_MODIFIER = {
    "any": ("ALL", "all {plural} mentioned anywhere in the conversation"),
    "current": ("CURRENT", "{plural} explicitly described as occurring at the present moment"),
    "historical": ("HISTORICAL", "{plural} explicitly described as past, earlier or already fixed"),
    "hypothetical": ("POSSIBLE", "{plural} described as suspected, possible or suggested, not confirmed"),
    "resolved": ("RESOLVED", "{plural} explicitly described as already fixed"),
}


def selector_matches(sel: Selector, ws: WorldState) -> list[str]:
    """Return matching mention tokens, ordered by textual position."""
    pos: dict[str, tuple[int, int]] = {}
    for spec in ws.messages:
        for i, token in enumerate(spec.mention_tokens):
            pos[token] = (spec.index, i)
    out = []
    for mention in ws.mentions:
        if mention.kind != sel.kind:
            continue
        if sel.status == "current" and mention.status != "current":
            continue
        if sel.status == "historical" and mention.status not in ("historical", "resolved"):
            continue
        if sel.status == "hypothetical" and mention.status != "hypothetical":
            continue
        if sel.status == "resolved" and mention.status != "resolved":
            continue
        if sel.speaker != "any" and mention.reported_by != sel.speaker:
            continue
        if sel.in_message is not None and mention.message_index != sel.in_message:
            continue
        if sel.of_component is not None and mention.of_component != sel.of_component:
            continue
        out.append(mention.token)
    out.sort(key=lambda t: pos[t])
    if sel.scope == "first" and out:
        out = out[:1]
    if sel.scope == "latest" and out:
        out = out[-1:]
    return out


def build_field(sel: Selector) -> SchemaField:
    mod, defn = STATUS_MODIFIER[sel.status]
    name = f"{mod}_{KIND_NAME[sel.kind]}"
    definition = defn.format(plural=KIND_PLURAL[sel.kind])
    if sel.speaker == "user":
        name += "_REPORTED_BY_USER"
        definition += ", as reported by the user (not the support agent)"
    elif sel.speaker == "agent":
        name += "_REPORTED_BY_AGENT"
        definition += ", as reported by the support agent"
    if sel.in_message is not None:
        name += f"_IN_MESSAGE_{sel.in_message}"
        definition += f", appearing in message {sel.in_message} of the conversation"
    if sel.of_component is not None:
        name += f"_OF_{sel.of_component.upper()}"
        definition += f", associated with the {sel.of_component} service"
    if sel.scope == "first":
        name = "FIRST_" + name
        definition = "the first mentioned one of: " + definition
    elif sel.scope == "latest":
        name = "LATEST_" + name
        definition = "the last mentioned one of: " + definition
    return SchemaField(name=name, definition=definition, selector=sel)


def derive_facts(schema: TaskSchema, ws: WorldState,
                 spans: dict[str, tuple[int, str, int, int]]) -> list[Fact]:
    facts: list[Fact] = []
    for field in schema.fields:
        for token in selector_matches(field.selector, ws):
            msg_i, value, start, end = spans[token]
            facts.append(Fact(
                type=field.name,
                value=value,
                evidence=Evidence(message_id=f"m{msg_i}", start=start, end=end),
            ))
    return facts


# ---------------------------------------------------------------------------
# Schema banks. train semantics/combos seen in SFT; dev adds unseen names of
# seen semantics plus a few unseen combos; sealed adds unseen compositions.
# ---------------------------------------------------------------------------

_TRAIN_KINDS = ["error_code", "version", "uuid", "ip", "url", "host", "path", "resource_id",
                "ticket_ref", "datetime"]
_TRAIN_STATUSES = ["any", "current", "historical", "hypothetical"]


def _bank_train() -> list[Selector]:
    sels: list[Selector] = []
    for kind in _TRAIN_KINDS:
        for status in _TRAIN_STATUSES:
            sels.append(Selector(kind=kind, status=status))
    sels += [
        Selector(kind="error_code", status="current", speaker="user"),
        Selector(kind="error_code", status="current", speaker="agent"),
        Selector(kind="host", status="current", of_component=None, scope="latest"),
        Selector(kind="version", status="any", scope="first"),
        Selector(kind="version", status="any", scope="latest"),
        Selector(kind="error_code", status="any", in_message=1),
        Selector(kind="error_code", status="any", in_message=2),
        Selector(kind="uuid", status="current", in_message=2),
    ]
    return sels


def _bank_dev() -> list[Selector]:
    # unseen names-of-seen-semantics (same selector shapes, rendered by different
    # example ids) + unseen kind/status combos not trained on
    return [
        Selector(kind="version", status="current", speaker="user"),
        Selector(kind="version", status="current", speaker="agent"),
        Selector(kind="uuid", status="any", scope="first"),
        Selector(kind="url", status="any", in_message=2),
        Selector(kind="ip", status="historical"),
        Selector(kind="port", status="any"),            # kind unseen in train banks
        Selector(kind="error_code", status="any", in_message=3),
        Selector(kind="host", status="any", scope="first"),
        Selector(kind="resource_id", status="hypothetical"),
        Selector(kind="datetime", status="current", speaker="user"),
    ]


def _bank_sealed() -> list[Selector]:
    # unseen semantic compositions: component-conditioned + scope + message mixes
    out: list[Selector] = []
    for comp in COMPONENT_WORDS[:4]:
        out.append(Selector(kind="error_code", status="current", of_component=comp))
    out += [
        Selector(kind="version", status="historical", of_component=COMPONENT_WORDS[4]),
        Selector(kind="host", status="current", of_component=COMPONENT_WORDS[5], scope="latest"),
        Selector(kind="error_code", status="historical", speaker="user", scope="first"),
        Selector(kind="uuid", status="any", in_message=1, scope="latest"),
        Selector(kind="path", status="current", speaker="agent"),
        Selector(kind="port", status="current", of_component=COMPONENT_WORDS[6]),
        Selector(kind="datetime", status="historical", scope="first"),
    ]
    return out


SCHEMA_BANKS = {"train": _bank_train, "dev": _bank_dev, "sealed": _bank_sealed}


def sample_schema(pool: str, rng: random.Random, ws: WorldState, max_fields: int = 3,
                  want_match: bool = True) -> TaskSchema | None:
    """Pick 1..max_fields selectors. want_match=True guarantees >=1 gold-producing field.
    want_match=False returns a schema with zero matches (empty-gold hard negative),
    or None when every bank selector matches this scene."""
    bank = SCHEMA_BANKS[pool]()
    rng.shuffle(bank)
    matching = {id(s) for s in bank if selector_matches(s, ws)}
    non_matching = [s for s in bank if id(s) not in matching]
    if want_match:
        candidates = [s for s in bank if id(s) in matching]
        fallback = [s for s in bank if id(s) not in matching]
        if not candidates:
            candidates, fallback = fallback, []
        first, rest = candidates[0], candidates[1:] + fallback
    else:
        if not non_matching:
            return None
        first, rest = non_matching[0], non_matching[1:]
    fields = [build_field(first)]
    limit = rng.randint(1, max_fields)
    for sel in rest:
        if len(fields) >= limit:
            break
        if rng.random() < 0.5:
            fields.append(build_field(sel))
    return TaskSchema(fields=fields)
