# Data design

## Principle

Gold is **derived mechanically from a latent world-state**, never annotated by the teacher.
The teacher (GLM) renders language around placeholders and audits ambiguity. Anything the
deterministic validator cannot confirm goes to quarantine — quality over quantity.

```text
latent world-state (structured scene)
→ task schema (named fields + natural-language definitions) chosen for that scene
→ template-family render (ru/en/mixed) with <PLACEHOLDER> tokens
→ deterministic literal substitution (per-pool)
→ gold derived mechanically from (world-state, schema, rendered text, spans)
→ span re-verification on the final string
→ (GLM critic pass: ambiguity audit — quarantines, never rewrites gold)
```

## Canonical task format

Input (chat): system = instructions + schema-as-data; user = ticket JSON. Output: facts
with exact-substring `value` + `evidence {message_id, start, end}` (Python Unicode char
offsets into that message's `content`).

## Schema space

A **schema** = 1–4 named fields, each with a natural-language definition composed from a
library of **primitive selectors**:

- literal-type selectors: error_code, version, uuid, ipv4/ipv6, url, host, port,
  path, resource_id, ticket_id, datetime, duration, quantity;
- filters: CURRENT_, HISTORICAL_, HYPOTHETICAL_, REPORTED_BY_USER, REPORTED_BY_AGENT,
  IN_MESSAGE_<k>, ASSOCIATED_WITH_<component>, LATEST_, ALL_, FIRST_MENTION;
- compositions: two selectors combined (e.g. `CURRENT_ERROR_OF_COMPONENT_X`,
  `ALL_VERSIONS_IN_MESSAGE_2`).

Field names are generated compositionally from selector grammar, e.g.
`CURRENT_ERROR_CODE`, `ALL_UUIDS`, `HISTORICAL_VERSION_OF_BILLING_API`.

**Pools:**

- `S_train` — selector combos + surface names seen in training;
- `S_dev` — unseen *names* of seen semantics + seen names, unseen combos (for DEV/DEV-TEST);
- `S_sealed` — unseen names AND unseen semantic compositions (for SEALED only);
- T2 unseen-schema suite draws only from held-out pools; naming style is matched so the
  model cannot detect held-out-ness from the string alone.

The model must read the *definition*; the name is a hint, never the rule.

## World-state

```json
{
  "ticket_id": "...",
  "messages": [{"id","role","content_template","lang"}],
  "components": ["billing-api", ...],
  "events": [{"kind":"error","code":"<ERR_A>","status":"current|resolved|hypothetical",
               "component":"<HOST_A>","reported_by":"user|agent"}],
  "versions": [{"value":"<VER_A>","component":"...","status":"current|outdated"}],
  "identifiers": [{"kind":"uuid|ip|url|path|resource","value":"<PLH>","in_message":"m2"}],
  "time_markers": ["вчера / currently / last week"],
  "corrections": [{"replaces":"<ERR_A>","with":"<ERR_B>"}]
}
```

Gold = selection over world-state objects matching the schema's selectors, mapped through
the placeholder→literal substitution table to verified spans.

## Literal pools (exactness-critical)

Each literal class has three disjoint pools (train / dev / sealed): digit spans, letter
prefixes and value grammars differ across pools so literal memorization cannot transfer.
Substitution is deterministic per seed; after substitution every evidence span is
re-located in the final string (first verified occurrence) — if a literal occurs
ambiguously (two identical values in one message), the example is quarantined.

## Template families

~24 families (roughly: short ticket, multi-message dialogue, log-snippet + comment,
agent-summary, no-fact ticket, correction thread, mixed-lang thread...), each with
ru/en/mixed surface variants and paraphrase slots. Split: 16 train / 4 dev / 4 sealed.
Families differ structurally (message count, noise style), not just lexically.

## Curriculum mix (pilot)

A literal copy 25% · B candidate disambiguation 15% · C multi-fact 15% · D empty/no-fact
hard negatives 30% · E noisy ru/mixed text 10% · F schema-defined semantics
(current/historical/by-speaker/correction) 5%+10% overlap with B/C. Hard negatives:
distractor literals that match the field's *type* but not its *definition* (docs codes,
suggestions, other services' errors).

## Splits

| split | schemas | template families | literal pools | purpose |
| --- | --- | --- | --- | --- |
| TRAIN | S_train | 16 train | train | SFT |
| DEV | S_dev-names | 4 dev | dev | checkpoint selection (extraction-F1) |
| DEV-TEST | S_dev-combos | 4 dev | dev | iteration error analysis |
| SEALED | S_sealed | 4 sealed | sealed | milestone verdicts only |

Suites: T1 literal (programmatic, all pools sealed-independent), T2 unseen-schema (≥200),
T3 technical holdout (GLM-rendered, critic-audited, human-spot-checked), T4 adversarial
(≥200). Freeze = hash manifests + generation seeds stored; no regeneration after seeing
scores. Dedup: exact + normalized + near-dup (shingle Jaccard) across ALL splits; any
cross-split hit drops both to quarantine.

## Quarantine

Triggers: ambiguous gold, multi-valid answers, unverifiable span, literal collision,
critic/validator disagreement, cross-split near-dup. Quarantined examples are counted in
the ledger (accept-rate) but never silently fixed.

## Acceptance invariants (tests)

- every gold value is an exact substring of the cited message;
- `content[start:end]` contains the value; offsets are Unicode codepoint indices;
- no two splits share templates/schemas/literals beyond allowed pools;
- empty-gold records exist in every family (≥15% of each suite);
- deterministic regeneration from seeds reproduces every suite byte-identically.
