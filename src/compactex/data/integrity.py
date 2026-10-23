"""Dataset acceptance-invariant helpers (docs/DATA_DESIGN.md, "Acceptance invariants").

Pure functions over :class:`~compactex.schemas.TaskExample` lists — no ML
dependencies — shared by the test suite and the dataset factory so both check
dedup / pool disjointness / coverage with exactly the same primitives:

- ``exact_dupes``: identical ticket content + schema field names appearing twice
  (dedup runs across ALL splits; any hit quarantines both records);
- ``near_duplicates``: word k-shingle Jaccard over normalized text;
- ``cross_pool_leak``: every gold literal must map to the split's literal pool or
  to a shared closed space (``pool_of_value`` -> ``None``);
- ``empty_gold_share``: hard-negative (empty-gold) coverage, >= 15% per suite.
"""

from __future__ import annotations

import hashlib
import json
import string
import unicodedata

from compactex.literals import POOL_NAMES, pool_of_value
from compactex.schemas import TaskExample

__all__ = [
    "ticket_content_hash",
    "exact_dupes",
    "normalized_text",
    "near_duplicates",
    "cross_pool_leak",
    "empty_gold_share",
]

_PUNCTUATION = str.maketrans("", "", string.punctuation)


def ticket_content_hash(example: TaskExample) -> str:
    """Stable content hash of one example.

    Canonical JSON of the ticket messages plus the schema field names.
    Deliberately excludes example id, ticket id, gold and meta: two records with
    the same conversation and the same schema field names are the same content,
    whatever their ids or derived gold say.
    """
    payload = {
        "messages": [
            {"id": m.id, "role": m.role, "content": m.content}
            for m in example.ticket.messages
        ],
        "schema_fields": [f.name for f in example.schema_.fields],
    }
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def exact_dupes(examples: list[TaskExample]) -> list[tuple[str, str]]:
    """All (id_a, id_b) pairs whose ticket-content hash collides.

    Each colliding group is reported pairwise, ids in input order. An empty list
    means the suite is exact-duplicate free.
    """
    groups: dict[str, list[str]] = {}
    for ex in examples:
        groups.setdefault(ticket_content_hash(ex), []).append(ex.id)
    pairs: list[tuple[str, str]] = []
    for ids in groups.values():
        if len(ids) > 1:
            pairs.extend(
                (ids[i], ids[j]) for i in range(len(ids)) for j in range(i + 1, len(ids))
            )
    return pairs


def normalized_text(s: str) -> str:
    """NFKC-fold, lowercase, strip punctuation, collapse whitespace runs to single spaces."""
    folded = unicodedata.normalize("NFKC", s).lower().translate(_PUNCTUATION)
    return " ".join(folded.split())


def _word_shingles(text: str, k: int) -> frozenset[tuple[str, ...]]:
    words = normalized_text(text).split()
    if not words:
        return frozenset()
    if len(words) < k:
        # Degenerate short text: the whole word tuple is its only shingle.
        return frozenset({tuple(words)})
    return frozenset(tuple(words[i : i + k]) for i in range(len(words) - k + 1))


def _jaccard(a: frozenset[tuple[str, ...]], b: frozenset[tuple[str, ...]]) -> float:
    if not a and not b:
        return 1.0  # two blank texts are trivially identical
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def near_duplicates(
    texts: list[str],
    shingle_k: int = 3,
    jaccard_threshold: float = 0.8,
) -> list[tuple[int, int]]:
    """Index pairs (i, j), i < j, whose word k-shingle Jaccard >= threshold.

    Texts are compared through ``normalized_text`` (case-, punctuation- and
    whitespace-insensitive). O(n^2) pairwise comparison; intended for dedup
    batches of n <= 2000.
    """
    shingles = [_word_shingles(t, shingle_k) for t in texts]
    return [
        (i, j)
        for i in range(len(texts))
        for j in range(i + 1, len(texts))
        if _jaccard(shingles[i], shingles[j]) >= jaccard_threshold
    ]


def cross_pool_leak(examples: list[TaskExample], expected_pool: str) -> list[str]:
    """Gold values whose literal-pool marker is neither shared nor the expected pool.

    For every gold fact value, ``literals.pool_of_value(value)`` must be ``None``
    (shared closed space: HTTP codes, ports, ...) or equal ``expected_pool``.
    Violations are formatted ``"<example_id>:<field>=<value> (pool <pool>)"`` so
    the quarantine ledger can cite the exact offending fact.
    """
    if expected_pool not in POOL_NAMES:
        raise ValueError(f"unknown pool {expected_pool!r}; expected one of {POOL_NAMES}")
    violations: list[str] = []
    for ex in examples:
        for fact in ex.gold:
            pool = pool_of_value(fact.value)
            if pool is not None and pool != expected_pool:
                violations.append(f"{ex.id}:{fact.type}={fact.value} (pool {pool})")
    return violations


def empty_gold_share(examples: list[TaskExample]) -> float:
    """Fraction of records with no gold facts (hard negatives). 0.0 for an empty suite."""
    if not examples:
        return 0.0
    return sum(1 for ex in examples if not ex.gold) / len(examples)
