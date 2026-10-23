"""Acceptance-invariant tests for dataset construction (docs/DATA_DESIGN.md).

Exercises the pure invariant helpers in ``compactex.data.integrity`` — the same
primitives the future dataset factory imports — on small synthetic
``TaskExample`` fixtures, plus the ``pool_of_value`` marker rules from
``compactex.literals`` that cross-pool leak detection is built on.

Note on IP literals: ``pool_of_value``'s version pattern also matches IPv4
strings, so an IP whose first octet escapes every version-major range (dev
octets 100-169, sealed 170-219) raises StopIteration instead of resolving to a
pool. IPs are therefore represented here only by first octet 10, which resolves
correctly; dev/sealed IPs are avoided until literals.py is fixed.
"""

from __future__ import annotations

import zlib
from collections.abc import Sequence

import pytest

from compactex.data.integrity import (
    cross_pool_leak,
    empty_gold_share,
    exact_dupes,
    near_duplicates,
    normalized_text,
)
from compactex.literals import pool_of_value
from compactex.schemas import (
    Evidence,
    Fact,
    Message,
    SchemaField,
    Selector,
    TaskExample,
    TaskSchema,
    Ticket,
)

# ---------------------------------------------------------------------------
# Synthetic fixture builder
# ---------------------------------------------------------------------------


def _ticket_id(seed: str) -> str:
    """Deterministic valid ticket id (REQ- prefix = dev pool)."""
    return f"REQ-{100000 + zlib.crc32(seed.encode()) % 900000}"


def make_example(
    ex_id: str,
    content: str,
    facts: Sequence[tuple[str, str]] = (),
    *,
    ticket_id: str | None = None,
    field_names: Sequence[str] | None = None,
    definitions: dict[str, str] | None = None,
) -> TaskExample:
    """Build a minimal valid TaskExample: one user message, exact-span gold.

    ``facts`` are ``(schema field name, value)`` pairs; each value must be an
    exact substring of ``content``. Schema fields default to the fact names
    (plus CURRENT_ERROR_CODE for empty-gold records).
    """
    definitions = definitions or {}
    if field_names is None:
        field_names = [name for name, _ in facts] or ["CURRENT_ERROR_CODE"]
    fields = [
        SchemaField(
            name=name,
            definition=definitions.get(name, f"Extraction rule for field {name}."),
            selector=Selector(kind="error_code"),
        )
        for name in dict.fromkeys(field_names)
    ]
    gold: list[Fact] = []
    for name, value in facts:
        start = content.index(value)
        gold.append(
            Fact(
                type=name,
                value=value,
                evidence=Evidence(message_id="m1", start=start, end=start + len(value)),
            )
        )
    ticket = Ticket(
        ticket_id=ticket_id or _ticket_id(ex_id),
        messages=[Message(id="m1", role="user", content=content)],
    )
    return TaskExample(id=ex_id, ticket=ticket, schema=TaskSchema(fields=fields), gold=gold)


# A three-sentence incident report and its shuffled / noisy / unrelated variants.
INCIDENT = (
    "Our billing gateway started failing yesterday and every invoice request now "
    "times out completely. The cache cluster in the frankfurt region keeps dropping "
    "connections during peak evening traffic windows. Queue consumers report repeated "
    "delivery stalls whenever the scheduler triggers its nightly reconciliation job."
)

INCIDENT_SHUFFLED = (
    "Queue consumers report repeated delivery stalls whenever the scheduler triggers "
    "its nightly reconciliation job! Our billing gateway started failing yesterday "
    "and every invoice request now times out completely? The cache cluster in the "
    "frankfurt region keeps dropping connections during peak evening traffic windows."
)

INCIDENT_NOISY = INCIDENT.upper().replace(".", " ... ")

UNRELATED = (
    "Please escalate this incident to the on-call engineer because customer data "
    "replication is lagging behind."
)

RELEASE_NOTE = "Totally different: we shipped the new release train early this morning."

# A dev-pool example: named error code, host, date and ticket ref from dev marker
# ranges, plus shared HTTP code and port (pool None).
DEV_CONTENT = (
    "Error KQR42-318 hit delivery-dev-03 on 2026-06-15; the tracker returned 404 on "
    "port 8443. Follow-up is REQ-100482."
)
DEV_FACTS = [
    ("CURRENT_ERROR_CODE", "KQR42-318"),
    ("AFFECTED_HOST", "delivery-dev-03"),
    ("INCIDENT_DATE", "2026-06-15"),
    ("HTTP_STATUS", "404"),
    ("AFFECTED_PORT", "8443"),
    ("TICKET_REFERENCE", "REQ-100482"),
]


# ---------------------------------------------------------------------------
# Pool marker rules (literals.pool_of_value) — the contract leak checks rely on
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "pool"),
    [
        # Named error codes [A-Z]{3}\d{2}-\d{3}: first letter picks the pool.
        ("ABC12-345", "train"),
        ("KQR42-318", "dev"),
        ("STZ88-777", "sealed"),
        # Versions: major in 1-11 / 12-27 / 28-47, optional v prefix and suffix.
        ("v3.2.1", "train"),
        ("14.2.6", "dev"),
        ("14.2.6-rc1", "dev"),
        ("31.0.0", "sealed"),
        # UUIDs: first hex char in 0-5 / 6-9ab / c-f.
        ("0a1b2c3d-1111-2222-3333-444444444444", "train"),
        ("7a1b2c3d-1111-2222-3333-444444444444", "dev"),
        ("c0a1b2c3-1111-2222-3333-444444444444", "sealed"),
        # Ticket refs: INC / REQ / OPS prefixes.
        ("INC-223344", "train"),
        ("REQ-100482", "dev"),
        ("OPS-445566", "sealed"),
        # Hosts: leading word (billing.../delivery.../replica...), optional -NN index.
        ("billing-prod", "train"),
        ("delivery-dev", "dev"),
        ("delivery-dev-03", "dev"),
        ("replica-core-11", "sealed"),
        # Datetimes 2026-MM-DD: month in 1-4 / 5-8 / 9-12.
        ("2026-02-15", "train"),
        ("2026-06-15", "dev"),
        ("2026-11-30", "sealed"),
        # IP first octet 10 falls in the train IP range and, parsed as version
        # major 10, still resolves to train (see module docstring for the quirk).
        ("10.0.0.5", "train"),
        # Shared closed spaces -> None: no leak is possible.
        ("404", None),
        ("429", None),
        ("8443", None),
        ("27017", None),
        # Grammars pool_of_value cannot see -> None (documented behavior).
        ("https://delivery-dev-03/ab12f", None),
        ("/mnt/seg42", None),
    ],
)
def test_pool_of_value_markers(value: str, pool: str | None) -> None:
    assert pool_of_value(value) == pool


# ---------------------------------------------------------------------------
# normalized_text
# ---------------------------------------------------------------------------


def test_normalized_text_folds_case_punctuation_and_whitespace() -> None:
    assert (
        normalized_text("  Error,  KQR42-318!!  hit   delivery-dev-03. ")
        == "error kqr42318 hit deliverydev03"
    )


def test_normalized_text_is_idempotent() -> None:
    once = normalized_text("Version  14.2.6-rc1 -- ROLLED BACK, on 2026-06-15!!")
    assert normalized_text(once) == once


# ---------------------------------------------------------------------------
# exact_dupes
# ---------------------------------------------------------------------------


def test_exact_dupes_catches_identical_tickets() -> None:
    content = "Error KQR42-318 on delivery-dev-03 since yesterday morning."
    a = make_example("a", content, [("CURRENT_ERROR_CODE", "KQR42-318")])
    b = make_example(
        "b", content, [("CURRENT_ERROR_CODE", "KQR42-318")], ticket_id="REQ-999999"
    )
    assert exact_dupes([a, b]) == [("a", "b")]


def test_exact_dupes_ignore_ids_gold_definitions_but_not_content_or_fields() -> None:
    content = "Error KQR42-318 logged; follow-up REQ-100482 tracks the rollback."
    fields = ["CURRENT_ERROR_CODE", "TICKET_REFERENCE"]
    a = make_example(
        "a", content, [("CURRENT_ERROR_CODE", "KQR42-318")], field_names=fields
    )
    # Same messages + same field names: different id, ticket id, gold and field
    # definitions are all irrelevant to content identity.
    b = make_example(
        "b",
        content,
        [("TICKET_REFERENCE", "REQ-100482")],
        field_names=fields,
        definitions={"CURRENT_ERROR_CODE": "Different wording, same field name."},
        ticket_id="REQ-888888",
    )
    assert exact_dupes([a, b]) == [("a", "b")]

    other_content = make_example(
        "c",
        "Error KQR99-111 logged; follow-up REQ-100482 tracks the rollback.",
        [("CURRENT_ERROR_CODE", "KQR99-111")],
        field_names=fields,
    )
    other_fields = make_example(
        "d",
        content,
        [("ACTIVE_ERROR_CODE", "KQR42-318")],
        field_names=["ACTIVE_ERROR_CODE", "TICKET_REFERENCE"],
    )
    assert exact_dupes([a, b, other_content, other_fields]) == [("a", "b")]


def test_exact_dupes_reports_every_pair_in_a_collision_group() -> None:
    content = "Error KQR42-318 on delivery-dev-03 since yesterday morning."
    x1 = make_example("x1", content)
    x2 = make_example("x2", content, ticket_id="REQ-222222")
    x3 = make_example("x3", content, ticket_id="REQ-333333")
    assert exact_dupes([x1, x2, x3]) == [
        ("x1", "x2"),
        ("x1", "x3"),
        ("x2", "x3"),
    ]


def test_exact_dupes_empty_input() -> None:
    assert exact_dupes([]) == []


# ---------------------------------------------------------------------------
# near_duplicates
# ---------------------------------------------------------------------------


def test_near_duplicates_catches_shuffled_sentences_but_not_other_texts() -> None:
    assert near_duplicates([INCIDENT, INCIDENT_SHUFFLED, UNRELATED]) == [(0, 1)]


def test_near_duplicates_ignores_case_and_punctuation() -> None:
    assert near_duplicates([INCIDENT, INCIDENT_NOISY]) == [(0, 1)]


def test_near_duplicates_flags_identical_texts() -> None:
    assert near_duplicates([UNRELATED, UNRELATED]) == [(0, 1)]


def test_near_duplicates_no_false_pairs_for_distinct_content() -> None:
    assert near_duplicates([INCIDENT, UNRELATED, RELEASE_NOTE]) == []


def test_near_duplicates_threshold_gates_the_shuffled_pair() -> None:
    texts = [INCIDENT, INCIDENT_SHUFFLED]
    # Jaccard is ~0.9: below a strict threshold, comfortably above a loose one.
    assert near_duplicates(texts, jaccard_threshold=0.95) == []
    assert near_duplicates(texts, jaccard_threshold=0.5) == [(0, 1)]


# ---------------------------------------------------------------------------
# cross_pool_leak
# ---------------------------------------------------------------------------


def test_cross_pool_leak_passes_correct_pool_and_shared_spaces() -> None:
    ex = make_example("dev-1", DEV_CONTENT, DEV_FACTS)
    assert cross_pool_leak([ex], "dev") == []


def test_cross_pool_leak_flags_train_literals_in_dev_example() -> None:
    content = "The billing-prod host reported ABC12-345 and INC-223344 was opened."
    ex = make_example(
        "leak-1",
        content,
        [
            ("CURRENT_ERROR_CODE", "ABC12-345"),
            ("AFFECTED_HOST", "billing-prod"),
            ("TICKET_REFERENCE", "INC-223344"),
        ],
    )
    violations = cross_pool_leak([ex], "dev")
    assert len(violations) == 3
    assert all(v.startswith("leak-1:") for v in violations)
    for value in ("ABC12-345", "billing-prod", "INC-223344"):
        assert any(value in v for v in violations)
    assert all("(pool train)" in v for v in violations)


def test_cross_pool_leak_reports_only_offending_examples() -> None:
    clean = make_example("dev-1", DEV_CONTENT, DEV_FACTS)
    leak = make_example(
        "leak-1",
        "The billing-prod host reported ABC12-345 and INC-223344 was opened.",
        [("CURRENT_ERROR_CODE", "ABC12-345"), ("AFFECTED_HOST", "billing-prod"),
         ("TICKET_REFERENCE", "INC-223344")],
    )
    assert cross_pool_leak([clean, leak], "dev") == cross_pool_leak([leak], "dev")


def test_cross_pool_leak_flags_sealed_literals_in_train_suite() -> None:
    content = "Escalated as OPS-445566 against replica-core-11 earlier today."
    ex = make_example(
        "train-1",
        content,
        [("TICKET_REFERENCE", "OPS-445566"), ("AFFECTED_HOST", "replica-core-11")],
    )
    violations = cross_pool_leak([ex], "train")
    assert len(violations) == 2
    assert all("(pool sealed)" in v for v in violations)


def test_cross_pool_leak_rejects_unknown_pool() -> None:
    with pytest.raises(ValueError, match="unknown pool"):
        cross_pool_leak([], "prod")


# ---------------------------------------------------------------------------
# empty_gold_share
# ---------------------------------------------------------------------------


def test_empty_gold_share_counts_records_without_facts() -> None:
    empty_a = make_example("e1", "Nothing extractable here, thanks for the quick update.")
    filled_a = make_example("f1", DEV_CONTENT, DEV_FACTS)
    empty_b = make_example("e2", "User just said hello and closed the chat politely.")
    filled_b = make_example(
        "f2", "Cache miss on queue-3f2a9c81 noted.", [("AFFECTED_RESOURCE", "queue-3f2a9c81")]
    )
    filled_c = make_example(
        "f3", "Version 14.2.6-rc1 rolled back today.", [("CURRENT_VERSION", "14.2.6-rc1")]
    )
    assert empty_gold_share([empty_a, filled_a, empty_b, filled_b, filled_c]) == pytest.approx(0.4)


def test_empty_gold_share_boundaries() -> None:
    a = make_example("a", "Hello world.")
    b = make_example("b", "Thanks, closing the ticket.")
    c = make_example("c", DEV_CONTENT, DEV_FACTS)
    assert empty_gold_share([a, b]) == 1.0
    assert empty_gold_share([c]) == 0.0
    assert empty_gold_share([]) == 0.0


# ---------------------------------------------------------------------------
# Combined acceptance invariants on a synthetic dev suite
# ---------------------------------------------------------------------------


def test_synthetic_dev_suite_satisfies_acceptance_invariants() -> None:
    suite = [
        make_example("dev-1", DEV_CONTENT, DEV_FACTS),
        make_example(
            "dev-2",
            "Version 14.2.6-rc1 on delivery-dev was rolled back on 2026-07-02.",
            [
                ("CURRENT_VERSION", "14.2.6-rc1"),
                ("AFFECTED_HOST", "delivery-dev"),
                ("INCIDENT_DATE", "2026-07-02"),
            ],
        ),
        make_example(
            "dev-3",
            "Scheduler crawler-test-04 returned 503 on port 9090; tracker assigned "
            "REQ-204815 overnight.",
            [
                ("AFFECTED_HOST", "crawler-test-04"),
                ("HTTP_STATUS", "503"),
                ("AFFECTED_PORT", "9090"),
                ("TICKET_REFERENCE", "REQ-204815"),
            ],
        ),
        make_example("dev-4", "Nothing to extract from this polite conversation."),
        make_example("dev-5", "No facts here, just a thank-you message and a goodbye."),
    ]
    assert exact_dupes(suite) == []
    assert near_duplicates([ex.ticket.messages[0].content for ex in suite]) == []
    assert cross_pool_leak(suite, "dev") == []
    share = empty_gold_share(suite)
    assert share == pytest.approx(0.4)
    assert share >= 0.15  # DATA_DESIGN: empty-gold records >= 15% of each suite
