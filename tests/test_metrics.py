"""Unit tests for compactex.evaluation.metrics using hand-built TaskExample fixtures."""

import pytest

from compactex.evaluation.metrics import (
    aggregate,
    bootstrap_ci,
    confusion_by_field,
    paired_bootstrap_delta,
    parse_prediction,
    score_record,
)
from compactex.schemas import (
    Evidence,
    Fact,
    FactPayload,
    Message,
    SchemaField,
    Selector,
    TaskExample,
    TaskSchema,
    Ticket,
    target_json,
)

MSG1 = Message(
    id="m1",
    role="user",
    content="We upgraded to 2.14.3 last night and now /api/v2/ship returns 502.",
)
MSG2 = Message(
    id="m2",
    role="agent",
    content="Rollback applied on host gw-prod-7; build 9.0.2-rc1 is live and the 502 is gone.",
)
SCHEMA = TaskSchema(
    fields=[
        SchemaField(
            name="CURRENT_ERROR_CODE",
            definition="HTTP status code the user currently reports; latest mention wins.",
            selector=Selector(kind="error_code", status="current", scope="latest"),
        ),
        SchemaField(
            name="ALL_VERSIONS",
            definition="Every version string mentioned anywhere in the conversation.",
            selector=Selector(kind="version", scope="all"),
        ),
    ]
)


def span_fact(type_: str, value: str, message: Message) -> Fact:
    start = message.content.index(value)
    return Fact(
        type=type_,
        value=value,
        evidence=Evidence(message_id=message.id, start=start, end=start + len(value)),
    )


def make_example(gold: list[Fact]) -> TaskExample:
    return TaskExample(
        id="t-001",
        ticket=Ticket(ticket_id="TKT-1", messages=[MSG1, MSG2]),
        schema=SCHEMA,
        gold=gold,
    )


GOLD = [
    span_fact("CURRENT_ERROR_CODE", "502", MSG2),
    span_fact("ALL_VERSIONS", "2.14.3", MSG1),
    span_fact("ALL_VERSIONS", "9.0.2-rc1", MSG2),
]


def test_perfect_prediction_scores_full_tp() -> None:
    example = make_example(GOLD)
    payload = FactPayload(facts=[f.model_copy(deep=True) for f in GOLD])
    record = score_record(example, payload)
    assert record["gold_n"] == 3
    assert record["tp"] == record["gold_n"]
    assert record["fp"] == 0
    assert record["fn"] == 0
    assert record["value_exact_copy"] == 3
    assert record["evidence_valid"] == 3
    assert record["unsupported"] == 0
    assert record["duplicates"] == 0
    assert record["unknown_field_types"] == []
    assert record["record_ok"] is True
    summary = aggregate([record])
    assert summary["precision"] == pytest.approx(1.0)
    assert summary["recall"] == pytest.approx(1.0)
    assert summary["f1"] == pytest.approx(1.0)
    assert summary["n_records"] == 1
    assert summary["no_fact_accuracy"] is None


def test_wrong_value_is_fn_and_fp() -> None:
    example = make_example(GOLD)
    start = MSG2.content.index("502")
    wrong = Fact(
        type="CURRENT_ERROR_CODE",
        value="500",
        evidence=Evidence(message_id="m2", start=start, end=start + 3),
    )
    payload = FactPayload(facts=[wrong] + [f.model_copy(deep=True) for f in GOLD[1:]])
    record = score_record(example, payload)
    assert record["tp"] == 2
    assert record["fp"] == 1
    assert record["fn"] == 1
    assert record["record_ok"] is False
    summary = aggregate([record])
    assert summary["precision"] == pytest.approx(2 / 3)
    assert summary["recall"] == pytest.approx(2 / 3)
    assert summary["f1"] == pytest.approx(2 / 3)
    confusion = confusion_by_field(example, payload)
    assert confusion["CURRENT_ERROR_CODE"] == {"gold": 1, "pred": 1, "tp": 0}
    assert confusion["ALL_VERSIONS"] == {"gold": 2, "pred": 2, "tp": 2}


def test_value_substring_check_catches_near_miss_corruption() -> None:
    example = make_example(GOLD)
    start = MSG1.content.index("2.14.3")
    corrupted = Fact(
        type="ALL_VERSIONS",
        value="2.14.30",  # trailing digit: value is NOT a substring of m1
        evidence=Evidence(message_id="m1", start=start, end=start + len("2.14.3")),
    )
    others = [f.model_copy(deep=True) for f in GOLD[:1] + GOLD[2:]]
    payload = FactPayload(facts=[corrupted, *others])
    record = score_record(example, payload)
    assert record["pred_n"] == 3
    assert record["value_exact_copy"] == 2
    assert record["unsupported"] == 1
    assert record["evidence_valid"] == 2  # slice yields "2.14.3", not the corrupted value
    assert record["tp"] == 2
    assert record["fn"] == 1
    assert aggregate([record])["unsupported_fact_rate"] == pytest.approx(1 / 3)


def test_unsupported_fact_detected() -> None:
    example = make_example(GOLD)
    hallucinated = Fact(
        type="CURRENT_ERROR_CODE",
        value="404",  # never appears in the ticket
        evidence=Evidence(message_id="m1", start=0, end=3),
    )
    payload = FactPayload(facts=[f.model_copy(deep=True) for f in GOLD] + [hallucinated])
    record = score_record(example, payload)
    assert record["pred_n"] == 4
    assert record["unsupported"] == 1
    assert record["value_exact_copy"] == 3
    assert record["evidence_valid"] == 3
    assert record["record_ok"] is False
    ghost_message = Fact(
        type="ALL_VERSIONS",
        value="1.0.0",
        evidence=Evidence(message_id="m99", start=0, end=5),
    )
    ghost_record = score_record(example, FactPayload(facts=[ghost_message]))
    assert ghost_record["unsupported"] == 1
    assert ghost_record["evidence_valid"] == 0
    assert aggregate([record, ghost_record])["unsupported_fact_rate"] == pytest.approx(2 / 5)


def test_empty_gold_and_empty_pred_is_no_fact_correct() -> None:
    example = make_example(gold=[])
    record = score_record(example, FactPayload(facts=[]))
    assert record["gold_n"] == 0
    assert record["pred_n"] == 0
    assert record["no_fact_correct"] is True
    assert record["record_ok"] is True
    summary = aggregate([record])
    assert summary["no_fact_accuracy"] == pytest.approx(1.0)
    assert summary["f1"] == 0.0
    noisy = score_record(example, FactPayload(facts=[span_fact("ALL_VERSIONS", "2.14.3", MSG1)]))
    assert noisy["no_fact_correct"] is False
    assert noisy["record_ok"] is False
    assert aggregate([record, noisy])["no_fact_accuracy"] == pytest.approx(0.5)


def test_duplicate_keys_counted() -> None:
    example = make_example(gold=[GOLD[0]])
    payload = FactPayload(facts=[GOLD[0].model_copy(deep=True), GOLD[0].model_copy(deep=True)])
    record = score_record(example, payload)
    assert record["pred_n"] == 2
    assert record["duplicates"] == 1
    assert record["tp"] == 1
    assert record["fp"] == 1
    assert record["record_ok"] is False
    assert aggregate([record])["duplicate_rate"] == pytest.approx(0.5)


def test_unknown_field_detected() -> None:
    example = make_example(GOLD)
    alien = span_fact("LAST_HOST", "gw-prod-7", MSG2)  # not a schema field name
    payload = FactPayload(facts=[alien])
    record = score_record(example, payload)
    assert record["unknown_field_types"] == ["LAST_HOST"]
    assert record["unknown_fields"] == 1
    assert record["tp"] == 0
    assert record["fn"] == 3
    summary = aggregate([record])
    assert summary["unknown_field_rate"] == pytest.approx(1.0)
    assert "LAST_HOST" in confusion_by_field(example, payload)
    assert confusion_by_field(example, payload)["LAST_HOST"]["pred"] == 1


def test_bootstrap_ci_on_constant_list_has_zero_width() -> None:
    lo, hi = bootstrap_ci([0.75] * 20)
    assert lo == pytest.approx(0.75)
    assert hi == pytest.approx(0.75)
    assert hi - lo == pytest.approx(0.0, abs=1e-12)


def test_bootstrap_ci_deterministic_and_brackets_mean() -> None:
    values = [1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0]
    first = bootstrap_ci(values, B=200, seed=7)
    second = bootstrap_ci(values, B=200, seed=7)
    assert first == second
    mean = sum(values) / len(values)
    assert first[0] < mean < first[1]


def test_paired_bootstrap_delta_constant_gap_zero_width() -> None:
    lo, hi, mean = paired_bootstrap_delta([1.0] * 8, [0.0] * 8)
    assert mean == pytest.approx(1.0)
    assert lo == pytest.approx(1.0)
    assert hi == pytest.approx(1.0)


def test_paired_bootstrap_delta_identical_lists_centered_on_zero() -> None:
    values = [0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0]
    lo, hi, mean = paired_bootstrap_delta(values, values)
    assert mean == pytest.approx(0.0)
    assert lo == pytest.approx(0.0)
    assert hi == pytest.approx(0.0)


def test_parse_prediction_invalid_json_returns_none_with_flags() -> None:
    payload, flags = parse_prediction("nope {")
    assert payload is None
    assert flags["json_valid"] is False
    assert flags["schema_valid"] is False
    assert flags["error"] is not None


def test_parse_prediction_rejects_markdown_fences() -> None:
    payload, flags = parse_prediction('```json\n{"facts": []}\n```')
    assert payload is None
    assert flags["json_valid"] is False
    assert flags["schema_valid"] is False


def test_parse_prediction_json_valid_but_schema_invalid() -> None:
    payload, flags = parse_prediction('{"facts": [{"type": 1, "value": "502"}]}')
    assert payload is None
    assert flags["json_valid"] is True
    assert flags["schema_valid"] is False
    assert flags["error"] is not None


def test_parse_prediction_roundtrips_gold_payload() -> None:
    payload, flags = parse_prediction(target_json(GOLD))
    assert flags == {"json_valid": True, "schema_valid": True, "error": None}
    assert payload is not None
    assert payload.facts == GOLD
    example = make_example(GOLD)
    record = score_record(example, payload) | flags
    assert aggregate([record])["json_valid_rate"] == pytest.approx(1.0)


def test_score_record_with_none_payload() -> None:
    example = make_example(GOLD)
    record = score_record(example, None)
    assert record["pred_n"] == 0
    assert record["fn"] == 3
    assert record["tp"] == 0
    assert record["json_valid"] is False
    assert record["schema_valid"] is False
    assert record["record_ok"] is False
