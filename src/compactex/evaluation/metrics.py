"""Pure scoring metrics for grounded fact extraction; stdlib + pydantic only, no ML deps."""

import json
import math
import random
from collections import Counter
from statistics import fmean
from typing import Any

from pydantic import ValidationError

from compactex.schemas import Fact, FactPayload, TaskExample

FactKey = tuple[str, str, str]  # (type, value, evidence.message_id)
ParseFlags = dict[str, bool | str | None]
Record = dict[str, Any]
Summary = dict[str, float | int | None]
FieldConfusion = dict[str, dict[str, int]]


def parse_prediction(raw_output: str) -> tuple[FactPayload | None, ParseFlags]:
    """Parse raw model output into a FactPayload; strict json.loads, no markdown repair."""
    try:
        data = json.loads(raw_output)
    except ValueError as exc:  # JSONDecodeError subclasses ValueError
        return None, {"json_valid": False, "schema_valid": False, "error": str(exc)}
    try:
        payload = FactPayload.model_validate(data)
    except ValidationError as exc:
        return None, {"json_valid": True, "schema_valid": False, "error": str(exc)}
    return payload, {"json_valid": True, "schema_valid": True, "error": None}


def _fact_key(fact: Fact) -> FactKey:
    return (fact.type, fact.value, fact.evidence.message_id)


def score_record(example: TaskExample, payload: FactPayload | None) -> Record:
    """Score one prediction against gold as per-record counts and validity flags.

    Fact identity for tp/fp/fn is (type, value, evidence.message_id) with multiset matching.
    `json_valid`/`schema_valid` are inferred from payload presence; when merging records,
    spread the parse_prediction flags over this dict to override them.
    """
    content = {m.id: m.content for m in example.ticket.messages}
    field_names = {f.name for f in example.schema_.fields}
    pred_facts: list[Fact] = [] if payload is None else list(payload.facts)

    gold_keys = Counter(_fact_key(f) for f in example.gold)
    pred_keys = Counter(_fact_key(f) for f in pred_facts)
    tp = sum(min(count, gold_keys[key]) for key, count in pred_keys.items())
    fp = sum(count - gold_keys[key] for key, count in pred_keys.items() if count > gold_keys[key])
    fn = sum(count - pred_keys[key] for key, count in gold_keys.items() if count > pred_keys[key])

    def cited(fact: Fact) -> str | None:
        return content.get(fact.evidence.message_id)

    def value_in_message(fact: Fact) -> bool:
        message = cited(fact)
        return message is not None and fact.value in message

    def span_matches(fact: Fact) -> bool:
        message = cited(fact)
        span = message[fact.evidence.start : fact.evidence.end] if message is not None else None
        return span == fact.value

    gold_n = len(example.gold)
    pred_n = len(pred_facts)
    value_exact_copy = sum(1 for f in pred_facts if value_in_message(f))
    evidence_valid = sum(1 for f in pred_facts if span_matches(f))
    unsupported = sum(1 for f in pred_facts if not value_in_message(f))
    duplicates = sum(count - 1 for count in pred_keys.values() if count > 1)
    unknown_types = sorted({f.type for f in pred_facts if f.type not in field_names})
    unknown_fields = sum(1 for f in pred_facts if f.type not in field_names)
    record_ok = (
        payload is not None
        and tp == gold_n
        and fp == 0
        and fn == 0
        and duplicates == 0
        and unsupported == 0
        and evidence_valid == pred_n
    )
    return {
        "gold_n": gold_n,
        "pred_n": pred_n,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "value_exact_copy": value_exact_copy,
        "evidence_valid": evidence_valid,
        "unsupported": unsupported,
        "duplicates": duplicates,
        "unknown_fields": unknown_fields,
        "unknown_field_types": unknown_types,
        "no_fact_correct": gold_n == 0 and pred_n == 0,
        "record_ok": record_ok,
        "json_valid": payload is not None,
        "schema_valid": payload is not None,
    }


def aggregate(records: list[Record]) -> Summary:
    """Micro-aggregate per-record counts into suite-level rates.

    Fact-level rates divide by total predicted facts and are 0.0 when nothing was
    predicted; `no_fact_accuracy` is None when no record has empty gold.
    """
    n_records = len(records)

    def total(key: str) -> int:
        return sum(int(r.get(key, 0)) for r in records)

    tp, fp, fn = total("tp"), total("fp"), total("fn")
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    pred_n = total("pred_n")

    def fact_rate(key: str) -> float:
        return total(key) / pred_n if pred_n else 0.0

    empty_gold = [r for r in records if int(r.get("gold_n", 0)) == 0]
    no_fact_accuracy = (
        fmean([float(r.get("no_fact_correct", False)) for r in empty_gold]) if empty_gold else None
    )
    json_valid_rate = (
        fmean([float(r.get("json_valid", False)) for r in records]) if records else 0.0
    )
    schema_valid_rate = (
        fmean([float(r.get("schema_valid", False)) for r in records]) if records else 0.0
    )
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "value_exact_copy_rate": fact_rate("value_exact_copy"),
        "unsupported_fact_rate": fact_rate("unsupported"),
        "evidence_valid_rate": fact_rate("evidence_valid"),
        "duplicate_rate": fact_rate("duplicates"),
        "unknown_field_rate": fact_rate("unknown_fields"),
        "no_fact_accuracy": no_fact_accuracy,
        "json_valid_rate": json_valid_rate,
        "schema_valid_rate": schema_valid_rate,
        "n_records": n_records,
    }


def _percentile(sorted_values: list[float], q: float) -> float:
    """Linear-interpolation percentile of an already sorted list; q in [0, 1]."""
    if not sorted_values:
        raise ValueError("empty sample")
    pos = q * (len(sorted_values) - 1)
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return sorted_values[lo]
    frac = pos - lo
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * frac


def bootstrap_ci(
    per_record_flags_or_counts: list[float],
    B: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile bootstrap CI of the mean via record-count resampling."""
    values = per_record_flags_or_counts
    if not values:
        raise ValueError("values must be non-empty")
    rng = random.Random(seed)
    n = len(values)
    means = sorted(fmean([values[rng.randrange(n)] for _ in range(n)]) for _ in range(B))
    return _percentile(means, alpha / 2), _percentile(means, 1 - alpha / 2)


def paired_bootstrap_delta(
    a: list[float],
    b: list[float],
    B: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Paired bootstrap CI of mean(a) - mean(b); indices are resampled once per replicate."""
    if not a or len(a) != len(b):
        raise ValueError("a and b must be non-empty and of equal length")
    rng = random.Random(seed)
    n = len(a)
    deltas: list[float] = []
    for _ in range(B):
        indices = [rng.randrange(n) for _ in range(n)]
        deltas.append(fmean([a[i] for i in indices]) - fmean([b[i] for i in indices]))
    deltas.sort()
    observed = fmean(a) - fmean(b)
    return _percentile(deltas, alpha / 2), _percentile(deltas, 1 - alpha / 2), observed


def confusion_by_field(example: TaskExample, payload: FactPayload | None) -> FieldConfusion:
    """Per-field gold/pred/tp counts for error analysis; unknown predicted types appended."""
    gold_keys = Counter(_fact_key(f) for f in example.gold)
    pred_facts: list[Fact] = [] if payload is None else payload.facts
    pred_keys = Counter(_fact_key(f) for f in pred_facts)
    tp_by_field: Counter[str] = Counter()
    for key, count in pred_keys.items():
        tp_by_field[key[0]] += min(count, gold_keys[key])
    gold_by_field: Counter[str] = Counter(f.type for f in example.gold)
    pred_by_field: Counter[str] = Counter(f.type for f in pred_facts)
    known = [f.name for f in example.schema_.fields]
    extra = sorted((set(gold_by_field) | set(pred_by_field)) - set(known))
    return {
        name: {"gold": gold_by_field[name], "pred": pred_by_field[name], "tp": tp_by_field[name]}
        for name in [*known, *extra]
    }
