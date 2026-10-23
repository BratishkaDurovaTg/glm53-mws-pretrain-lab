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


def harvest_pairs(raw_output: str) -> list[tuple[str, str]]:
    """DIAGNOSTIC-ONLY lenient harvest of (type, value) pairs from any JSON shape.

    Recovers semantic pairs from outputs that fail the strict payload schema
    (e.g. `{"FIELD": ["v1"]}` or nested lists). Never used for strict scoring.
    """
    try:
        data = json.loads(raw_output)
    except ValueError:
        return []
    pairs: list[tuple[str, str]] = []

    def add(k: str, v: Any) -> None:
        if isinstance(v, str) and v.strip():
            pairs.append((k, v))
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            pairs.append((k, str(v)))

    def walk_typed(node: Any, key: str | None = None) -> None:
        if isinstance(node, dict):
            t = node.get("type") or node.get("field") or node.get("name")
            v = node.get("value") if isinstance(t, str) else None
            if isinstance(t, str) and v is not None:
                add(t, v)
                return
            for k, v in node.items():
                if isinstance(v, (list, dict)):
                    walk_typed(v, k if isinstance(v, list) else None)
                elif k not in ("evidence", "message_id", "start", "end"):
                    add(k, v)
        elif isinstance(node, list):
            for item in node:
                walk_typed(item, key)
        elif key is not None:
            add(key, node)

    walk_typed(data)
    return pairs


def lenient_tv_counts(example: TaskExample, raw_output: str) -> dict[str, int]:
    """Multiset (type, value) match of harvested pairs vs gold; diagnostic only."""
    gold_tv = Counter((f.type, f.value) for f in example.gold)
    pred_tv = Counter(harvest_pairs(raw_output))
    tp = sum(min(c, gold_tv[k]) for k, c in pred_tv.items())
    fp = sum(c - gold_tv[k] for k, c in pred_tv.items() if c > gold_tv[k])
    fn = sum(c - pred_tv[k] for k, c in gold_tv.items() if c > pred_tv[k])
    return {"tv_tp": tp, "tv_fp": fp, "tv_fn": fn}


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

    # diagnostic decomposition: semantic match (type, value) ignoring evidence,
    # and type-level match ignoring value — separates extraction semantics from
    # serialization/evidence mechanics (strict full F1 above stays primary)
    gold_tv = Counter((f.type, f.value) for f in example.gold)
    pred_tv = Counter((f.type, f.value) for f in pred_facts)
    tv_tp = sum(min(c, gold_tv[k]) for k, c in pred_tv.items())
    tv_fp = sum(c - gold_tv[k] for k, c in pred_tv.items() if c > gold_tv[k])
    tv_fn = sum(c - pred_tv[k] for k, c in gold_tv.items() if c > pred_tv[k])
    gold_types = Counter(f.type for f in example.gold)
    pred_types = Counter(f.type for f in pred_facts)
    type_tp = sum(min(c, gold_types[k]) for k, c in pred_types.items())
    type_fp = sum(c - gold_types[k] for k, c in pred_types.items() if c > gold_types[k])
    type_fn = sum(c - pred_types[k] for k, c in gold_types.items() if c > pred_types[k])

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
        "tv_tp": tv_tp,
        "tv_fp": tv_fp,
        "tv_fn": tv_fn,
        "type_tp": type_tp,
        "type_fp": type_fp,
        "type_fn": type_fn,
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

    def prf(t: int, f: int, n: int) -> tuple[float, float, float]:
        p = t / (t + f) if t + f else 0.0
        r = t / (t + n) if t + n else 0.0
        return p, r, (2 * p * r / (p + r) if p + r else 0.0)

    tv_p, tv_r, tv_f1 = prf(total("tv_tp"), total("tv_fp"), total("tv_fn"))
    ty_p, ty_r, ty_f1 = prf(total("type_tp"), total("type_fp"), total("type_fn"))
    tv_tp_total = total("tv_tp")
    type_tp_total = total("type_tp")

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
        "type_value_precision": tv_p,
        "type_value_recall": tv_r,
        "type_value_f1": tv_f1,
        "type_precision": ty_p,
        "type_recall": ty_r,
        "type_f1": ty_f1,
        "evidence_accuracy_given_tv": (tp / tv_tp_total) if tv_tp_total else None,
        "value_exact_match_given_type": (tv_tp_total / type_tp_total) if type_tp_total else None,
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
