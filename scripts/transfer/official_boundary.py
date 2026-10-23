"""THIN BOUNDARY to official MWS data. Implement these two functions on Day 1.

Everything else in the lab (runner, metrics, SFT, checkpoints) works through
this boundary unchanged. Keep the implementations in ONE place; do not let
official-format knowledge leak into the lab code.
"""

from pathlib import Path

from compactex.schemas import Evidence, Fact, Message, TaskExample, TaskSchema
from compactex.suites import load_suite


def load_official_examples(split: str = "dev") -> list[TaskExample]:
    """Convert official data to canonical TaskExample.

    Implement when official materials arrive:
    - read official tickets (their input format);
    - map their fact ontology to TaskSchema fields WITH definitions phrased from
      the official annotation guidelines (schema-as-data, same as the lab);
    - map their gold facts to Fact(type=<official field>, value=<exact substring>,
      evidence=Message id + char span [start, end) inside that message);
    - normalizations: if official uses token/byte offsets or global offsets,
      convert HERE once.
    """
    raise NotImplementedError("fill after official data arrives")


def score_official(raw_outputs: dict[str, str], examples: list[TaskExample]) -> dict:
    """Score raw model outputs with the OFFICIAL scorer.

    Default until the official scorer is wired: our exact-F1 (identity
    (type, value, message_id)) — clearly labelled as internal, not official.
    Replace by calling the official scorer on the converted submission format.
    """
    from compactex.evaluation.metrics import aggregate, parse_prediction, score_record

    recs = []
    by_id = {ex.id: ex for ex in examples}
    for ex_id, raw in raw_outputs.items():
        ex = by_id[ex_id]
        payload, flags = parse_prediction(raw)
        rec = score_record(ex, payload)
        rec |= flags
        recs.append(rec)
    metrics = aggregate(recs)
    metrics["scorer"] = "INTERNAL_exact_f1_NOT_OFFICIAL"
    return metrics


def dev_path() -> Path:
    """Path to frozen official-dev once split is made (reference repo owns the split)."""
    raise NotImplementedError
