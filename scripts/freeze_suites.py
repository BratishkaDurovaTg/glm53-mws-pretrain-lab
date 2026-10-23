"""Generate and freeze ALL evaluation suites + train/dev splits. Run once per lab stage.

Freeze = write JSONL + sha256 manifest. Never regenerate after seeing scores; a new
generation must use new seeds and a new stage name.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from compactex.suites import SuiteConfig, freeze_suite, generate_suite  # noqa: E402

SUITES = [
    # splits
    SuiteConfig("train", "train", n=2500, seed_base=100_000,
                note="SFT corpus, seen schemas/families"),
    SuiteConfig("dev", "dev", n=150, seed_base=200_000, empty_rate=0.25,
                note="checkpoint selection, unseen names/combos"),
    # benchmarks (dev side: inspectable)
    SuiteConfig("t1_literal", "dev", n=200, seed_base=310_000, empty_rate=0.25,
                kind_weights={"error_code": 4, "version": 4, "uuid": 3, "ip": 3, "url": 2,
                              "host": 2, "path": 2, "resource_id": 2, "ticket_ref": 1,
                              "datetime": 1, "port": 1},
                family_filter=["kv_lines", "table_lines"],
                note="deterministic literal copying/candidate selection"),
    SuiteConfig("t2_unseen_schema", "dev", n=250, seed_base=320_000, empty_rate=0.3,
                note="schema-following, dev bank only"),
    SuiteConfig("t4_adversarial", "dev", n=200, seed_base=330_000, empty_rate=0.3,
                negative_rate=0.4, n_mentions=(4, 9),
                status_weights={"current": 0.3, "historical": 0.3, "hypothetical": 0.2,
                                "resolved": 0.2},
                note="near-miss candidates, distractors, competing statuses"),
    # benchmarks (sealed side: milestone only)
    SuiteConfig("t1_literal", "sealed", n=150, seed_base=510_000, empty_rate=0.25,
                kind_weights={"error_code": 4, "version": 4, "uuid": 3, "ip": 3, "url": 2,
                              "host": 2, "path": 2, "resource_id": 2, "ticket_ref": 1,
                              "datetime": 1, "port": 1},
                family_filter=["stacked_logs", "rapid_chat", "report_style"],
                note="sealed literal"),
    SuiteConfig("t2_unseen_schema", "sealed", n=250, seed_base=520_000, empty_rate=0.3,
                note="sealed unseen-schema compositions"),
    SuiteConfig("t4_adversarial", "sealed", n=150, seed_base=530_000, empty_rate=0.3,
                negative_rate=0.4, n_mentions=(4, 9),
                status_weights={"current": 0.3, "historical": 0.3, "hypothetical": 0.2,
                                "resolved": 0.2},
                note="sealed adversarial"),
    SuiteConfig("sealed_final", "sealed", n=250, seed_base=540_000, empty_rate=0.3,
                note="SEALED FINAL TEST - milestone verdicts only"),
]

OUT = {
    ("train", "train"): ROOT / "data" / "train",
    ("dev", "dev"): ROOT / "data" / "dev",
    ("t1_literal", "dev"): ROOT / "benchmarks" / "literal" / "dev",
    ("t1_literal", "sealed"): ROOT / "benchmarks" / "literal" / "sealed",
    ("t2_unseen_schema", "dev"): ROOT / "benchmarks" / "unseen_schema" / "dev",
    ("t2_unseen_schema", "sealed"): ROOT / "benchmarks" / "unseen_schema" / "sealed",
    ("t4_adversarial", "dev"): ROOT / "benchmarks" / "adversarial" / "dev",
    ("t4_adversarial", "sealed"): ROOT / "benchmarks" / "adversarial" / "sealed",
    ("sealed_final", "sealed"): ROOT / "benchmarks" / "sealed_final",
}


def main() -> None:
    for cfg in SUITES:
        print(f"generating {cfg.name}/{cfg.pool} n={cfg.n} ...")
        result = generate_suite(cfg)
        out = OUT[(cfg.name, cfg.pool)]
        path = freeze_suite(result, out)
        m = result.quarantined
        print(f"  -> {path} | empty-gold share: "
              f"{sum(1 for e in result.examples if not e.gold)/max(len(result.examples),1):.0%}"
              f" | quarantined: {m} | dupes dropped: {result.dropped_dupe}")


if __name__ == "__main__":
    main()
