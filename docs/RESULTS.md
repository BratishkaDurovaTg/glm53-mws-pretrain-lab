# RESULTS (pre-spec stage 1)

Status: PILOT IN PROGRESS — this file is finalized after the pilot + sealed milestone.
Numbers below marked TBD until then.

## Setup

- Student: Qwen/Qwen2.5-0.5B-Instruct @ 7ae55760 (MPS/BF16, greedy, no thinking).
- Training: LoRA r=16 α=32 (q,k,v,o), 2.16M trainable (0.44%), 2500 synthetic
  schema-conditioned examples, 1 epoch (313 steps, effective batch 8), bf16 CE on
  completion tokens only, cosine LR 1e-4, 20 warmup.
- Gates before pilot: smoke PASS; tiny-overfit PASS (loss 1.36→0.0001 AND 2/2 exact
  regeneration from the saved adapter, including exact evidence spans).

## Base baseline (all 8 suites + T3)

Uniform: **F1 = 0.000, schema_valid = 0.00**, json_valid 0.85–0.97. The base model
produces valid JSON (often `{FIELD: [values]}`) but never the canonical
`{"facts":[{type,value,evidence}]}` contract; it also invents values (enumerating
plausible port numbers). Reference repo's earlier finding (0/6 schema-valid with a
stricter schema) fully reproduced at scale. Base no_fact_accuracy is vacuous
(nothing parses → empty predictions).

## Pilot results — TBD

| suite | base F1 | tuned F1 | Δ [95% CI] | verdict |
| --- | --- | --- | --- | --- |
| dev (unseen names/combos) | 0.000 | TBD | TBD | TBD |
| T2 unseen-schema | 0.000 | TBD | TBD | TBD |
| T1 literal | 0.000 | TBD | TBD | TBD |
| T4 adversarial | 0.000 | TBD | TBD | TBD |
| T3 technical (hand-written) | 0.000 | TBD | TBD | TBD |
| SEALED milestone | 0.000 | TBD | TBD | TBD |

Guard metrics (must not degrade materially): exact-copy rate, unsupported-fact
rate, no-fact accuracy — TBD.

## Known limitations (honest list)

1. All suites are synthetic/hand-written proxies; nothing here is MWS quality.
2. Frame bank is shared across template families — family isolation is structural
   (assembly patterns), not lexical.
3. "Resolved" wording paired with dates reads oddly ("2026-08-15 уже исправлено") —
   mechanically consistent, linguistically awkward.
4. Log-line rendering puts UUIDs under `lvl=err` labels — superficial error-code
   distractor, kind definitions disambiguate.
5. HTTP codes and ports are shared closed pools (documented non-disjoint).
6. Sealed suites have high empty-gold share (37–50%) — F1 driven by fewer
   positive records; no_fact accuracy reported alongside.

## Token / compute economics (stage 1)

- GLM tokens (design, code, review, hand-rendered T3): order of a few hundred
  thousand — far below the stage-1 ceiling; no paid teacher generation was needed
  because gold is mechanical.
- Student compute: smoke 26s; tiny-overfit 17.7 min (contended); baseline ~2.2h
  wall (1600+ generations); pilot ~1h.
- Detailed ledger: experiments/runs.jsonl.
