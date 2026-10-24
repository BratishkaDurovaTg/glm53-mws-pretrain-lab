# RESULTS — pilot1 (pre-spec stage 1) — FINAL

## Setup

- Student: Qwen/Qwen2.5-0.5B-Instruct @ 7ae55760 (MPS/BF16, greedy, no thinking).
- SFT: LoRA r=16 α=32 (q,k,v,o) — 2,162,688 trainable (0.44%), 2500 mechanical-gold
  schema-conditioned examples (~29% empty-gold hard negatives), 1 epoch = 313 steps,
  effective batch 8, bf16 completion-only CE, cosine 1e-4, warmup 20. Wall: 67 min.
- Gates: smoke PASS; tiny-overfit PASS (loss 1.36→0.0001 AND 2/2 exact regeneration
  from the saved adapter, exact evidence spans included).
- Checkpoints: early(78) / best-by-dev-gen-F1(313) / final(313) — best == final.

## Base vs tuned (strict full grounded F1: type+value+message_id+exact span)

| suite | base strict | base lenient (type,value) | tuned strict | tuned tv_f1 | schema_valid | paired Δ [95% CI] |
| --- | --- | --- | --- | --- | --- | --- |
| dev | 0.000 | 0.261 | **0.499** | 0.516 | 1.00 | +0.247 [+0.141,+0.351] ✓ |
| T1 literal | 0.000 | 0.251 | **0.522** | 0.522 | 0.99 | +0.240 [+0.140,+0.338] ✓ |
| T2 unseen-schema | 0.000 | 0.209 | **0.527** | 0.527 | 1.00 | +0.137 [+0.043,+0.230] ✓ |
| T4 adversarial | 0.000 | 0.213 | **0.531** | 0.535 | 0.99 | +0.257 [+0.166,+0.347] ✓ |
| T3 technical (hand-written) | 0.000 | 0.500 | **0.406** | 0.406 | 1.00 | +0.348 [+0.143,+0.540] ✓ |

Paired Δ computed on per-record strict F1 (empty↔empty credits 1.0 to both sides;
base gets vacuous credit on empty-gold records — the DELTA is the meaningful number).

### SEALED milestone (aggregates only, run once)

| suite | base lenient tv | tuned strict F1 | schema | unsup |
| --- | --- | --- | --- | --- |
| T1 literal sealed | 0.080 | **0.337** | 0.99 | 0.18 |
| T2 unseen-schema sealed | 0.119 | **0.319** | 1.00 | 0.23 |
| T4 adversarial sealed | 0.073 | **0.333** | 0.99 | 0.03 |
| SEALED FINAL | 0.085 | **0.367** | 1.00 | 0.15 |

Sealed direction is CONSISTENT with dev-side (no contradiction; lower absolute —
everything unseen: structures, schemas, literal pools, higher empty share).

## Diagnostic decomposition (owner-requested)

- `tv_f1 ≈ strict f1` almost everywhere → **evidence/serialization mechanics were
  learned essentially perfectly** (evidence_accuracy_given_tv ≈ 1.0).
- Base had hidden semantics (lenient tv 0.21–0.50) blocked by serialization;
  tuned roughly doubled semantic quality (T2: 0.209→0.527) AND fixed serialization
  (schema 0.00→1.00). Both components improved — not just format learning.
- type_f1 (0.61–0.70 dev side) > tv_f1 → residual errors are mostly wrong VALUE
  selection among candidates, not wrong field choice.

## Failure modes (t2+t4 dev-side clustering)

| bucket | count |
| --- | --- |
| false-positive on empty gold | 91 |
| wrong selection (fp+fn mixed) | 58 |
| over-extraction (fp only) | 44 |
| unsupported value | 32 |
| total miss | 31 |
| under-extraction (fn only) | 23 |

**Dominant: over-extraction / abstention failure** (no_fact_accuracy 0.26–0.52 dev
side). The model prefers guessing over abstaining. T3 shows the opposite face:
conservative recall (R=0.286) on out-of-template text — decision style tuned to the
template distribution.

## Verdict

**KEEP (conditional).** The core hypothesis — schema-conditioned synthetic SFT
creates transferable extraction behaviour on unseen schemas — is supported on every
independent suite (dev, T1, T2, T4, T3) with significant paired CIs, and confirmed
in direction by the sealed milestone (0.32–0.37 strict from 0.00). Caveats kept on
record: (1) abstention is the top weakness; (2) T3 out-of-template recall is low;
(3) all of this remains synthetic/domain-proxy — the decisive test is official MWS
data via `docs/48H_HANDOFF.md`.

## Next single highest-value experiment (ONE)

Error-driven batch 2 targeting **abstention/hard negatives**: raise empty-gold +
distractor share and near-miss candidate density (t4-style) in a ~2–3k addition,
retrain same recipe, re-evaluate frozen suites; success criterion = no_fact_accuracy
≥0.7 on dev side without T2 regression beyond CI. No scaling to 20k, no RL, no new
branches before that reads out.

## Honest limitations

1. Everything is synthetic/hand-written proxy; not MWS quality.
2. Frame bank shared across template families (isolation is structural).
3. "Resolved" wording + dates reads oddly; log-line uuid under `lvl=err` is a
   superficial distractor (definitions disambiguate).
4. HTTP codes/ports are shared closed pools.
5. Base no_fact_accuracy (vacuous 1.0) vs tuned (real) is not a like-for-like
   comparison; over-extraction is measured against gold, not against base.
6. Sealed boundary is procedural (aggregate-only tooling), not physical.

## Token / compute economics

- GLM tokens: ~10⁵–10⁶ order total (design, code, review, T3 hand-rendering) —
  far under stage ceilings; no paid teacher generation (gold is mechanical).
- Student compute: baseline ~2.2 h, pilot 67 min, evals ~1.5 h, sealed ~40 min.
