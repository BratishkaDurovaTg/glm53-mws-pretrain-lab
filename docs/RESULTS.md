# RESULTS — pilot1 (pre-spec stage 1) — CORRECTED SCORING

## Correction note (owner review, this revision)

The previously reported "strict full grounded F1" was in fact **message-grounded**
(identity `(type, value, message_id)`); it never checked `start/end`. Historical
numbers remain reproducible under the renamed `message_grounded_f1` (the `f1`
aggregate key is kept as a documented alias). New `exact_grounded_*` metrics use
the full 5-tuple identity. The claim "evidence mechanics learned essentially
perfectly" was WRONG and is retracted: it compared message ids, not spans.

## Setup

- Student: Qwen/Qwen2.5-0.5B-Instruct @ 7ae55760 (MPS/BF16, greedy, no thinking).
- SFT: LoRA r=16 α=32 (q,k,v,o) — 2,162,688 trainable (0.44%), 2500 mechanical-gold
  schema-conditioned examples (~29% empty-gold hard negatives), 1 epoch = 313 steps,
  effective batch 8, bf16 completion-only CE, cosine 1e-4, warmup 20. Wall: 67 min.
- Gates: smoke PASS; tiny-overfit PASS (loss→0.0001 AND 2/2 exact regeneration of
  full JSON targets — offsets included — from overfit memory; NOT evidence of
  offset competence on unseen data).

## Corrected results — claims separated

### 1. FORMAT LEARNING — strong

schema_valid 0.00 → **0.99–1.00** on every suite (dev side). json_valid stays high.
The canonical `{"facts":[{type,value,evidence}]}` contract was learned.

### 2. SEMANTIC EXTRACTION — significant on template-adjacent + unseen schemas; NOT demonstrated on out-of-template text

Apples-to-apples: same lenient (type,value) harvester applied to BOTH raw outputs.

| suite | base tv | tuned tv | paired Δ [95% CI] | verdict |
| --- | --- | --- | --- | --- |
| **T2 unseen-schema** | 0.209 | 0.527 | **+0.203 [+0.151,+0.253]** | SIG |
| T4 adversarial | 0.213 | 0.533 | +0.223 [+0.167,+0.279] | SIG |
| T1 literal | 0.251 | 0.519 | +0.209 [+0.149,+0.272] | SIG |
| dev | 0.261 | 0.516 | +0.193 [+0.120,+0.264] | SIG |
| T3 technical (hand-written) | 0.500 | 0.406 | −0.111 [−0.270,+0.049] | **ns** |

Unseen-SCHEMA transfer is real (T2 CI-solid). On hand-written out-of-template
text the base already extracted ~0.50 semantically and the pilot added nothing
(numerically negative, not significant). Breadth of semantic transfer beyond
the template distribution is UNPROVEN.

### 3. MESSAGE GROUNDING — strong

`(type,value,message_id)` F1: 0.000 → **0.499–0.531** (all suites, paired CIs
+0.39–0.44, SIG). message_accuracy_given_tv ≈ 1.0: when the fact is right, the
right message is cited.

### 4. EXACT SPAN GROUNDING — FAILED

`exact_grounded_f1` = **0.000–0.004** everywhere (paired Δ ns). Worse:
`evidence_valid_rate` (source-consistency: `content[start:end] == value`) is
0.000–0.011 — the generated offsets do not even slice out the model's own value.
The model emits plausible integer offsets without character counting.

**Assessment (§9, documented not implemented):** this is primarily an
OUTPUT-DESIGN problem, not just a model problem. Asking a 0.5B BPE-decoder to
count Unicode codepoints is brittle by construction. Candidate architecture if
official MWS requires offsets: model predicts `(type, exact value, message_id)`;
a deterministic postprocessor derives `start/end` by locating the value in the
cited message (unique occurrence) with occurrence-aware disambiguation
otherwise. Decision deferred to official requirements — do NOT re-engineer
outputs for synthetic metrics.

### 5. ABSTENTION — weak

no_fact_accuracy 0.26–0.52 (dev side); over-extraction dominates failures
(91 FP-on-empty + 44 over vs 54 under in t2+t4 clustering).

### SEALED milestone (aggregates only; exact-grounded not recomputable — per-example raws destroyed by sealed boundary; next milestone will report it)

message-grounded: T1s 0.337 / T2s 0.319 / T4s 0.333 / final 0.367 (base 0.000;
base lenient tv 0.073–0.119), schema 0.99–1.00. Direction consistent with dev.

## Verdict — CORRECTED

**KEEP-CONDITIONAL, scope sharpened.**

Supported by evidence:
- canonical format following (0→1);
- semantic (type,value) extraction on unseen schemas: +0.20 CI[+0.15,+0.25];
- message-level grounding: +0.39 CI-solid.

Not supported / failed:
- semantic transfer to out-of-template hand-written text (T3 ns);
- exact character offsets (failed; likely output-design, see §4);
- abstention (weak).

The pre-model is useful as an initializer IF official MWS scoring rewards
(type/value/message)-level extraction or tolerates programmatic span derivation —
decisive test remains the official data via `docs/48H_HANDOFF.md`.

## Next single highest-value experiment

Unchanged candidate (abstention batch 2), but the corrected decomposition adds a
candidate: T3-style out-of-template semantic breadth. Owner decides which ONE:
(a) abstention/hard-negative batch 2; (b) out-of-template semantic breadth probe
(hand-written-style training data); (c) drop offsets from the canonical target
and re-baseline (only if we accept offsets as postprocessing — an output-design
change, NOT a metric chase).

## Honest limitations

1. Everything is synthetic/hand-written proxy; not MWS quality.
2. Frame bank shared across template families (structural isolation only).
3. Historical "f1" numbers = message-grounded; do not cite as span-exact.
4. Sealed exact-grounded pending next milestone run (boundary policy).
5. Base no_fact_accuracy (vacuous 1.0) vs tuned (real) not like-for-like.
6. Sealed boundary is procedural, not physical.

## Token / compute economics

- GLM tokens: ~10⁵–10⁶ order total; no paid teacher generation (gold mechanical).
- Student compute: baseline ~2.2 h, pilot 67 min, evals ~1.5 h, sealed ~40 min.
- This correction pass: rescoring only, zero new inference.

---

# EXPERIMENT 2 — abstention/hard-negative batch2 — REJECT

## Design (single variable: training data distribution)

P2 = pilot1/ckpt_best + continuation (1 epoch, 313 steps, same LoRA/optimizer/LR/
batch/decoding/prompts/evaluator) on batch2: 2500 fresh examples — 45% empty-gold
hard negatives (literal-rich scenes, kind present, schema filters exclude all),
27.5% near-miss disambiguation (3–6 same-kind candidates, subset gold),
27.5% mixed positive/negative schemas. Manifest `data/batch2/manifest.json`
(sha256 8793a8c…). Continuation smoke PASS (loss starts 0.17 — P1 knowledge
persisted). P1 frozen; identical eval protocol; paired stats vs P1.

## Result: abstention achieved, extraction destroyed

| metric | P1 | P2 | paired Δ (semantic tv) |
| --- | --- | --- | --- |
| no_fact_accuracy (dev) | 0.41 | **0.78** (T4: 0.91) | — target ≥0.70 MET |
| precision / recall (dev) | 0.44 / 0.57 | **0.76 / 0.33** | recall −0.24 |
| T2 unseen semantic F1 | 0.527 | 0.420 | **−0.175 [−0.226,−0.125] SIG** |
| T4 adversarial semantic F1 | 0.535 | 0.343 | **−0.204 [−0.260,−0.148] SIG** (R 0.61→0.23) |
| T3 technical | 0.406 | 0.305 | −0.133 [−0.258,−0.028] SIG |
| dev semantic F1 | 0.516 | 0.455 | −0.125 [−0.188,−0.061] SIG |
| T1 literal (micro) | 0.519 | 0.559 | −0.046 ns (fact-level micro up, per-record down) |
| unsupported rate | 0.10–0.16 | **0.03–0.10** | improved |
| exact-copy rate | 0.84–0.90 | **0.90–0.97** | improved |

## VERDICT: REJECT

Guardrails explicitly violated: T2 semantic regression is material and CI-solid;
T4 crashed instead of remaining stable. The no-fact gain was purchased by a
global under-emission prior — the exact trivial optimum the protocol warned
about (precision solved by refusing to extract).

## Why abstention targeting failed (analysis before any new branch)

1. **Distribution pressure**: 45% empty + subset-gold near-misses (emit 1 of 3–5)
   + a full epoch at unchanged LR = a strong, uniform push toward under-emission.
2. **Capacity/style, not rule-learning**: at 0.5B the abstention decision appears
   largely global (a decision-threshold shift), not schema-conditioned per-field
   judgment — precision and recall moved together everywhere, including suites
   whose data P2 never saw (T3 regression without any T3-like negatives).
3. **No recall anchor**: no replay of pilot1 positives during continuation, so
   nothing opposed the drift.

## Actionable follow-ups (NOT auto-executed)

- Milder knob along the same axis: empty share ~20%, positive replay, shorter
  continuation — walks the P/R curve; intermediate checkpoints (e.g. ckpt_200)
  likely already sit mid-curve. Cheap to probe, owner decides.
- Precision without recall loss more plausibly comes from queue items (g)
  generator+verifier or (n) calibration/threshold than from more SFT pressure.
- The pre-model of record remains **pilot1/ckpt_best** (v0.1.0-pilot1).
