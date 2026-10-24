# Experiment plan

One question: **does schema-conditioned synthetic SFT teach a compact model the *skill* of
grounded exact extraction — measurable on schemas it has never seen — rather than an
ontology?**

## Hypothesis

- **H1 (main):** LoRA SFT of Qwen2.5-0.5B-Instruct on `TEXT + SCHEMA → SUPPORTED FACTS`
  synthetic data (mixed schemas, mechanical gold, ~30% hard negatives) improves exact fact
  F1 on **unseen schemas** (T2), adversarial selection (T4) and technical-domain holdout
  (T3), without degrading exact-copy accuracy, unsupported-fact rate, or no-fact accuracy.
- **H0:** gains appear only on seen schema labels / seen template families → the model
  memorized our ontology; the pre-model is rejected as an MWS initializer (the negative
  result still buys us: known failure modes, hardened eval, data factory).

## Design

```text
freeze evals (T1/T2/T3/T4 + SEALED)
→ base Qwen baseline on every suite (provenance-locked)
→ gates: smoke → tiny-overfit (2 examples, loss→~0, exact regeneration)
→ pilot LoRA SFT on 1–3k examples
→ eval on DEV + DEV-TEST suites; bootstrap CIs
→ KEEP/REJECT gate
→ only if KEEP: error-driven batch 2, ablations, scale decisions
→ SEALED run exactly once per milestone (never for iteration)
```

Canonical chat contract (frozen for all experiments):

- system: extraction instructions + **the schema of this task** as named fields with
  natural-language definitions (schema is data, not identity);
- user: JSON `{"ticket_id", "messages": [{"id","role","content"}]}`;
- assistant target: `{"facts": [{"type": <field>, "value": <exact substring>,
  "evidence": {"message_id","start","end"}}]}`, `{"facts": []}` when nothing applies.

Decoding: greedy, `enable_thinking=False`, fixed seed. Any change to prompts/decoding
re-runs the baseline and is labelled in the ledger.

## Metrics

Per suite and per slice (lang: ru / en / mixed; schema: seen / unseen; template family:
seen / held-out):

| metric | definition |
| --- | --- |
| json_valid | output parses as JSON object |
| schema_valid | parses into the canonical FactPayload, fields/enums correct |
| fact P / R / F1 (exact) | fact = (type, value, message_id, start, end); exact set match |
| value exact-copy | predicted value is a literal substring of the cited message |
| evidence correct | text-span consistency: content[start:end] contains value |
| unsupported-fact rate | predicted facts with no supporting span in source |
| no-fact accuracy | empty-gold records answered with empty facts |
| duplicate rate | repeated fact keys in one prediction |

Decision metrics (in priority order): **unseen-schema F1**, exact-copy accuracy,
unsupported-fact rate, technical-domain F1. No single weighted score; Pareto view.

## Uncertainty

Bootstrap 95% CI (percentile, B=1000, record-count resampling) on every reported metric;
**paired bootstrap** on base-vs-tuned deltas. A delta whose CI crosses 0 is INCONCLUSIVE,
not a win. Suites are sized for signal: T1/T2/T4 target ≥200 records each so that ±0.05 F1
deltas are resolvable.

## Experiment contract (every non-trivial run)

```text
id / hypothesis / change / expected / decision rule / cost
→ result / KEEP|REJECT|INCONCLUSIVE / reason / next
```

Logged in `experiments/runs.jsonl` (append-only) + per-run dir.

## KEEP / REJECT gate (first pilot)

KEEP requires ALL of:

1. unseen-schema F1 (T2): paired-bootstrap CI of (tuned − base) excludes 0, positive;
2. ≥1 more independent suite improves beyond CI (T4 or T3);
3. no material degradation: exact-copy acc, unsupported rate, no-fact accuracy
   (material = CI-backed worsening > 0.02 absolute on any);
4. tiny-overfit gate passed before the pilot is trusted.

REJECT if gains concentrate on seen labels/families, or SEALED contradicts DEV-TEST
direction (investigate leakage before any claim).

## Scale-stop rules

Stop or change strategy when: two consecutive scale-ups fail to improve independent
holdouts; gains fall inside CI; unseen-schema stops improving; unsupported rate worsens
materially; capacity is the demonstrated bottleneck (then — one challenger, §50 mission).

## Stage budget discipline

Stage 1 target: single-digit millions of GLM tokens (design + factory + audits). Stage 2
only after KEEP. Reserve preserved for official MWS data. Ledger in
`experiments/token_ledger.jsonl`.

## Non-goals (this stage)

No vLLM/constrained decoding; no quantization; no model zoo; no RL (permitted only after
stable SFT + KEEP); no official-MWS guessing; reference repo stays untouched.

## Research queue (owner-approved, one at a time, evidence-gated)

1. [RUNNING] Exp2 — abstention/hard-negative batch2 (P1+batch2 vs P1).
2. Exp3 — (d) natural-language breadth: GLM renderer over latent world-state + mechanical
   gold (crooked RU, mixed RU/EN, logs, stack traces, support dialogues, command scraps,
   doc paste-through). Strongest candidate after abstention.
3. (e) counterfactual training: flip one fact, only corresponding output changes.
4. (f) hard candidate ranking / contrastive auxiliary.
5. (g) generator + small verifier (SUPPORTED/UNSUPPORTED/WRONG_VALUE) — strong vs over-extraction.
6. (i) custom objective — only for a concrete surviving failure (copy/abstention/contrastive).
7. (j) active learning loop over a large synthetic pool (uncertainty/conflict selection).
8. (k) GLM distillation with deterministic verifier selection; gold stays mechanical where possible.
9. (l) RL/GRPO — only when SFT saturates; reward from type/value F1 + abstention + unsupported.
10. (m) ONE challenger model — only if Qwen-0.5B proven capacity-limited.
11. (n) calibration/confidence thresholds once the official metric's precision/recall orientation is known.
12. (h) offsets-as-postprocessing redesign + (o) few-shot adaptation study (1/5/10%) — AFTER official spec.
