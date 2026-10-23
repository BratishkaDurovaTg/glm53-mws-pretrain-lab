# 48-hour handoff: what to do the moment official MWS materials arrive

Owner: hackathon team. Source of truth: this repo (pre-model lab) + the reference
repo `mws-for-mts-hackaton` @ `1cd21a3` (stable platform, READ-ONLY during prep).

## Assets produced pre-spec

- `experiments/runs/pilot1/ckpt_best` (+ early/final) — LoRA adapter, R&D ONLY
  (external synthetic training), disposable by design.
- Frozen suites: T1 literal / T2 unseen-schema / T3 technical / T4 adversarial,
  dev side + SEALED side (`benchmarks/`), hashes in manifests.
- Base baseline + tuned metrics: `experiments/baseline_base_qwen/`,
  `experiments/eval_*/`, sealed milestone runs `experiments/sealed_runs/`.
- The lab question answered: see `docs/RESULTS.md` (does schema-conditioned SFT
  transfer to unseen schemas — yes/no, with CIs).

## 0–30 minutes: read and archive official materials

Archive verbatim: schema, scorer, examples, runtime limits, external-data and
model rules, submission format. Then diff against pretraining assumptions:

| assumption we made | check against official |
| --- | --- |
| input = ticket with messages [{id, role, content}] | official input format |
| task = extract facts per given field definitions | official fact ontology |
| output = {facts:[{type,value,evidence{message_id,start,end}}]} | official submission format |
| char offsets, end-exclusive, per message | official span convention (tokens? bytes? global?) |
| greedy, no thinking, local weights | allowed decoding / runtime limits |
| synthetic external data allowed for OUR adapter | external-data policy — if forbidden, DROP the adapter, keep the knowledge |

## 30 min – 2 h: stable platform does official integration

In the reference repo: official adapter (their format → canonical ticket),
official scorer wrapper, EDA, frozen split. Do NOT touch this lab's frozen
benchmarks to make them look like official data.

## 2–4 h: the critical test (one command each)

Using `scripts/transfer/compare_official.py` (thin boundary — implement the two
functions in `scripts/transfer/official_boundary.py` first, see its docstring):

```bash
# A. base zero-shot on official dev
uv run --extra ml python scripts/transfer/compare_official.py --arm base-zeroshot
# B. pre-model zero-shot (adapter = experiments/runs/pilot1/ckpt_best)
uv run --extra ml python scripts/transfer/compare_official.py --arm premodel-zeroshot --adapter experiments/runs/pilot1/ckpt_best
# C/D. official SFT from base / from pre-model (equal budget, same recipe)
uv run --extra ml python scripts/transfer/compare_official.py --arm base-sft --steps <N>
uv run --extra ml python scripts/transfer/compare_official.py --arm premodel-sft --adapter experiments/runs/pilot1/ckpt_best --steps <N>
```

All arms: identical official dev split, identical scorer, identical decoding.

## Pre-model decision (be brutal)

- pre-model beats base on official dev → KEEP as initialization.
- no difference → prefer base (simpler).
- worse or slower to adapt → REJECT NOW. Sunk cost is not an argument.

Also try `ckpt_early` vs `ckpt_best` vs `ckpt_final` as SFT initialization if the
differences on dev are material (§62 of the mission).

## 4–8 h: first official SFT with gates

Reference repo runbook: prompt-only baseline → smoke → tiny-overfit → official
SFT. Checkpoint selection by OFFICIAL dev extraction metric, not loss.

## Rest of the hackathon

Error-driven loop against the real objective. Synthetic augmentation ONLY for
measured official failures and ONLY if the rules allow external data.

## What NOT to trust from the lab

Every number here is synthetic/domain-proxy. Trust only: the direction of
base-vs-premodel deltas measured on OFFICIAL data with the OFFICIAL scorer.
The sealed-suite verdict is a leakage check, not a leaderboard.
