# glm53-mws-pretrain-lab

Pre-spec R&D lab for **MWS / CompactFactExtraction** (ИННОГЛОБАЛХАК-2026, МТС case).

**One empirical question:** can a compact model (Qwen2.5-0.5B-Instruct) be adapted to become
substantially better at **grounded, exact, schema-conditioned fact extraction** in technical
text — including schemas it has never seen — without becoming dependent on any invented
ontology?

We train a **skill**, not a guessed MWS task:

```text
TEXT + SCHEMA/FIELD DEFINITIONS + INSTRUCTIONS → SUPPORTED STRUCTURED FACTS
```

The schema defines what counts as a fact. The model must follow the schema.

## Status

Phase 0 — environment + skeleton (this commit). See `docs/EXPERIMENT_PLAN.md` for the roadmap
and `docs/DATA_DESIGN.md` for dataset/eval design.

## Ground rules (non-negotiable)

- Reference repo `mws-for-mts-hackaton` @ `1cd21a3` is READ-ONLY (source of verified
  contracts: schemas, local-only HF loading, PEFT recipe).
- Evaluation suites are frozen BEFORE training. A **sealed final test** exists and is only
  run at milestone gates.
- Gold labels are derived mechanically from latent world-state + placeholder substitution,
  never from teacher self-annotation. Uncertain examples go to quarantine.
- Abstention (return no fact) is a first-class behaviour; hard negatives are a large share
  of training data.
- No vLLM / constrained decoding / quantization / model zoo at this stage.
- The pre-model is **disposable**: if it does not help the official objective within hours
  of the hackathon starting, it gets thrown away (`docs/48H_HANDOFF.md`).

## Layout

```text
docs/          experiment plan, data design, results, 48h handoff
src/compactex  data | synthetic | evaluation | training | analysis
benchmarks/    literal/ unseen_schema/ technical/ adversarial/ (frozen suites + manifests)
experiments/   run ledger (runs.jsonl) + per-run artifacts
models/        base weights provenance (weights via git-lfs, see models/README.md)
```

## Student model

`Qwen/Qwen2.5-0.5B-Instruct` @ `7ae557604adf67be50417f59c2c2f167def9a775` — the exact
revision already runtime-verified in the reference repo (MPS/BF16, LoRA smoke passed).
See `models/README.md`.

## License / usage

Adapter produced here is labeled **R&D ONLY — external synthetic training** until official
rules are known. Do not submit anything from this repo without checking organizer policy.
