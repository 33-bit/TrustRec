# TrustRec

TrustRec is a capstone research system for evidence-aware product recommendation. It combines historical ratings, aspect-level opinions, and a user–item interaction graph to produce top-K recommendations with traceable evidence.

The repository is intentionally **data-contract first**. The design document is a proposal; no metric, dataset size, or model quality is treated as an achieved result until it is produced by a recorded experiment.

Manual labels are sampled held-out evaluation data. They never train recommendation, aspect extraction, or sentiment models. An optional development/pilot subset may refine guidelines or prompts; the final manual test subset is frozen and used only for final aspect, sentiment, and explanation evaluation. The source remains Amazon Reviews 2023 by McAuley Lab; the whole corpus is not manually labeled.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
make install
make check
make demo
```

The current demo is a small shell for validating the serving contract. Real Amazon Reviews files are not committed. Put local downloads under `data/raw/`, generate a snapshot manifest, and record the dataset hash before running experiments.

## Optional local Laya labeling

Laya is an optional local annotator for silver labels. It downloads a checkpoint from Hugging Face on first inference and then runs locally; no LLM API key is required. Install it separately because PyTorch and model weights are larger than the core test harness:

```bash
make install-laya
PYTHONPATH=src python scripts/run_laya_labeling.py \
  --input docs/tasks/t2_1_ai_pilot_labels.jsonl \
  --output docs/tasks/t2_1_laya_pilot_labels.jsonl \
  --model laya
```

The output is marked `ai_preliminary`. It must be reviewed before becoming gold data. The runner stores confidence, presence probabilities, model name, and the selected sentence/clause as the evidence span.

## Repository map

- `src/trustrec/` — production modules and typed contracts.
- `scripts/` — reproducible command-line entry points.
- `configs/` — versioned protocol and model configuration.
- `tests/` — contract, unit, and integration tests.
- `app/` — Streamlit demonstration.
- `docs/adr/` — architectural decisions.
- `docs/specs/` — stable data and API contracts.
- `docs/plans/` and `docs/tasks/` — execution roadmap and backlog.
- `docs/specs/` — research, data, model, evaluation, serving, and reporting contracts.
- `docs/design/` and `docs/reference/` — source design, terminology, and claims policy.
- `schemas/` — machine-readable artifact and annotation schemas.

Read [AGENTS.md](AGENTS.md) before contributing. The source design is preserved at `docs/superpowers/specs/2026-10-02-trustrec-foundation-design.md`.
