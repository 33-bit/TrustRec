# ADR-0013: LLM-only annotation policy

Status: Accepted. Date: 2026-10-04.

## Decision

Amazon Reviews 2023 by McAuley Lab remains the main data source. An LLM creates aspect, sentiment, and evidence labels. The project does not require human annotation.

The project uses these roles:

| Role | Use |
| --- | --- |
| `llm_silver` | Development labels, NLP features, evidence construction, and optional NLP training. |
| `llm_pseudo_test` | A frozen LLM pass for aspect, sentiment, and evidence consistency checks. |
| `recommendation_test` | The real Amazon temporal test interactions and ratings for recommendation metrics. |

LLM labels are not human gold labels. The project does not report human annotator agreement, Cohen kappa, or ground-truth accuracy for these labels. Reports use pseudo-label agreement or consistency.

The development pilot can refine prompts, guidelines, thresholds, and model settings. Freeze the `llm_pseudo_test` split before any such choice. Do not use the frozen split for tuning, prompt selection, threshold selection, or hyperparameter selection.

Recommendation metrics use `recommendation_test` interactions from the temporal split. NLP and explanation checks use the LLM pseudo-test only. Manual labeling is not required for the corpus.

## Supersession

This ADR supersedes the human-annotation parts of [ADR-0007](0007-silver-gold-test-label-policy.md). ADR-0007 remains unchanged as a historical record. The original design source also remains unchanged.

## Consequences

Every LLM record and manifest stores the source snapshot, snapshot dataset hash, exact cutoff, split role, label status, model revision, prompt version, temperature, seed, generation time, source-text hash, configuration hash, model hash, evidence offsets, and usage limits. Results must name the label source and the limits of LLM agreement.
