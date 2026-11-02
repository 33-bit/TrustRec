# ADR-0015: Rule-based evidence for the full review corpus

Status: Proposed. Date: 2026-11-02.

## Context

ADR-0013 says an LLM creates aspect, sentiment, and evidence labels. The
evidence profiles behind H0 and T0 need labels for every pre-cutoff review:
about 1.37 million reviews before `T0` and 1.55 million before `T1`. Running an
LLM over every review has a real cost, and that choice belongs to the project
owners. The repository already has a deterministic dictionary baseline (T2.3).
On the frozen pseudo-test it reaches aspect micro F1 0.5085 against LLM labels.

## Decision

`scripts/label_corpus_evidence.py` labels the corpus through a labeler
interface (`trustrec.nlp.corpus.ReviewLabeler`). The first labeler is the T2.3
dictionary baseline with label source `dictionary-evidence-v1` and label status
`rule_baseline`.

- The labeler reads only review text. It never reads ratings.
- A review is labeled only when its ID is in the cutoff's interaction table and
  its timestamp is before the cutoff.
- Evidence offsets refer to the original text, and the evidence text equals that
  slice. Each review's text hash must match `source_text_sha256`.
- The manifest records the snapshot, cutoff, source hashes, a hash of every
  dictionary rule, counts, refusal reasons, and the output hash.

## Open decision

The project owners decide whether `rule_baseline` evidence may feed reported
T0 results, or whether an LLM labeler must replace it first. An LLM labeler
plugs into the same interface. It must record the ADR-0013 provenance fields,
and its run must not use the frozen `llm_pseudo_test` split for tuning.

## Consequences

Reports that use this evidence must name `rule_baseline` as the label source
and state the T2.3 consistency limits. These labels are not LLM labels and not
human gold labels.
