import json
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from trustrec.nlp.llm_curation import (
    CurationStats,
    clean_labels,
    collapse_cache,
    curate_records,
    is_curated,
    load_curated,
    read_curated_manifest,
    write_curated,
)
from trustrec.nlp.llm_labeler import PROMPT_VERSION, CachedLLMLabeler
from trustrec.nlp.records import source_hash


def _label(text: str, evidence: str, aspect: str, polarity: str, start: int | None = None):
    start = text.index(evidence) if start is None else start
    return {
        "aspect": aspect,
        "polarity": polarity,
        "start": start,
        "end": start + len(evidence),
        "evidence": evidence,
    }


def _record(review_id: str, text: str, labels=(), **extra):
    return {
        "review_id": review_id,
        "source_text_sha256": source_hash(text),
        "model_id": "m",
        "reasoning_effort": None,
        "prompt_version": PROMPT_VERSION,
        "out_of_scope": False,
        "needs_review": False,
        "labels": list(labels),
        "dropped_labels": 0,
        "truncated": False,
        "error": None,
        **extra,
    }


def test_clean_labels_trims_neutralizes_merges_and_sorts() -> None:
    text = "Fun but hard. Great art, artwork is fine. Runs badly "
    actions: Counter[str] = Counter()
    labels = [
        _label(text, "Runs badly ", "performance", "negative"),
        _label(text, "Fun but hard.", "gameplay", "positive"),
        _label(text, "Fun but hard.", "gameplay", "negative"),
        _label(text, "Great art, artwork", "graphics", "positive"),
        _label(text, "artwork is fine", "graphics", "positive"),
        _label(text, "artwork", "graphics", "neutral"),
    ]

    cleaned = clean_labels(text, labels, actions)

    assert [(c["aspect"], c["polarity"], c["evidence"]) for c in cleaned] == [
        ("gameplay", "neutral", "Fun but hard."),
        ("graphics", "positive", "Great art, artwork is fine"),
        ("graphics", "neutral", "artwork"),
        ("performance", "negative", "Runs badly"),
    ]
    assert all(text[c["start"] : c["end"]] == c["evidence"] for c in cleaned)
    assert actions == {
        "trimmed_whitespace": 1,
        "neutralized_mixed_polarity": 1,
        "merged_overlap": 1,
    }


def test_clean_labels_rejects_evidence_that_does_not_match_offsets() -> None:
    label = {"aspect": "story", "polarity": "positive", "start": 0, "end": 4, "evidence": "nope"}
    with pytest.raises(ValueError, match="offsets"):
        clean_labels("Story is great.", [label], Counter())


def test_collapse_cache_keeps_the_last_success_per_review(tmp_path: Path) -> None:
    text = "Great story."
    good = _record("r1", text, [_label(text, "Great story", "story", "positive")])
    lines = [
        _record("r1", text, error="missing_in_response"),
        _record("r2", text, model_id="other"),
        good,
        {**good, "needs_review": True},
    ]
    (tmp_path / "part-00000.jsonl").write_text(
        "\n".join(json.dumps(line) for line in lines) + "\n{partial"
    )
    stats = CurationStats()

    records = collapse_cache(tmp_path, "m", None, stats)

    assert set(records) == {"r1"} and records["r1"]["needs_review"] is True
    assert (stats.raw_records, stats.error_records, stats.other_setting_records) == (4, 1, 1)
    assert (stats.superseded_records, stats.invalid_json_lines) == (1, 1)


def test_curated_tables_round_trip_into_the_cached_labeler(tmp_path: Path) -> None:
    texts = {
        "a": "Great story but the controls are bad.",
        "b": "Fast shipping.",
        "c": "   ",
        "d": "Nothing specific here.",
        "e": "Text that changed.",
    }
    records = {
        "a": _record(
            "a",
            texts["a"],
            [
                _label(texts["a"], "the controls are bad", "controls", "negative"),
                _label(texts["a"], "Great story", "story", "positive"),
            ],
        ),
        "b": _record("b", texts["b"], out_of_scope=True),
        "c": _record("c", texts["c"], needs_review=True),
        "d": _record("d", texts["d"], needs_review=True),
        "e": _record("e", "Original text."),
        "z": _record("z", "Not in scope."),
    }
    stats = CurationStats()
    review_rows, label_rows = curate_records(records, texts, stats)
    output = tmp_path / "curated"

    manifest = write_curated(output, review_rows, label_rows, {"model_id": "m"})

    assert [row["status"] for row in review_rows] == [
        "labeled",
        "out_of_scope",
        "empty_text",
        "no_aspect",
    ]
    assert (stats.text_changed, stats.out_of_scope_ids, stats.labels) == (1, 1, 2)
    assert manifest["counts"]["statuses"] == {
        "labeled": 1,
        "no_aspect": 1,
        "out_of_scope": 1,
        "empty_text": 1,
    }
    labels = pq.read_table(output / "labels.parquet").to_pylist()
    assert [(row["label_index"], row["aspect"]) for row in labels] == [
        (0, "story"),
        (1, "controls"),
    ]
    assert "author_id" not in pq.read_schema(output / "reviews.parquet").names
    assert is_curated(output)
    labeler = CachedLLMLabeler(load_curated(output))
    prediction = labeler.predict(texts["a"], review_id="a")
    assert [label["evidence"] for label in prediction["labels"]] == [
        "Great story",
        "the controls are bad",
    ]
    assert labeler.predict(texts["b"], review_id="b")["refusal_reason"] == "out_of_scope"
    assert labeler.predict(texts["d"], review_id="d")["refusal_reason"] == "no_explicit_aspect"

    (output / "labels.parquet").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="hash"):
        read_curated_manifest(output)


def test_hf_sync_uploads_only_files_that_match_its_patterns(tmp_path: Path) -> None:
    pytest.importorskip("huggingface_hub")
    from trustrec.data.hf_sync import HFSync

    class FakeApi:
        commits: list[list[str]] = []

        def dataset_info(self, repo_id):
            return type("Info", (), {"private": True})()

        def create_commit(self, **kwargs):
            self.commits.append(sorted(op.path_in_repo for op in kwargs["operations"]))
            return type("Commit", (), {"oid": "abc"})()

    for name in ("labels.parquet", "reviews.parquet", "manifest.json", "notes.txt"):
        (tmp_path / name).write_text("x")
    api = FakeApi()
    sync = HFSync("o/r", tmp_path, "curated/m", api=api, patterns=("*.parquet", "manifest.json"))

    assert sync.push()["uploaded"] == ["labels.parquet", "manifest.json", "reviews.parquet"]
    assert api.commits == [
        ["curated/m/labels.parquet", "curated/m/manifest.json", "curated/m/reviews.parquet"]
    ]
