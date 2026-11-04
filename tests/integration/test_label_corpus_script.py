import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from scripts.build_evidence_profiles import build_profiles
from scripts.label_corpus_evidence import label_corpus

CUTOFF = "1970-01-01T00:00:00.100000+00:00"
TEXTS = (
    "The story is great. The controls are terrible.",
    "Graphics look beautiful but the game crashes constantly.",
    "Fun gameplay and a lot of content to replay.",
    "Not worth the price.",
)


def _write_snapshot(tmp_path: Path) -> Path:
    interactions, texts = [], []
    for position in range(40):
        review_id = f"r{position:02d}"
        text = TEXTS[position % len(TEXTS)]
        timestamp = position * 4
        texts.append(
            {
                "review_id": review_id,
                "item_id": f"i{position % 5}",
                "text": text,
                "timestamp": timestamp,
                "source_text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "duplicate_group": f"g{position}",
            }
        )
        if timestamp < 100 and position != 3:
            interactions.append(
                {
                    "review_id": review_id,
                    "user_id": f"u{position % 7}",
                    "item_id": f"i{position % 5}",
                    "rating": 5.0,
                    "timestamp": timestamp,
                }
            )
    pq.write_table(pa.Table.from_pylist(interactions), tmp_path / "train.parquet")
    pq.write_table(pa.Table.from_pylist(texts), tmp_path / "texts.parquet")
    manifest = tmp_path / "snapshot.json"
    manifest.write_text(
        json.dumps(
            {
                "snapshot_id": "snap-1",
                "dataset_hash": "d" * 64,
                "cutoffs": {"t0": CUTOFF},
                "artifact_paths": {
                    "train_interactions": "train.parquet",
                    "review_texts": "texts.parquet",
                },
            }
        ),
        encoding="utf-8",
    )
    return manifest


def test_label_corpus_feeds_evidence_profiles(tmp_path: Path) -> None:
    manifest = _write_snapshot(tmp_path)
    output = tmp_path / "evidence.jsonl"
    run_manifest = label_corpus(manifest, cutoff_name="t0", output_path=output, batch_size=7)
    rows = [json.loads(line) for line in output.read_text().splitlines()]

    assert run_manifest["status"] == "complete"
    assert run_manifest["summary"]["reviews"] == 24
    assert rows
    assert all(row["timestamp"] < 100 for row in rows)
    assert "r03" not in {row["review_id"] for row in rows}
    profiles = build_profiles(
        output,
        snapshot_id="snap-1",
        dataset_hash="d" * 64,
        cutoff_timestamp=CUTOFF,
        output_path=tmp_path / "profiles.json",
    )
    assert profiles["profiles"]


def test_label_corpus_workers_match_a_single_process(tmp_path: Path) -> None:
    manifest = _write_snapshot(tmp_path)
    single = tmp_path / "single.jsonl"
    forked = tmp_path / "forked.jsonl"
    label_corpus(manifest, cutoff_name="t0", output_path=single, batch_size=5)
    label_corpus(manifest, cutoff_name="t0", output_path=forked, batch_size=5, workers=2)

    assert single.read_bytes() == forked.read_bytes()


def test_label_corpus_refuses_to_replace_a_completed_manifest(tmp_path: Path) -> None:
    manifest = _write_snapshot(tmp_path)
    output = tmp_path / "evidence.jsonl"
    label_corpus(manifest, cutoff_name="t0", output_path=output)

    with pytest.raises(FileExistsError, match="completed"):
        label_corpus(manifest, cutoff_name="t0", output_path=output)


def test_llm_labeling_resumes_and_feeds_the_evidence_pipeline(tmp_path: Path) -> None:
    from scripts.label_reviews_llm import run_llm_labeling

    from trustrec.nlp.llm_labeler import ChatClient

    manifest = _write_snapshot(tmp_path)
    data = json.loads(manifest.read_text())
    data["cutoffs"]["t1"] = CUTOFF
    data["artifact_paths"]["fit_interactions"] = "train.parquet"
    manifest.write_text(json.dumps(data))
    calls = []

    def post(*args, **kwargs):
        reviews = json.loads(kwargs["json"]["messages"][1]["content"])
        calls.append(len(reviews))
        answer = {
            "reviews": [
                {
                    "id": review["id"],
                    "out_of_scope": False,
                    "needs_review": False,
                    "labels": [
                        {"aspect": "story", "polarity": "positive", "evidence": review["text"][:9]}
                    ],
                }
                for review in reviews
            ]
        }

        class Response:
            status_code = 200

            def json(self):
                return {"choices": [{"message": {"content": json.dumps(answer)}}], "usage": {}}

        return Response()

    client = ChatClient(api_key="k", model_id="fake", post=post)
    cache = tmp_path / "cache.jsonl"
    first = run_llm_labeling(
        manifest,
        cache_path=cache,
        client=client,
        batch_size=5,
        workers=2,
        max_reviews=10,
        min_balance=None,
    )
    second = run_llm_labeling(
        manifest, cache_path=cache, client=client, batch_size=5, workers=2, min_balance=None
    )

    assert first["status"] == "partial"
    assert second["status"] == "complete"
    assert sum(calls) == second["scope_reviews"]
    output = tmp_path / "llm_evidence.jsonl"
    run = label_corpus(
        manifest,
        cutoff_name="t0",
        output_path=output,
        labeler="llm-cache",
        label_cache_path=cache,
        llm_model_id="fake",
    )
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert run["label_status"] == "llm_silver"
    assert rows and all(row["aspect"] == "story" for row in rows)


def test_llm_labeling_cools_down_and_retries_failed_batches(tmp_path: Path, monkeypatch) -> None:
    from scripts.label_reviews_llm import run_llm_labeling

    from trustrec.nlp.llm_labeler import ChatClient

    monkeypatch.setattr("time.sleep", lambda _: None)
    manifest = _write_snapshot(tmp_path)
    data = json.loads(manifest.read_text())
    data["cutoffs"]["t1"] = CUTOFF
    data["artifact_paths"]["fit_interactions"] = "train.parquet"
    manifest.write_text(json.dumps(data))
    calls = {"count": 0}

    def post(*args, **kwargs):
        calls["count"] += 1
        reviews = json.loads(kwargs["json"]["messages"][1]["content"])

        class Response:
            status_code = 503 if calls["count"] <= 4 else 200

            def json(self):
                answer = {"reviews": [{"id": r["id"], "labels": []} for r in reviews]}
                return {"choices": [{"message": {"content": json.dumps(answer)}}], "usage": {}}

        return Response()

    client = ChatClient(api_key="k", model_id="fake", post=post, max_retries=0)
    run = run_llm_labeling(
        manifest,
        cache_path=tmp_path / "cache",
        client=client,
        batch_size=5,
        workers=1,
        min_balance=None,
        max_consecutive_failures=2,
        cooldown_s=0.0,
    )

    assert run["status"] == "complete"
    assert run["this_run"]["usage"]["failed_requests"] == 4


def test_curated_llm_labels_feed_the_evidence_pipeline(tmp_path: Path) -> None:
    from scripts.curate_llm_labels import curate_llm_labels
    from scripts.label_reviews_llm import run_llm_labeling

    from trustrec.nlp.llm_labeler import ChatClient

    manifest = _write_snapshot(tmp_path)
    data = json.loads(manifest.read_text())
    data["cutoffs"]["t1"] = CUTOFF
    data["artifact_paths"]["fit_interactions"] = "train.parquet"
    manifest.write_text(json.dumps(data))

    def post(*args, **kwargs):
        reviews = json.loads(kwargs["json"]["messages"][1]["content"])
        answer = {
            "reviews": [
                {
                    "id": review["id"],
                    "out_of_scope": False,
                    "needs_review": False,
                    "labels": [
                        {"aspect": "story", "polarity": "positive", "evidence": review["text"][:9]},
                        {"aspect": "story", "polarity": "positive", "evidence": review["text"][:4]},
                    ],
                }
                for review in reviews
            ]
        }

        class Response:
            status_code = 200

            def json(self):
                return {"choices": [{"message": {"content": json.dumps(answer)}}], "usage": {}}

        return Response()

    cache = tmp_path / "cache"
    cache.mkdir()
    client = ChatClient(api_key="k", model_id="fake", post=post, reasoning_effort=None)
    run_llm_labeling(manifest, cache_path=cache, client=client, batch_size=5, min_balance=None)
    raw_run = label_corpus(
        manifest,
        cutoff_name="t0",
        output_path=tmp_path / "raw.jsonl",
        labeler="llm-cache",
        label_cache_path=cache,
        llm_model_id="fake",
        llm_reasoning_effort=None,
    )
    curated = curate_llm_labels(
        manifest,
        cache_path=cache,
        model_id="fake",
        reasoning_effort=None,
        output_dir=tmp_path / "curated",
    )
    curated_run = label_corpus(
        manifest,
        cutoff_name="t0",
        output_path=tmp_path / "curated.jsonl",
        labeler="llm-cache",
        label_cache_path=tmp_path / "curated",
    )

    assert curated["status"] == "complete"
    assert curated["counts"]["labels"] == curated["scope_reviews"]
    assert curated["raw_cache"]["label_actions"] == {"merged_overlap": curated["scope_reviews"]}
    assert raw_run["summary"]["evidence_rows"] == 2 * curated_run["summary"]["evidence_rows"]
    assert curated_run["label_source"].endswith(":llm-curation-v1")
    rows = [json.loads(line) for line in (tmp_path / "curated.jsonl").read_text().splitlines()]
    assert {row["evidence"] for row in rows} == {"Fun gamep", "Graphics", "Not worth", "The story"}
