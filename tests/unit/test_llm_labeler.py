import json

import pytest

from trustrec.nlp.llm_labeler import (
    CachedLLMLabeler,
    ChatClient,
    InsufficientBalanceError,
    label_batch,
    load_cache,
    locate_evidence,
    parse_response,
)
from trustrec.nlp.records import source_hash


def _review(review_id: str, text: str) -> dict[str, str]:
    return {"review_id": review_id, "text": text, "source_text_sha256": source_hash(text)}


def test_locate_evidence_returns_exact_source_offsets() -> None:
    text = "The Story is great.\nControls   feel clunky."

    assert locate_evidence(text, "Story is great") == (4, 18)
    start, end = locate_evidence(text, "controls feel clunky")
    assert text[start:end] == "Controls   feel clunky"
    assert locate_evidence(text, "not in the review") is None


def test_parse_response_keeps_valid_labels_and_counts_dropped_ones() -> None:
    reviews = [_review("r1", "Great story, awful camera."), _review("r2", "Fast shipping.")]
    content = json.dumps(
        {
            "reviews": [
                {
                    "id": 0,
                    "out_of_scope": False,
                    "needs_review": False,
                    "labels": [
                        {"aspect": "story", "polarity": "positive", "evidence": "Great story"},
                        {"aspect": "controls", "polarity": "negative", "evidence": "awful camera"},
                        {"aspect": "price", "polarity": "negative", "evidence": "awful camera"},
                        {"aspect": "graphics", "polarity": "positive", "evidence": "made up"},
                    ],
                }
            ]
        }
    )
    first, second = parse_response(content, reviews, max_chars=1000)

    assert [(label["aspect"], label["evidence"]) for label in first.labels] == [
        ("controls", "awful camera"),
        ("story", "Great story"),
    ]
    assert first.dropped_labels == 2
    assert second.error == "missing_in_response"


class _FakeResponse:
    def __init__(self, status: int, payload: dict | None = None) -> None:
        self.status_code = status
        self._payload = payload or {}

    def json(self) -> dict:
        return self._payload


def test_chat_client_retries_and_stops_on_insufficient_balance(monkeypatch) -> None:
    monkeypatch.setattr("time.sleep", lambda _: None)
    calls = []

    def post(*args, **kwargs):
        calls.append(kwargs["json"])
        if len(calls) == 1:
            return _FakeResponse(429)
        return _FakeResponse(
            200, {"choices": [{"message": {"content": '{"reviews": []}'}}], "usage": {}}
        )

    content, _ = ChatClient(api_key="k", model_id="m", post=post).complete("[]")
    assert content == '{"reviews": []}'
    assert calls[0]["temperature"] == 0
    assert calls[0]["reasoning_effort"] == "none"

    broke = ChatClient(api_key="k", model_id="m", post=lambda *a, **k: _FakeResponse(402))
    with pytest.raises(InsufficientBalanceError):
        broke.complete("[]")


def test_label_batch_marks_invalid_json_for_a_later_retry() -> None:
    post = lambda *a, **k: _FakeResponse(  # noqa: E731
        200, {"choices": [{"message": {"content": "not json"}}], "usage": {}}
    )
    results, _ = label_batch(
        ChatClient(api_key="k", model_id="m", post=post), [_review("r", "x")], 100
    )

    assert results[0].error == "invalid_response"


def test_cache_skips_failed_records_and_other_settings(tmp_path) -> None:
    path = tmp_path / "cache.jsonl"
    rows = [
        {
            "review_id": "a",
            "model_id": "m",
            "reasoning_effort": "none",
            "prompt_version": "t6.2-llm-batch-v1",
            "error": None,
        },
        {
            "review_id": "b",
            "model_id": "m",
            "reasoning_effort": "none",
            "prompt_version": "t6.2-llm-batch-v1",
            "error": "invalid_response",
        },
        {
            "review_id": "c",
            "model_id": "m",
            "reasoning_effort": "low",
            "prompt_version": "t6.2-llm-batch-v1",
            "error": None,
        },
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n{partial", encoding="utf-8")

    assert set(load_cache(path, "m", "none")) == {"a"}


def test_cached_labeler_serves_labels_and_refusals() -> None:
    text = "Great story."
    labeler = CachedLLMLabeler(
        {
            "r1": {
                "source_text_sha256": source_hash(text),
                "out_of_scope": False,
                "labels": [
                    {
                        "aspect": "story",
                        "polarity": "positive",
                        "start": 0,
                        "end": 11,
                        "evidence": "Great story",
                    }
                ],
            }
        }
    )

    labeled = labeler.predict(text, review_id="r1")
    assert labeled["labels"][0]["sentiment_score"] == 1.0
    assert labeler.predict(text, review_id="missing")["refusal_reason"] == "not_labeled"
    assert labeler.predict("changed", review_id="r1")["refusal_reason"] == "text_changed"


def test_parse_response_recovers_the_last_complete_object() -> None:
    reviews = [_review("r1", "Great story.")]
    label = {"aspect": "story", "polarity": "positive", "evidence": "Great story"}
    good = json.dumps({"reviews": [{"id": 0, "labels": [label]}]})
    broken = '{"reviews": [{"id": 0, "labels": {"aspect": "story"' + good
    fenced = "```json\n" + good + "\n```"

    for content in (broken, fenced):
        (result,) = parse_response(content, reviews, max_chars=100)
        assert [label["aspect"] for label in result.labels] == ["story"]


def test_sharded_writer_rotates_parts_and_resumes(tmp_path) -> None:
    from trustrec.nlp.llm_labeler import ShardedCacheWriter, cache_files

    record = {"model_id": "m", "reasoning_effort": None, "prompt_version": "t6.2-llm-batch-v1"}
    writer = ShardedCacheWriter(tmp_path, rows_per_part=2)
    for index in range(3):
        writer.write(json.dumps({**record, "review_id": f"r{index}", "error": None}))
    writer.close()
    with (tmp_path / "part-00001.jsonl").open("a") as handle:
        handle.write('{"partial')
    writer = ShardedCacheWriter(tmp_path, rows_per_part=2)
    writer.write(json.dumps({**record, "review_id": "r3", "error": None}))
    writer.close()

    assert [path.name for path in cache_files(tmp_path)] == [
        "part-00000.jsonl",
        "part-00001.jsonl",
        "part-00002.jsonl",
    ]
    assert set(load_cache(tmp_path, "m", None)) == {"r0", "r1", "r2", "r3"}


def test_hf_sync_refuses_public_repos_and_uploads_only_changes(tmp_path, monkeypatch) -> None:
    pytest.importorskip("huggingface_hub")
    from trustrec.data.hf_sync import HFSync

    class FakeApi:
        def __init__(self, private: bool) -> None:
            self.private, self.commits = private, []

        def dataset_info(self, repo_id):
            return type("Info", (), {"private": self.private})()

        def create_commit(self, **kwargs):
            self.commits.append([op.path_in_repo for op in kwargs["operations"]])
            return type("Commit", (), {"oid": "abc"})()

    (tmp_path / "part-00000.jsonl").write_text("{}\n")
    with pytest.raises(RuntimeError, match="public"):
        HFSync("o/r", tmp_path, "llm/m", api=FakeApi(private=False)).push()
    api = FakeApi(private=True)
    sync = HFSync("o/r", tmp_path, "llm/m", api=api)
    assert sync.push()["uploaded"] == ["part-00000.jsonl"]
    assert sync.push()["uploaded"] == []
    (tmp_path / "part-00000.jsonl").write_text("{}\n{}\n")
    assert sync.push()["uploaded"] == ["part-00000.jsonl"]
    assert api.commits == [["llm/m/part-00000.jsonl"], ["llm/m/part-00000.jsonl"]]
