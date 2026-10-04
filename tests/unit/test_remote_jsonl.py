import hashlib

from trustrec.data import remote_jsonl


def test_full_scan_records_chunk_and_source_hashes(monkeypatch) -> None:
    payload = b'{"id": 1}\n{"id": 2}\n'
    monkeypatch.setattr(remote_jsonl, "source_size", lambda url: len(payload))
    monkeypatch.setattr(
        remote_jsonl,
        "fetch_window",
        lambda url, start, end: (payload[start : end + 1], len(payload)),
    )
    provenance = {}

    rows = list(remote_jsonl.iter_jsonl_records("https://example.test/data.jsonl", 8, provenance))

    assert rows == [{"id": 1}, {"id": 2}]
    assert provenance["scan"] == "full_sequential_range_scan"
    assert provenance["source_bytes"] == len(payload)
    assert provenance["source_sha256"] == hashlib.sha256(payload).hexdigest()
    assert len(provenance["chunk_hashes"]) == 3
