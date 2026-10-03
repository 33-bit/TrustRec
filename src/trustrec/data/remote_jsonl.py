"""Deterministic byte-window access to large public JSONL sources."""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any
from urllib.request import Request, urlopen

DEFAULT_WINDOW_BYTES = 32 * 1024 * 1024
DEFAULT_WINDOW_FRACTIONS = (0.0, 0.5, 1.0)


def fetch_window(url: str, start: int, end: int) -> tuple[bytes, int]:
    request = Request(url, headers={"Range": f"bytes={start}-{end}"})
    with urlopen(request, timeout=120) as response:  # noqa: S310 - fixed HTTPS source
        payload = response.read()
        content_range = response.headers.get("Content-Range", "")
        total = (
            int(content_range.rsplit("/", 1)[1])
            if "/" in content_range
            else int(response.headers["Content-Length"])
        )
    return payload, total


def source_size(url: str) -> int:
    """Return the remote file size using a one-byte range request."""

    _, total = fetch_window(url, 0, 0)
    return total


def iter_jsonl_records(
    url: str, chunk_bytes: int = DEFAULT_WINDOW_BYTES
) -> Iterator[dict[str, Any]]:
    """Stream every complete JSONL record through sequential HTTP ranges."""

    total = source_size(url)
    remainder = b""
    for start in range(0, total, chunk_bytes):
        payload, _ = fetch_window(url, start, min(total - 1, start + chunk_bytes - 1))
        lines = (remainder + payload).split(b"\n")
        remainder = lines.pop()
        for line in lines:
            if line.strip():
                yield json.loads(line)
    if remainder.strip():
        yield json.loads(remainder)


def sample_records(
    url: str,
    window_bytes: int = DEFAULT_WINDOW_BYTES,
    window_fractions: tuple[float, ...] = DEFAULT_WINDOW_FRACTIONS,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return complete JSONL records from fixed byte windows and provenance."""

    total = source_size(url)
    windows: list[tuple[int, bytes]] = []
    for fraction in window_fractions:
        start = max(0, min(total - window_bytes, int(total * fraction)))
        if any(existing_start == start for existing_start, _ in windows):
            continue
        payload, _ = fetch_window(url, start, min(total - 1, start + window_bytes - 1))
        windows.append((start, payload))

    records: list[dict[str, Any]] = []
    for start, payload in windows:
        chunk = payload
        if start > 0:
            _, _, chunk = chunk.partition(b"\n")
        if start + len(payload) < total:
            chunk = chunk.rsplit(b"\n", 1)[0]
        for line in chunk.splitlines():
            if line.strip():
                records.append(json.loads(line))

    return records, {
        "source_url": url,
        "source_bytes": total,
        "window_bytes": window_bytes,
        "window_count": len(windows),
        "window_starts": [start for start, _ in windows],
    }
