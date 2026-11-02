"""Label review aspects with an OpenAI-compatible chat model and cache the results."""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .records import ASPECTS, SENTIMENTS

PROMPT_VERSION = "t6.2-llm-batch-v1"
SYSTEM_PROMPT = """You label source reviews from Amazon Reviews 2023.

Use only claims that a review states. Do not infer a label from the star rating.

Use these aspects:

- `gameplay`: mechanics, pacing, difficulty, and play experience.
- `story`: plot, characters, dialogue, campaign, and narrative.
- `graphics`: visuals, art, animation, and presentation.
- `performance`: bugs, crashes, freezes, lag, loading, and stability.
- `controls`: input, mapping, responsiveness, camera, and handling.
- `multiplayer`: online play, local play, matchmaking, and player interaction.
- `content_replay`: content amount, length, endgame, and replay value.
- `value`: price, cost, worth, and value for money or time.

Use `out_of_scope = true` for accessories, consoles, controllers, cables, headsets, sellers,
shipping, packaging, subscriptions, or unrelated hardware. Return no aspect labels for an
out-of-scope review.

For each explicit claim, return its aspect, `positive`, `negative`, or `neutral` polarity, and
an evidence string copied exactly, character for character, from that review. Return
`needs_review = true` when a review is unclear, too short, or lacks an explicit aspect claim.

You receive a JSON list of reviews with `id` and `text`. Return one JSON object:
{"reviews": [{"id": <id>, "out_of_scope": <bool>, "needs_review": <bool>,
"labels": [{"aspect": <aspect>, "polarity": <polarity>, "evidence": <exact substring>}]}]}
Include exactly one entry for every input id. Do not include a summary or a star-rating label."""

RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}


def prompt_hash() -> str:
    """Hash the prompt text and version recorded with every label."""

    return hashlib.sha256(f"{PROMPT_VERSION}\n{SYSTEM_PROMPT}".encode()).hexdigest()


def locate_evidence(text: str, evidence: str) -> tuple[int, int] | None:
    """Find evidence in the original text and return offsets of the exact source slice.

    An exact match wins. Otherwise a case- and whitespace-insensitive match is
    accepted, and the returned offsets point at the original characters.
    """

    if not isinstance(evidence, str):
        return None
    candidate = evidence.strip().strip("\"'")
    if not candidate:
        return None
    position = text.find(candidate)
    if position >= 0:
        return position, position + len(candidate)
    pattern = r"\s+".join(re.escape(part) for part in candidate.split())
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if match:
        return match.start(), match.end()
    return None


@dataclass
class ReviewResult:
    """Validated labels for one review from one model response."""

    review_id: str
    source_text_sha256: str
    out_of_scope: bool
    needs_review: bool
    labels: list[dict[str, Any]]
    dropped_labels: int = 0
    truncated: bool = False
    error: str | None = None

    def to_record(
        self, model_id: str, request_id: str, reasoning_effort: str | None = None
    ) -> dict[str, Any]:
        return {
            "review_id": self.review_id,
            "source_text_sha256": self.source_text_sha256,
            "model_id": model_id,
            "reasoning_effort": reasoning_effort,
            "prompt_version": PROMPT_VERSION,
            "out_of_scope": self.out_of_scope,
            "needs_review": self.needs_review,
            "labels": self.labels,
            "dropped_labels": self.dropped_labels,
            "truncated": self.truncated,
            "error": self.error,
            "request_id": request_id,
            "generated_at": datetime.now(UTC).isoformat(),
        }


def extract_json_object(content: str) -> Any:
    """Parse a JSON response, or the last complete object with a ``reviews`` key.

    Some OpenAI-compatible gateways wrap JSON in code fences or emit an
    interrupted attempt before the full answer. Plain JSON parses directly.
    """

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    found: Any = None
    for match in re.finditer(r"\{\s*\"reviews\"", content):
        try:
            value, _ = decoder.raw_decode(content, match.start())
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and isinstance(value.get("reviews"), list):
            found = value
    if found is None:
        raise ValueError("response has no complete JSON object with reviews")
    return found


def parse_response(
    content: str, reviews: Sequence[Mapping[str, Any]], max_chars: int
) -> list[ReviewResult]:
    """Validate a batch response against the reviews that were sent."""

    payload = extract_json_object(content)
    entries = payload.get("reviews") if isinstance(payload, dict) else None
    if not isinstance(entries, list):
        raise ValueError("response has no reviews list")
    by_id: dict[int, Mapping[str, Any]] = {}
    for entry in entries:
        if isinstance(entry, dict) and isinstance(entry.get("id"), int):
            by_id.setdefault(entry["id"], entry)
    results: list[ReviewResult] = []
    for position, review in enumerate(reviews):
        text = review["text"]
        entry = by_id.get(position)
        base = {
            "review_id": review["review_id"],
            "source_text_sha256": review["source_text_sha256"],
            "truncated": len(text) > max_chars,
        }
        if entry is None:
            results.append(
                ReviewResult(
                    **base,
                    out_of_scope=False,
                    needs_review=True,
                    labels=[],
                    error="missing_in_response",
                )
            )
            continue
        out_of_scope = bool(entry.get("out_of_scope", False))
        labels: list[dict[str, Any]] = []
        dropped = 0
        raw_labels = entry.get("labels") if isinstance(entry.get("labels"), list) else []
        for label in raw_labels:
            if not isinstance(label, dict):
                dropped += 1
                continue
            aspect, polarity = label.get("aspect"), label.get("polarity")
            span = locate_evidence(text, label.get("evidence"))
            if aspect not in ASPECTS or polarity not in SENTIMENTS or span is None:
                dropped += 1
                continue
            start, end = span
            labels.append(
                {
                    "aspect": aspect,
                    "polarity": polarity,
                    "start": start,
                    "end": end,
                    "evidence": text[start:end],
                }
            )
        if out_of_scope:
            dropped += len(labels)
            labels = []
        unique = {(lab["aspect"], lab["polarity"], lab["start"], lab["end"]): lab for lab in labels}
        results.append(
            ReviewResult(
                **base,
                out_of_scope=out_of_scope,
                needs_review=bool(entry.get("needs_review", False)),
                labels=[unique[key] for key in sorted(unique)],
                dropped_labels=dropped,
            )
        )
    return results


class InsufficientBalanceError(RuntimeError):
    """The provider refused a request because the account has no balance."""


@dataclass
class ChatClient:
    """Minimal OpenAI-compatible chat client with retries."""

    api_key: str
    model_id: str
    base_url: str = "https://api.deepseek.com"
    reasoning_effort: str | None = "none"
    timeout: float = 180.0
    max_retries: int = 6
    post: Callable[..., Any] | None = None

    def complete(self, user_content: str) -> tuple[str, dict[str, Any]]:
        import requests

        post = self.post or requests.post
        body: dict[str, Any] = {
            "model": self.model_id,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
        }
        if self.reasoning_effort:
            body["reasoning_effort"] = self.reasoning_effort
        delay = 2.0
        for attempt in range(self.max_retries + 1):
            try:
                response = post(
                    f"{self.base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json=body,
                    timeout=self.timeout,
                )
            except Exception as error:  # network errors are retried
                if attempt == self.max_retries:
                    raise RuntimeError(f"request failed: {error}") from error
                time.sleep(delay)
                delay = min(delay * 2, 60.0)
                continue
            if response.status_code == 402:
                raise InsufficientBalanceError("provider reports insufficient balance")
            if response.status_code in RETRYABLE_STATUS and attempt < self.max_retries:
                time.sleep(delay)
                delay = min(delay * 2, 60.0)
                continue
            if response.status_code != 200:
                raise RuntimeError(f"request failed with status {response.status_code}")
            payload = response.json()
            return payload["choices"][0]["message"]["content"], payload.get("usage", {})
        raise RuntimeError("request retries exhausted")


@dataclass
class UsageTotals:
    requests: int = 0
    reviews: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    cache_hit_tokens: int = 0
    failed_requests: int = 0
    errors: dict[str, int] = field(default_factory=dict)

    def add(self, usage: Mapping[str, Any], reviews: int) -> None:
        self.requests += 1
        self.reviews += reviews
        self.prompt_tokens += int(usage.get("prompt_tokens", 0))
        self.completion_tokens += int(usage.get("completion_tokens", 0))
        details = usage.get("completion_tokens_details") or {}
        self.reasoning_tokens += int(details.get("reasoning_tokens", 0))
        self.cache_hit_tokens += int(usage.get("prompt_cache_hit_tokens", 0))

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def label_batch(
    client: ChatClient, reviews: Sequence[Mapping[str, Any]], max_chars: int
) -> tuple[list[ReviewResult], dict[str, Any]]:
    """Send one batch and return validated results and token usage.

    A malformed response is retried once; a second failure marks every review
    in the batch with an error so that a later run can retry it.
    """

    user_content = json.dumps(
        [{"id": index, "text": review["text"][:max_chars]} for index, review in enumerate(reviews)],
        ensure_ascii=False,
    )
    usage: dict[str, Any] = {}
    for attempt in range(2):
        content, usage = client.complete(user_content)
        try:
            return parse_response(content, reviews, max_chars), usage
        except (ValueError, json.JSONDecodeError):
            if attempt == 1:
                break
    return [
        ReviewResult(
            review_id=review["review_id"],
            source_text_sha256=review["source_text_sha256"],
            out_of_scope=False,
            needs_review=True,
            labels=[],
            error="invalid_response",
        )
        for review in reviews
    ], usage


def cache_key(review_id: str, text_hash: str, model_id: str) -> str:
    return f"{review_id}\x1f{text_hash}\x1f{model_id}\x1f{PROMPT_VERSION}"


def load_cache(
    path: Path, model_id: str, reasoning_effort: str | None = None
) -> dict[str, dict[str, Any]]:
    """Load cached records for one model, reasoning setting, and prompt.

    Failed records and records from other settings are skipped, so a later run
    labels those reviews again.
    """

    records: dict[str, dict[str, Any]] = {}
    for part in cache_files(path):
        with part.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue  # a partial last line from an interrupted run
                if record.get("error") or record.get("prompt_version") != PROMPT_VERSION:
                    continue
                if record.get("model_id") != model_id:
                    continue
                if record.get("reasoning_effort") != reasoning_effort:
                    continue
                records[record["review_id"]] = record
    return records


def cache_files(path: Path) -> list[Path]:
    """Return cache part files: one file, or ``part-*.jsonl`` inside a directory."""

    if path.is_dir():
        return sorted(path.glob("part-*.jsonl"))
    return [path] if path.is_file() else []


class ShardedCacheWriter:
    """Append records to ``part-NNNNN.jsonl`` files of bounded size.

    Finished parts never change, so a sync step uploads each one once and only
    re-uploads the open part. A partial last line from a crash is skipped on
    load and the next record starts on a new line.
    """

    def __init__(self, directory: Path, rows_per_part: int = 50_000) -> None:
        if rows_per_part < 1:
            raise ValueError("rows_per_part must be positive")
        self.directory = directory
        self.rows_per_part = rows_per_part
        directory.mkdir(parents=True, exist_ok=True)
        parts = cache_files(directory)
        self.index = int(parts[-1].stem.split("-")[1]) if parts else 0
        self.rows = self._count(self._path()) if parts else 0
        self.handle = self._open()

    def _path(self) -> Path:
        return self.directory / f"part-{self.index:05d}.jsonl"

    @staticmethod
    def _count(path: Path) -> int:
        if not path.is_file():
            return 0
        with path.open("rb") as handle:
            return sum(1 for _ in handle)

    def _open(self) -> Any:
        path = self._path()
        handle = path.open("a", encoding="utf-8")
        if path.stat().st_size and not path.read_bytes().endswith(b"\n"):
            handle.write("\n")
        return handle

    def write(self, line: str) -> None:
        if self.rows >= self.rows_per_part:
            self.handle.close()
            self.index += 1
            self.rows = 0
            self.handle = self._open()
        self.handle.write(line if line.endswith("\n") else line + "\n")
        self.rows += 1

    def flush(self) -> None:
        self.handle.flush()

    def close(self) -> None:
        self.handle.close()


class CachedLLMLabeler:
    """Serve cached LLM labels through the ``ReviewLabeler`` interface."""

    name = "llm_cache"

    def __init__(self, records: Mapping[str, Mapping[str, Any]]) -> None:
        self.records = records

    def predict(self, text: str, *, review_id: str = "", unit_id: str = "") -> dict[str, Any]:
        record = self.records.get(review_id)
        if record is None:
            return {"labels": [], "refusal_reason": "not_labeled"}
        if record.get("source_text_sha256") != hashlib.sha256(text.encode()).hexdigest():
            return {"labels": [], "refusal_reason": "text_changed"}
        if record.get("out_of_scope"):
            return {"labels": [], "refusal_reason": "out_of_scope"}
        labels = [
            {
                **label,
                "confidence": 1.0,
                "confidence_kind": "llm_label",
                "sentiment_score": {"positive": 1.0, "negative": -1.0, "neutral": 0.0}[
                    label["polarity"]
                ],
            }
            for label in record.get("labels", [])
        ]
        if not labels:
            return {"labels": [], "refusal_reason": "no_explicit_aspect"}
        return {"labels": labels, "refusal_reason": None}


def batched(items: Iterable[Any], size: int) -> Iterable[list[Any]]:
    batch: list[Any] = []
    for item in items:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch
