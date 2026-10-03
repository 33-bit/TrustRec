"""Run local Laya decisions over the TrustRec pilot units."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from trustrec.nlp.laya_adapter import build_questions, create_router, normalize_prediction


def label_record(
    router: Any, model: str, record: dict[str, Any], min_presence: float
) -> dict[str, Any]:
    prediction = router.predict(record["text"], build_questions(), model=model)
    normalized = normalize_prediction(record["text"], prediction, min_presence=min_presence)
    labels = [] if normalized["out_of_scope"] else normalized["labels"]
    for label in labels:
        label.update(
            {
                "start": record["start"],
                "end": record["end"],
                "evidence": record["text"],
            }
        )
    return {
        "unit_id": record["unit_id"],
        "source_annotation_id": record["source_annotation_id"],
        "review_id": record["review_id"],
        "item_id": record["item_id"],
        "timestamp": record["timestamp"],
        "source_text_sha256": record["source_text_sha256"],
        "start": record["start"],
        "end": record["end"],
        "text": record["text"],
        "labels": labels,
        "out_of_scope": normalized["out_of_scope"],
        "needs_adjudication": normalized["needs_adjudication"],
        "presence_probability": normalized["presence_probability"],
        "confidence": normalized["confidence"],
        "review_reasons": normalized["review_reasons"],
        "provenance": {
            "label_status": "ai_preliminary",
            "annotator": "laya-local",
            "model": normalized["model"] or model,
            "generated_at": datetime.now(UTC).isoformat(),
            "method": "Laya typed decisions; evidence span is the selected sentence/clause",
            "min_presence": min_presence,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("docs/tasks/t2_1_ai_pilot_labels.jsonl"))
    parser.add_argument(
        "--output", type=Path, default=Path("docs/tasks/t2_1_laya_pilot_labels.jsonl")
    )
    parser.add_argument(
        "--model", default="laya", choices=("laya", "multilingual", "typed-decisions")
    )
    parser.add_argument("--device", default=None, help="Optional torch device: cpu, mps, or cuda")
    parser.add_argument(
        "--revision", default=None, help="Optional pinned Hugging Face revision SHA"
    )
    parser.add_argument("--min-presence", type=float, default=0.85)
    args = parser.parse_args()
    if not 0.0 <= args.min_presence <= 1.0:
        raise SystemExit("--min-presence must be between 0 and 1")
    records = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
    router, model = create_router(model=args.model, device=args.device, revision=args.revision)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(label_record(router, model, record, args.min_presence)) + "\n")
    print(f"Wrote {len(records)} Laya preliminary labels to {args.output}")


if __name__ == "__main__":
    main()
