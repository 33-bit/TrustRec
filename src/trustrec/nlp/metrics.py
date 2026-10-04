"""Consistency metrics and error samples for NLP baseline predictions."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from typing import Any

from sklearn.metrics import f1_score, precision_score, recall_score

from .records import ASPECTS, SENTIMENTS


def _label_pairs(labels: Iterable[dict[str, Any]]) -> set[tuple[str, str]]:
    return {(label["aspect"], label["polarity"]) for label in labels}


def _safe_rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _pair_metrics(
    targets: list[set[tuple[str, str]]], predictions: list[set[tuple[str, str]]]
) -> dict[str, Any]:
    true_positive = sum(
        len(target & prediction) for target, prediction in zip(targets, predictions, strict=True)
    )
    false_positive = sum(
        len(prediction - target) for target, prediction in zip(targets, predictions, strict=True)
    )
    false_negative = sum(
        len(target - prediction) for target, prediction in zip(targets, predictions, strict=True)
    )
    precision = _safe_rate(true_positive, true_positive + false_positive)
    recall = _safe_rate(true_positive, true_positive + false_negative)
    f1 = _safe_rate(2 * precision * recall, precision + recall)
    return {
        "micro_precision": precision,
        "micro_recall": recall,
        "micro_f1": f1,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "support": true_positive + false_negative,
    }


def _evidence_metrics(
    records: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
) -> dict[str, Any]:
    reference_by_key = {(row["unit_id"]): row for row in records}
    valid = 0
    total = 0
    overlap = 0
    overlap_total = 0
    for prediction in predictions:
        reference = reference_by_key.get(prediction.get("unit_id"))
        if reference is None:
            continue
        text = reference["text"]
        target_by_aspect = {}
        for label in reference.get("labels", []):
            target_by_aspect.setdefault(label["aspect"], []).append(label)
        for label in prediction.get("labels", []):
            total += 1
            start, end = label.get("start"), label.get("end")
            if (
                isinstance(start, int)
                and isinstance(end, int)
                and 0 <= start <= end <= len(text)
                and text[start:end] == label.get("evidence")
            ):
                valid += 1
            candidates = target_by_aspect.get(label.get("aspect"), [])
            evidence_is_valid = (
                isinstance(start, int)
                and isinstance(end, int)
                and 0 <= start <= end <= len(text)
                and text[start:end] == label.get("evidence")
            )
            if candidates and evidence_is_valid:
                overlap_total += 1
                if any(start <= target["start"] <= target["end"] <= end for target in candidates):
                    overlap += 1
    return {
        "valid_count": valid,
        "total_count": total,
        "valid_rate": _safe_rate(valid, total) if total else 1.0,
        "reference_overlap_count": overlap,
        "reference_overlap_total": overlap_total,
        "reference_overlap_rate": _safe_rate(overlap, overlap_total) if overlap_total else 0.0,
    }


def evaluate_predictions(
    reference_records: Iterable[dict[str, Any]],
    predictions: Iterable[dict[str, Any]],
    *,
    max_error_samples: int = 20,
) -> dict[str, Any]:
    """Compare predictions with eligible frozen pseudo-label records."""

    records = list(reference_records)
    prediction_rows = list(predictions)
    by_unit = {row["unit_id"]: row for row in prediction_rows}
    if len(by_unit) != len(prediction_rows):
        raise ValueError("predictions contain duplicate unit_id values")
    eligible = [
        row for row in records if not row.get("out_of_scope") and not row.get("needs_review")
    ]
    excluded = [row for row in records if row not in eligible]
    target_aspects: list[set[str]] = []
    predicted_aspects: list[set[str]] = []
    target_pairs: list[set[tuple[str, str]]] = []
    predicted_pairs: list[set[tuple[str, str]]] = []
    target_sentiments: list[str] = []
    predicted_sentiments: list[str] = []
    detected_sentiment_count = 0
    target_sentiment_count = 0
    errors: list[dict[str, Any]] = []
    excluded_errors: list[dict[str, Any]] = []
    valid_prediction_rows = [
        by_unit[row["unit_id"]] for row in eligible if row["unit_id"] in by_unit
    ]
    excluded_prediction_rows = [
        by_unit[row["unit_id"]] for row in excluded if row["unit_id"] in by_unit
    ]

    for row in eligible:
        prediction = by_unit.get(
            row["unit_id"], {"labels": [], "refusal_reason": "missing_prediction"}
        )
        target_labels = row.get("labels", [])
        predicted_labels = prediction.get("labels", [])
        target_aspect_set = {label["aspect"] for label in target_labels}
        predicted_aspect_set = {label["aspect"] for label in predicted_labels}
        target_pair_set = _label_pairs(target_labels)
        predicted_pair_set = _label_pairs(predicted_labels)
        target_aspects.append(target_aspect_set)
        predicted_aspects.append(predicted_aspect_set)
        target_pairs.append(target_pair_set)
        predicted_pairs.append(predicted_pair_set)

        matched_predictions: list[tuple[dict[str, Any], dict[str, Any]]] = []
        unused_predictions = set(range(len(predicted_labels)))
        for target in target_labels:
            target_sentiment_count += 1
            candidates = [
                index
                for index in unused_predictions
                if predicted_labels[index].get("aspect") == target["aspect"]
            ]
            if candidates:
                predicted_index = min(
                    candidates,
                    key=lambda index: abs(predicted_labels[index]["start"] - target["start"]),
                )
                unused_predictions.remove(predicted_index)
                predicted = predicted_labels[predicted_index]
                detected_sentiment_count += 1
                target_sentiments.append(target["polarity"])
                predicted_sentiments.append(predicted["polarity"])
                matched_predictions.append((target, predicted))

        error_types: list[str] = []
        if target_aspect_set - predicted_aspect_set:
            error_types.append("aspect_false_negative")
        if predicted_aspect_set - target_aspect_set:
            error_types.append("aspect_false_positive")
        if any(
            target["polarity"] != predicted["polarity"] for target, predicted in matched_predictions
        ):
            error_types.append("sentiment_mismatch")
        if len(matched_predictions) < len(target_labels):
            error_types.append("abstention")
        if prediction.get("refusal_reason"):
            error_types.append("refusal")
        if error_types:
            errors.append(
                {
                    "unit_id": row["unit_id"],
                    "review_id": row["review_id"],
                    "text": row["text"],
                    "target_labels": target_labels,
                    "predicted_labels": predicted_labels,
                    "refusal_reason": prediction.get("refusal_reason"),
                    "error_types": sorted(set(error_types)),
                }
            )

    for row in excluded:
        prediction = by_unit.get(
            row["unit_id"], {"labels": [], "refusal_reason": "missing_prediction"}
        )
        if prediction.get("labels") or prediction.get("refusal_reason"):
            excluded_errors.append(
                {
                    "unit_id": row["unit_id"],
                    "review_id": row["review_id"],
                    "text": row["text"],
                    "excluded_reason": "out_of_scope"
                    if row.get("out_of_scope")
                    else "needs_review",
                    "target_labels": row.get("labels", []),
                    "predicted_labels": prediction.get("labels", []),
                    "refusal_reason": prediction.get("refusal_reason"),
                }
            )

    if target_aspects:
        aspect_target_matrix = [
            [int(aspect in values) for aspect in ASPECTS] for values in target_aspects
        ]
        aspect_prediction_matrix = [
            [int(aspect in values) for aspect in ASPECTS] for values in predicted_aspects
        ]
        aspect_metrics = {
            "micro_precision": float(
                precision_score(
                    aspect_target_matrix, aspect_prediction_matrix, average="micro", zero_division=0
                )
            ),
            "micro_recall": float(
                recall_score(
                    aspect_target_matrix, aspect_prediction_matrix, average="micro", zero_division=0
                )
            ),
            "micro_f1": float(
                f1_score(
                    aspect_target_matrix, aspect_prediction_matrix, average="micro", zero_division=0
                )
            ),
            "macro_f1": float(
                f1_score(
                    aspect_target_matrix, aspect_prediction_matrix, average="macro", zero_division=0
                )
            ),
            "support": sum(len(values) for values in target_aspects),
        }
    else:
        aspect_metrics = {
            "micro_precision": 0.0,
            "micro_recall": 0.0,
            "micro_f1": 0.0,
            "macro_f1": 0.0,
            "support": 0,
        }

    if target_sentiments:
        sentiment_macro_f1 = float(
            f1_score(
                target_sentiments,
                predicted_sentiments,
                labels=list(SENTIMENTS),
                average="macro",
                zero_division=0,
            )
        )
    else:
        sentiment_macro_f1 = 0.0
    sentiment_detected = {
        "macro_f1": sentiment_macro_f1,
        "support": target_sentiment_count,
        "detected": detected_sentiment_count,
        "coverage": _safe_rate(detected_sentiment_count, target_sentiment_count),
    }
    metrics = {
        "eligible_units": len(eligible),
        "excluded_units": len(excluded),
        "out_of_scope_units": sum(bool(row.get("out_of_scope")) for row in excluded),
        "needs_review_units": sum(bool(row.get("needs_review")) for row in excluded),
        "prediction_rows": len(valid_prediction_rows),
        "aspect": aspect_metrics,
        "aspect_polarity_pair": _pair_metrics(target_pairs, predicted_pairs),
        "sentiment_on_detected_aspects": sentiment_detected,
        "evidence": _evidence_metrics(eligible, valid_prediction_rows),
        "excluded_evidence": _evidence_metrics(excluded, excluded_prediction_rows),
        "refusal_counts": dict(
            Counter(
                row.get("refusal_reason")
                for row in valid_prediction_rows
                if row.get("refusal_reason")
            )
        ),
        "error_samples": sorted(errors, key=lambda row: row["unit_id"])[:max_error_samples],
        "excluded_error_samples": sorted(excluded_errors, key=lambda row: row["unit_id"])[
            :max_error_samples
        ],
    }
    return metrics


def evaluate_target_sentiment(
    reference_records: Iterable[dict[str, Any]],
    model: Any,
) -> dict[str, Any]:
    """Measure sentiment on each supplied reference aspect and evidence span."""

    targets: list[str] = []
    predictions: list[str] = []
    covered_targets: list[str] = []
    covered_predictions: list[str] = []
    abstained = 0
    for row in reference_records:
        if row.get("out_of_scope") or row.get("needs_review"):
            continue
        for target in row.get("labels", []):
            result = model.predict_target_sentiment(target["evidence"], target["aspect"])
            targets.append(target["polarity"])
            polarity = result["polarity"]
            if result["refusal_reason"] is not None or polarity is None:
                predictions.append("abstain")
                abstained += 1
                continue
            predictions.append(polarity)
            covered_targets.append(target["polarity"])
            covered_predictions.append(polarity)
    macro_f1 = (
        float(
            f1_score(
                targets, predictions, labels=list(SENTIMENTS), average="macro", zero_division=0
            )
        )
        if targets
        else 0.0
    )
    covered_f1 = (
        float(
            f1_score(
                covered_targets,
                covered_predictions,
                labels=list(SENTIMENTS),
                average="macro",
                zero_division=0,
            )
        )
        if covered_targets
        else 0.0
    )
    return {
        "macro_f1": macro_f1,
        "covered_macro_f1": covered_f1,
        "support": len(targets),
        "predicted": len(covered_targets),
        "abstained": abstained,
        "coverage": _safe_rate(len(covered_targets), len(targets)),
        "condition": "reference_aspect_and_evidence_span",
    }
