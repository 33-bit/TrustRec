"""Validate a completed T2.3 baseline run and its provenance."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from trustrec.nlp.records import ASPECTS, SENTIMENTS, load_jsonl_records

ROOT = Path(__file__).resolve().parents[1]
SHA256_LENGTH = 64
REQUIRED_MODELS = ("dictionary", "tfidf_linear_svm")
REQUIRED_ARTIFACTS = (
    "dictionary_predictions",
    "tfidf_linear_svm_predictions",
    "dictionary_error_samples",
    "tfidf_linear_svm_error_samples",
    "dictionary_excluded_error_samples",
    "tfidf_linear_svm_excluded_error_samples",
    "tfidf_linear_svm_model",
    "dictionary_target_sentiment",
    "tfidf_linear_svm_target_sentiment",
)


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def _check_prediction_file(
    path: Path,
    *,
    model_name: str,
    expected_units: dict[str, dict[str, Any]],
    expected_model_hash: str,
    expected_configuration_hash: str,
    expected_snapshot_id: str,
    expected_cutoff: str,
) -> int:
    rows = load_jsonl_records(path)
    seen: set[str] = set()
    for row in rows:
        unit_id = row.get("unit_id")
        if not isinstance(unit_id, str) or unit_id in seen:
            raise ValueError(f"{path} contains a duplicate or invalid unit_id")
        seen.add(unit_id)
        source = expected_units.get(unit_id)
        if source is None:
            raise ValueError(f"{path} contains an unknown unit_id: {unit_id}")
        if row.get("schema_version") != 2:
            raise ValueError(f"{path} has an invalid schema_version for {unit_id}")
        for field, expected in (
            ("model", model_name),
            ("model_hash", expected_model_hash),
            ("configuration_hash", expected_configuration_hash),
            ("source_snapshot_id", expected_snapshot_id),
            ("source_cutoff_timestamp", expected_cutoff),
            ("split_role", "llm_pseudo_test"),
            ("label_status", "baseline_prediction"),
            ("source_text_sha256", source["source_text_sha256"]),
            ("unit_start", source["unit_start"]),
            ("unit_end", source["unit_end"]),
            ("unit_text", source["unit_text"]),
        ):
            if row.get(field) != expected:
                raise ValueError(f"{path} has incorrect {field} for {unit_id}")
        text = row.get("text")
        if text != source["text"]:
            raise ValueError(f"{path} text does not match the frozen pseudo-test for {unit_id}")
        labels = row.get("labels")
        if not isinstance(labels, list):
            raise ValueError(f"{path} labels must be a list")
        for label in labels:
            if label.get("aspect") not in ASPECTS or label.get("polarity") not in SENTIMENTS:
                raise ValueError(f"{path} contains an invalid label class for {unit_id}")
            start, end = label.get("start"), label.get("end")
            if (
                not isinstance(start, int)
                or not isinstance(end, int)
                or not 0 <= start <= end <= len(text)
                or text[start:end] != label.get("evidence")
            ):
                raise ValueError(f"{path} contains an invalid evidence span for {unit_id}")
            score = label.get("sentiment_score")
            if not isinstance(score, (int, float)) or not -1 <= score <= 1:
                raise ValueError(f"{path} contains an invalid sentiment score for {unit_id}")
            confidence_kind = label.get("confidence_kind")
            if confidence_kind not in {"rule_score", "calibrated_probability", "decision_label"}:
                raise ValueError(f"{path} contains an invalid confidence kind for {unit_id}")
            confidence = label.get("confidence")
            if confidence is not None and (
                not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1
            ):
                raise ValueError(f"{path} contains an invalid confidence for {unit_id}")
        expected_offsets = [{"start": label["start"], "end": label["end"]} for label in labels]
        if row.get("evidence_offsets") != expected_offsets:
            raise ValueError(f"{path} evidence_offsets do not match labels for {unit_id}")
    if set(expected_units) != seen:
        raise ValueError(f"{path} does not contain exactly the frozen pseudo-test units")
    return len(rows)


def _check_target_sentiment_file(
    path: Path,
    *,
    model_name: str,
    expected_units: dict[str, dict[str, Any]],
    expected_model_hash: str,
    expected_configuration_hash: str,
    expected_snapshot_id: str,
    expected_cutoff: str,
) -> int:
    """Validate target-conditioned sentiment rows against frozen labels."""

    rows = load_jsonl_records(path)
    expected: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    for source in expected_units.values():
        for index, label in enumerate(source.get("labels", []), start=1):
            expected[f"{source['unit_id']}:{index}"] = (source, label)
    seen: set[str] = set()
    for row in rows:
        prediction_id = row.get("prediction_id")
        if not isinstance(prediction_id, str) or prediction_id in seen:
            raise ValueError(f"{path} contains a duplicate or invalid prediction_id")
        seen.add(prediction_id)
        source_label = expected.get(prediction_id)
        if source_label is None:
            raise ValueError(f"{path} contains an unknown target prediction: {prediction_id}")
        source, target = source_label
        for field, expected_value in (
            ("schema_version", 1),
            ("unit_id", source["unit_id"]),
            ("review_id", source["review_id"]),
            ("item_id", source["item_id"]),
            ("source_snapshot_id", expected_snapshot_id),
            ("source_snapshot_dataset_hash", source["source_snapshot_dataset_hash"]),
            ("source_cutoff_timestamp", expected_cutoff),
            ("source_text_sha256", source["source_text_sha256"]),
            ("split_role", "llm_pseudo_test"),
            ("model", model_name),
            ("model_hash", expected_model_hash),
            ("configuration_hash", expected_configuration_hash),
            ("aspect", target["aspect"]),
            ("target_polarity", target["polarity"]),
            ("target_start", target["start"]),
            ("target_end", target["end"]),
            ("target_evidence", target["evidence"]),
        ):
            if row.get(field) != expected_value:
                raise ValueError(f"{path} has incorrect {field} for {prediction_id}")
        predicted_polarity = row.get("predicted_polarity")
        if predicted_polarity is not None and predicted_polarity not in SENTIMENTS:
            raise ValueError(f"{path} has invalid predicted polarity for {prediction_id}")
        confidence = row.get("confidence")
        if confidence is not None and (
            not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1
        ):
            raise ValueError(f"{path} has invalid confidence for {prediction_id}")
    if seen != set(expected):
        raise ValueError(f"{path} does not contain every target sentiment prediction")
    return len(rows)


def validate_run(manifest_path: Path) -> dict[str, Any]:
    """Validate one T2.3 manifest and all listed artifacts."""

    manifest = _load_json(manifest_path)
    for field in (
        "run_id",
        "snapshot_id",
        "dataset_hash",
        "source_cutoff_timestamp",
        "configuration_hash",
        "metrics_path",
        "artifact_paths",
        "artifact_sha256",
    ):
        if field not in manifest:
            raise ValueError(f"manifest lacks {field}")
    if manifest.get("task") != "T2.3" or manifest.get("status") != "complete":
        raise ValueError("manifest is not a completed T2.3 run")
    if manifest.get("frozen_pseudo_test") is not True:
        raise ValueError("manifest must mark the pseudo-test as frozen")
    if manifest.get("pseudo_test_tuning_allowed") is not False:
        raise ValueError("manifest must prohibit pseudo-test tuning")
    metrics_path = _resolve(manifest["metrics_path"])
    metrics = _load_json(metrics_path)
    if metrics.get("run_id") != manifest["run_id"]:
        raise ValueError("metrics run_id does not match manifest")
    if metrics.get("snapshot_id") != manifest["snapshot_id"]:
        raise ValueError("metrics snapshot_id does not match manifest")
    if metrics.get("dataset_hash") != manifest["dataset_hash"]:
        raise ValueError("metrics dataset_hash does not match manifest")
    if metrics.get("source_cutoff_timestamp") != manifest["source_cutoff_timestamp"]:
        raise ValueError("metrics cutoff does not match manifest")
    if metrics.get("configuration_hash") != manifest["configuration_hash"]:
        raise ValueError("metrics configuration hash does not match manifest")
    if set(metrics.get("metrics", {})) != set(REQUIRED_MODELS):
        raise ValueError("metrics must contain both T2.3 baselines")
    for model_name in REQUIRED_MODELS:
        if model_name not in metrics.get("models", {}):
            raise ValueError(f"metrics lacks model provenance for {model_name}")

    input_info = metrics.get("inputs", {}).get("llm_pseudo_test")
    if not isinstance(input_info, dict) or "path" not in input_info:
        raise ValueError("metrics lacks pseudo-test input provenance")
    pseudo_path = _resolve(input_info["path"])
    if _sha256_file(pseudo_path) != input_info.get("sha256"):
        raise ValueError("pseudo-test input hash does not match metrics")
    pseudo_rows = load_jsonl_records(pseudo_path)
    expected_units = {row["unit_id"]: row for row in pseudo_rows}
    if not expected_units:
        raise ValueError("pseudo-test input is empty")
    model_hashes = {name: metrics["models"][name]["model_hash"] for name in REQUIRED_MODELS}
    model_artifact_path = _resolve(manifest["artifact_paths"].get("tfidf_linear_svm_model", ""))
    if model_artifact_path.suffix != ".json":
        raise ValueError("the learned model artifact must use stable JSON")
    model_artifact = _load_json(model_artifact_path)
    if (
        model_artifact.get("schema_version") != 1
        or model_artifact.get("model") != "tfidf_linear_svm"
    ):
        raise ValueError("learned model artifact has an invalid schema")
    if (
        model_artifact.get("training_provenance", {}).get("model_hash")
        != model_hashes["tfidf_linear_svm"]
    ):
        raise ValueError("learned model artifact hash does not match metrics")
    prediction_models = {
        "dictionary_predictions": ("dictionary", model_hashes["dictionary"]),
        "tfidf_linear_svm_predictions": (
            "tfidf_linear_svm",
            model_hashes["tfidf_linear_svm"],
        ),
    }
    validated_units = {}
    for artifact_name, (model_name, model_hash) in prediction_models.items():
        artifact_path = _resolve(manifest["artifact_paths"].get(artifact_name, ""))
        if not artifact_path.is_file():
            raise ValueError(f"artifact does not exist: {artifact_name}")
        validated_units[artifact_name] = _check_prediction_file(
            artifact_path,
            model_name=model_name,
            expected_units=expected_units,
            expected_model_hash=model_hash,
            expected_configuration_hash=manifest["configuration_hash"],
            expected_snapshot_id=manifest["snapshot_id"],
            expected_cutoff=manifest["source_cutoff_timestamp"],
        )
    target_models = {
        "dictionary_target_sentiment": ("dictionary", model_hashes["dictionary"]),
        "tfidf_linear_svm_target_sentiment": (
            "tfidf_linear_svm",
            model_hashes["tfidf_linear_svm"],
        ),
    }
    validated_target_units = {}
    for artifact_name, (model_name, model_hash) in target_models.items():
        artifact_path = _resolve(manifest["artifact_paths"].get(artifact_name, ""))
        if not artifact_path.is_file():
            raise ValueError(f"artifact does not exist: {artifact_name}")
        validated_target_units[artifact_name] = _check_target_sentiment_file(
            artifact_path,
            model_name=model_name,
            expected_units=expected_units,
            expected_model_hash=model_hash,
            expected_configuration_hash=manifest["configuration_hash"],
            expected_snapshot_id=manifest["snapshot_id"],
            expected_cutoff=manifest["source_cutoff_timestamp"],
        )

    artifact_hashes = manifest["artifact_sha256"]
    missing_artifacts = set(REQUIRED_ARTIFACTS) - set(manifest["artifact_paths"])
    if missing_artifacts:
        raise ValueError("manifest lacks artifact paths: " + ", ".join(sorted(missing_artifacts)))
    missing_hashes = (set(REQUIRED_ARTIFACTS) | {"metrics"}) - set(artifact_hashes)
    if missing_hashes:
        raise ValueError("manifest lacks artifact hashes: " + ", ".join(sorted(missing_hashes)))
    for name, expected_hash in artifact_hashes.items():
        path = (
            metrics_path
            if name == "metrics"
            else _resolve(manifest["artifact_paths"].get(name, ""))
        )
        if not path.is_file():
            raise ValueError(f"artifact does not exist: {name}")
        if not isinstance(expected_hash, str) or len(expected_hash) != SHA256_LENGTH:
            raise ValueError(f"artifact hash is invalid: {name}")
        if _sha256_file(path) != expected_hash:
            raise ValueError(f"artifact hash mismatch: {name}")
    return {
        "run_id": manifest["run_id"],
        "validated_pseudo_test_units": len(expected_units),
        "validated_prediction_units": validated_units,
        "validated_target_sentiment_rows": validated_target_units,
        "artifact_count": len(artifact_hashes),
        "status": manifest["status"],
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "manifest",
        nargs="?",
        type=Path,
        default=ROOT / "docs/tasks/t2_3_nlp_baselines.manifest.json",
    )
    args = parser.parse_args()
    print(json.dumps(validate_run(args.manifest), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
