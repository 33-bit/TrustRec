"""Run the deterministic and learned T2.3 NLP baselines."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tomllib
from collections.abc import Iterable
from pathlib import Path
from typing import Any

try:
    from scripts.validate_llm_pseudo_test import validate_pseudo_test_artifacts
except ModuleNotFoundError:
    from validate_llm_pseudo_test import validate_pseudo_test_artifacts
from trustrec.nlp.baselines import (
    DictionaryBaseline,
    assert_disjoint_splits,
    evaluate_predictions,
    evaluate_target_sentiment,
    load_jsonl_records,
    train_tfidf_svm,
)
from trustrec.nlp.dictionary import (
    ASPECT_TERMS,
    NEGATIONS,
    NEGATIVE_TERMS,
    NEUTRAL_TERMS,
    OUT_OF_SCOPE_TERMS,
    POSITIVE_TERMS,
)
from trustrec.nlp.records import validate_annotation_records

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PILOT = ROOT / "docs/tasks/t2_1_llm_review.jsonl"
DEFAULT_PSEUDO_TEST = ROOT / "docs/tasks/t2_2_llm_pseudo_test.jsonl"
DEFAULT_PSEUDO_LABELS = ROOT / "docs/tasks/t2_2_llm_pseudo_test.csv"
DEFAULT_PSEUDO_MANIFEST = ROOT / "docs/tasks/t2_2_llm_pseudo_test.manifest.json"
DEFAULT_CONFIG = ROOT / "configs/nlp_baselines.toml"
DEFAULT_OUTPUT = ROOT / "artifacts/t2_3"
DEFAULT_MANIFEST = ROOT / "docs/tasks/t2_3_nlp_baselines.manifest.json"
DEFAULT_REPORT = ROOT / "docs/tasks/t2_3-nlp-baselines.md"


def sha256_bytes(value: bytes) -> str:
    """Return a hexadecimal SHA-256 digest."""

    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    """Hash one file in binary mode."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def root_path(path: Path) -> Path:
    """Resolve a CLI path against the repository root."""

    return path if path.is_absolute() else ROOT / path


def canonical_json(value: Any) -> bytes:
    """Serialize a value for stable artifact hashing."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def write_json(path: Path, value: Any) -> None:
    """Write formatted JSON with a final newline."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    """Write JSON Lines records in the supplied order."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")


def code_hash() -> str:
    """Hash the source files that define this run."""

    paths = [
        ROOT / "scripts/run_nlp_baselines.py",
        ROOT / "src/trustrec/nlp/baselines.py",
        ROOT / "src/trustrec/nlp/dictionary.py",
        ROOT / "src/trustrec/nlp/metrics.py",
        ROOT / "src/trustrec/nlp/records.py",
        ROOT / "src/trustrec/nlp/svm.py",
        ROOT / "src/trustrec/nlp/text.py",
    ]
    digest = hashlib.sha256()
    for path in paths:
        relative = path.relative_to(ROOT).as_posix()
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def git_revision() -> str:
    """Return the current Git revision or ``unavailable`` outside Git."""

    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def load_configuration(path: Path) -> dict[str, Any]:
    """Load and validate the fixed baseline configuration."""

    with path.open("rb") as handle:
        configuration = tomllib.load(handle)
    required = {"run", "model", "evaluation"}
    if set(configuration) != required:
        raise ValueError("baseline configuration sections are incomplete")
    if configuration["run"] != {
        "protocol_version": "nlp-baseline-v1",
        "seed": 7,
        "calibration_folds": 3,
    }:
        raise ValueError("baseline run configuration does not match the fixed T2.3 contract")
    if configuration["model"] != {
        "ngram_min": 1,
        "ngram_max": 2,
        "min_df": 1,
        "sublinear_tf": True,
        "c_value": 1.0,
        "aspect_threshold": 0.5,
        "sentiment_threshold": 0.6,
    }:
        raise ValueError("baseline model configuration does not match the fixed T2.3 contract")
    if configuration["evaluation"] != {"max_error_samples": 20}:
        raise ValueError("baseline evaluation configuration does not match the fixed T2.3 contract")
    return configuration


def _prediction_record(
    source: dict[str, Any], prediction: dict[str, Any], model_hash: str, config_hash: str, seed: int
) -> dict[str, Any]:
    """Attach source lineage to one model prediction."""

    return {
        "schema_version": 2,
        "unit_id": source["unit_id"],
        "review_id": source["review_id"],
        "item_id": source["item_id"],
        "source_snapshot_id": source["source_snapshot_id"],
        "source_snapshot_dataset_hash": source["source_snapshot_dataset_hash"],
        "source_cutoff_timestamp": source["source_cutoff_timestamp"],
        "source_text_sha256": source["source_text_sha256"],
        "timestamp": source["timestamp"],
        "split_role": source["split_role"],
        "label_status": "baseline_prediction",
        "model": prediction["model"],
        "model_id": prediction["model"],
        "model_revision": model_hash,
        "prompt_version": "not_applicable",
        "temperature": "not_applicable",
        "seed": seed,
        "generated_at": source.get("generated_at", "unavailable"),
        "model_hash": model_hash,
        "configuration_hash": config_hash,
        "unit_start": source["unit_start"],
        "unit_end": source["unit_end"],
        "unit_text": source["unit_text"],
        "text": source["text"],
        "labels": prediction["labels"],
        "evidence_offsets": [
            {"start": label["start"], "end": label["end"]} for label in prediction["labels"]
        ],
        "usage_restrictions": {
            "allowed_uses": ["aspect_consistency", "sentiment_consistency", "evidence_consistency"],
            "prohibited_uses": ["recommendation_metrics", "human_gold_claim", "ground_truth_claim"],
        },
        "refusal_reason": prediction["refusal_reason"],
    }


def _relative_or_absolute(path: Path) -> str:
    """Return a repository path when possible, otherwise an absolute path."""

    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def _target_sentiment_records(
    model: Any,
    records: list[dict[str, Any]],
    model_hash: str,
    config_hash: str,
    seed: int,
) -> list[dict[str, Any]]:
    """Export sentiment predictions for each supplied reference aspect span."""

    rows = []
    for source in records:
        for index, target in enumerate(source["labels"], start=1):
            result = model.predict_target_sentiment(target["evidence"], target["aspect"])
            rows.append(
                {
                    "schema_version": 1,
                    "prediction_id": f"{source['unit_id']}:{index}",
                    "unit_id": source["unit_id"],
                    "review_id": source["review_id"],
                    "item_id": source["item_id"],
                    "source_snapshot_id": source["source_snapshot_id"],
                    "source_snapshot_dataset_hash": source["source_snapshot_dataset_hash"],
                    "source_cutoff_timestamp": source["source_cutoff_timestamp"],
                    "source_text_sha256": source["source_text_sha256"],
                    "split_role": source["split_role"],
                    "model": model.name,
                    "model_hash": model_hash,
                    "configuration_hash": config_hash,
                    "seed": seed,
                    "aspect": target["aspect"],
                    "target_polarity": target["polarity"],
                    "target_start": target["start"],
                    "target_end": target["end"],
                    "target_evidence": target["evidence"],
                    "predicted_polarity": result["polarity"],
                    "confidence": result["confidence"],
                    "refusal_reason": result["refusal_reason"],
                }
            )
    return rows


def _model_hash(name: str, config: dict[str, Any], training_records: list[dict[str, Any]]) -> str:
    payload = {
        "model": name,
        "configuration": config,
        "training_units": [
            {"unit_id": row["unit_id"], "source_text_sha256": row["source_text_sha256"]}
            for row in training_records
        ],
    }
    return sha256_bytes(canonical_json(payload))


def _provenance_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "snapshot_ids": sorted({row["source_snapshot_id"] for row in records}),
        "dataset_hashes": sorted({row["source_snapshot_dataset_hash"] for row in records}),
        "cutoffs": sorted({row["source_cutoff_timestamp"] for row in records}),
        "unit_count": len(records),
        "unit_ids_sha256": sha256_bytes(canonical_json([row["unit_id"] for row in records])),
    }


def _consistent_provenance(records: list[dict[str, Any]], role: str) -> dict[str, str]:
    """Require one source snapshot, hash, and cutoff for one split."""

    fields = ("source_snapshot_id", "source_snapshot_dataset_hash", "source_cutoff_timestamp")
    values = {field: {str(row[field]) for row in records} for field in fields}
    if any(len(value) != 1 for value in values.values()):
        raise ValueError(f"{role} contains mixed source provenance")
    return {field: next(iter(value)) for field, value in values.items()}


def _consistent_fields(
    records: list[dict[str, Any]], fields: tuple[str, ...], role: str
) -> dict[str, str]:
    """Require one value for each listed provenance field."""

    values = {field: {str(row.get(field)) for row in records} for field in fields}
    if any(len(value) != 1 for value in values.values()):
        raise ValueError(f"{role} contains mixed {', '.join(fields)}")
    return {field: next(iter(value)) for field, value in values.items()}


def _classifier_artifact(classifier: Any) -> dict[str, Any]:
    """Serialize fitted classifier parameters in stable JSON form."""

    payload: dict[str, Any] = {"type": type(classifier).__name__}
    if hasattr(classifier, "value"):
        payload["value"] = classifier.value
        payload["classes"] = classifier.classes_.tolist() if classifier.classes_ is not None else []
        return payload
    payload["calibration_status"] = classifier.calibration_status
    payload["classes"] = classifier.classes_.tolist() if classifier.classes_ is not None else []
    payload["base"] = {
        "classes": classifier.base.classes_.tolist(),
        "coef": classifier.base.coef_.tolist(),
        "intercept": classifier.base.intercept_.tolist(),
    }
    if classifier.calibrator is not None:
        payload["calibrator"] = {
            "classes": classifier.calibrator.classes_.tolist(),
            "coef": classifier.calibrator.coef_.tolist(),
            "intercept": classifier.calibrator.intercept_.tolist(),
        }
    return payload


def model_artifact_payload(model: Any) -> dict[str, Any]:
    """Return a deterministic, reconstructable summary of a fitted model."""

    return {
        "schema_version": 1,
        "model": model.name,
        "config": model.config,
        "training_provenance": model.training_provenance,
        "aspect_vectorizer": {
            "vocabulary": sorted(model.aspect_vectorizer.vocabulary_.items()),
            "idf": model.aspect_vectorizer.idf_.tolist(),
        },
        "aspect_classifiers": {
            aspect: _classifier_artifact(model.aspect_classifiers[aspect])
            for aspect in sorted(model.aspect_classifiers)
        },
        "sentiment_vectorizer": {
            "vocabulary": sorted(model.sentiment_vectorizer.vocabulary_.items()),
            "idf": model.sentiment_vectorizer.idf_.tolist(),
        },
        "sentiment_classifier": _classifier_artifact(model.sentiment_classifier),
    }


def run_baselines(
    *,
    pilot_path: Path = DEFAULT_PILOT,
    pseudo_test_path: Path = DEFAULT_PSEUDO_TEST,
    pseudo_labels_path: Path = DEFAULT_PSEUDO_LABELS,
    pseudo_manifest_path: Path = DEFAULT_PSEUDO_MANIFEST,
    config_path: Path = DEFAULT_CONFIG,
    output_dir: Path = DEFAULT_OUTPUT,
    manifest_path: Path = DEFAULT_MANIFEST,
    report_path: Path = DEFAULT_REPORT,
    run_id: str = "t2.3-nlp-baselines-v1",
) -> dict[str, Any]:
    """Fit, assess, and export both T2.3 baselines."""

    pilot_path = root_path(pilot_path)
    pseudo_test_path = root_path(pseudo_test_path)
    config_path = root_path(config_path)
    output_dir = root_path(output_dir)
    manifest_path = root_path(manifest_path)
    report_path = root_path(report_path)
    if manifest_path.exists():
        raise FileExistsError(f"completed manifest already exists: {manifest_path}")
    if report_path.exists():
        raise FileExistsError(f"completed report already exists: {report_path}")
    pseudo_labels_path = root_path(pseudo_labels_path)
    pseudo_manifest_path = root_path(pseudo_manifest_path)
    validate_pseudo_test_artifacts(pseudo_labels_path, pseudo_test_path, pseudo_manifest_path)
    configuration = load_configuration(config_path)
    pilot = validate_annotation_records(
        load_jsonl_records(pilot_path),
        expected_role="development_pilot",
        expected_status="llm_silver",
    )
    pseudo_test = validate_annotation_records(
        load_jsonl_records(pseudo_test_path),
        expected_role="llm_pseudo_test",
        expected_status="llm_pseudo_test",
    )
    pilot_provenance = _consistent_provenance(pilot, "development_pilot")
    pseudo_provenance = _consistent_provenance(pseudo_test, "llm_pseudo_test")
    pseudo_label_provenance = _consistent_fields(
        pseudo_test,
        ("model_id", "model_revision", "prompt_version", "temperature", "seed"),
        "llm_pseudo_test",
    )
    if pilot_provenance["source_snapshot_id"] != pseudo_provenance["source_snapshot_id"]:
        raise ValueError("pilot and pseudo-test snapshot IDs do not match")
    if (
        pilot_provenance["source_snapshot_dataset_hash"]
        != pseudo_provenance["source_snapshot_dataset_hash"]
    ):
        raise ValueError("pilot and pseudo-test dataset hashes do not match")
    assert_disjoint_splits(pilot, pseudo_test)

    run_configuration = {
        "run": configuration["run"],
        "model": configuration["model"],
        "evaluation": configuration["evaluation"],
    }
    config_hash = sha256_file(config_path)
    seed = int(configuration["run"]["seed"])
    model_configuration = {
        "seed": seed,
        "c_value": float(configuration["model"]["c_value"]),
        "calibration_folds": int(configuration["run"]["calibration_folds"]),
        "aspect_threshold": float(configuration["model"]["aspect_threshold"]),
        "sentiment_threshold": float(configuration["model"]["sentiment_threshold"]),
    }
    svm = train_tfidf_svm(
        pilot,
        seed=seed,
        cutoff_timestamp=pilot_provenance["source_cutoff_timestamp"],
        c_value=model_configuration["c_value"],
        calibration_folds=model_configuration["calibration_folds"],
        aspect_threshold=model_configuration["aspect_threshold"],
        sentiment_threshold=model_configuration["sentiment_threshold"],
    )
    dictionary = DictionaryBaseline()
    dictionary_hash = _model_hash(
        "dictionary",
        {
            **run_configuration,
            "aspect_terms": ASPECT_TERMS,
            "positive_terms": POSITIVE_TERMS,
            "negative_terms": NEGATIVE_TERMS,
            "neutral_terms": NEUTRAL_TERMS,
            "negations": sorted(NEGATIONS),
            "out_of_scope_terms": OUT_OF_SCOPE_TERMS,
        },
        [],
    )
    svm_hash = svm.model_hash

    dictionary_predictions = [
        _prediction_record(
            row,
            dictionary.predict(row["text"], review_id=row["review_id"], unit_id=row["unit_id"]),
            dictionary_hash,
            config_hash,
            seed,
        )
        for row in pseudo_test
    ]
    svm_predictions = [
        _prediction_record(
            row,
            svm.predict(row["text"], review_id=row["review_id"], unit_id=row["unit_id"]),
            svm_hash,
            config_hash,
            seed,
        )
        for row in pseudo_test
    ]
    dictionary_target_sentiment = _target_sentiment_records(
        dictionary, pseudo_test, dictionary_hash, config_hash, seed
    )
    svm_target_sentiment = _target_sentiment_records(svm, pseudo_test, svm_hash, config_hash, seed)
    dictionary_metrics = evaluate_predictions(
        pseudo_test,
        dictionary_predictions,
        max_error_samples=int(configuration["evaluation"]["max_error_samples"]),
    )
    svm_metrics = evaluate_predictions(
        pseudo_test,
        svm_predictions,
        max_error_samples=int(configuration["evaluation"]["max_error_samples"]),
    )
    dictionary_metrics["sentiment_on_reference_aspects"] = evaluate_target_sentiment(
        pseudo_test, dictionary
    )
    svm_metrics["sentiment_on_reference_aspects"] = evaluate_target_sentiment(pseudo_test, svm)

    output_dir.mkdir(parents=True, exist_ok=False)
    prediction_paths = {
        "dictionary": output_dir / "dictionary_predictions.jsonl",
        "tfidf_linear_svm": output_dir / "tfidf_linear_svm_predictions.jsonl",
    }
    write_jsonl(prediction_paths["dictionary"], dictionary_predictions)
    write_jsonl(prediction_paths["tfidf_linear_svm"], svm_predictions)
    error_paths = {
        "dictionary": output_dir / "dictionary_error_samples.jsonl",
        "tfidf_linear_svm": output_dir / "tfidf_linear_svm_error_samples.jsonl",
    }
    write_jsonl(error_paths["dictionary"], dictionary_metrics["error_samples"])
    write_jsonl(error_paths["tfidf_linear_svm"], svm_metrics["error_samples"])
    excluded_error_paths = {
        "dictionary": output_dir / "dictionary_excluded_error_samples.jsonl",
        "tfidf_linear_svm": output_dir / "tfidf_linear_svm_excluded_error_samples.jsonl",
    }
    write_jsonl(excluded_error_paths["dictionary"], dictionary_metrics["excluded_error_samples"])
    write_jsonl(excluded_error_paths["tfidf_linear_svm"], svm_metrics["excluded_error_samples"])
    target_sentiment_paths = {
        "dictionary": output_dir / "dictionary_target_sentiment.jsonl",
        "tfidf_linear_svm": output_dir / "tfidf_linear_svm_target_sentiment.jsonl",
    }
    write_jsonl(target_sentiment_paths["dictionary"], dictionary_target_sentiment)
    write_jsonl(target_sentiment_paths["tfidf_linear_svm"], svm_target_sentiment)
    model_path = output_dir / "tfidf_linear_svm_model.json"
    write_json(model_path, model_artifact_payload(svm))

    generated_at = pseudo_test[0].get("generated_at", "unavailable")
    metrics = {
        "schema_version": 1,
        "run_id": run_id,
        "task": "T2.3",
        "protocol_version": configuration["run"]["protocol_version"],
        "generated_at": generated_at,
        "snapshot_id": pseudo_provenance["source_snapshot_id"],
        "dataset_hash": pseudo_provenance["source_snapshot_dataset_hash"],
        "source_cutoff_timestamp": pseudo_provenance["source_cutoff_timestamp"],
        "pseudo_test_label_provenance": pseudo_label_provenance,
        "seed": seed,
        "config_path": _relative_or_absolute(config_path),
        "configuration_hash": config_hash,
        "code_hash": code_hash(),
        "code_revision": git_revision(),
        "inputs": {
            "development_pilot": {
                "path": _relative_or_absolute(pilot_path),
                "sha256": sha256_file(pilot_path),
                **_provenance_summary(pilot),
            },
            "llm_pseudo_test": {
                "path": _relative_or_absolute(pseudo_test_path),
                "sha256": sha256_file(pseudo_test_path),
                **_provenance_summary(pseudo_test),
            },
        },
        "models": {
            "dictionary": {
                "model_hash": dictionary_hash,
                "training_unit_count": 0,
                "configuration": "fixed lexical aspects, sentiment, and negation rules",
            },
            "tfidf_linear_svm": {
                "model_hash": svm_hash,
                "training_unit_count": svm.training_provenance["training_unit_count"],
                "configuration": svm.config,
                "training_provenance": svm.training_provenance,
                "sentiment_classes": sorted(
                    {label["polarity"] for row in pilot for label in row["labels"]}
                ),
            },
        },
        "metrics": {"dictionary": dictionary_metrics, "tfidf_linear_svm": svm_metrics},
    }
    metrics_path = output_dir / "metrics.json"
    write_json(metrics_path, metrics)
    artifact_files = {
        "dictionary_predictions": prediction_paths["dictionary"],
        "tfidf_linear_svm_predictions": prediction_paths["tfidf_linear_svm"],
        "dictionary_error_samples": error_paths["dictionary"],
        "tfidf_linear_svm_error_samples": error_paths["tfidf_linear_svm"],
        "dictionary_excluded_error_samples": excluded_error_paths["dictionary"],
        "tfidf_linear_svm_excluded_error_samples": excluded_error_paths["tfidf_linear_svm"],
        "dictionary_target_sentiment": target_sentiment_paths["dictionary"],
        "tfidf_linear_svm_target_sentiment": target_sentiment_paths["tfidf_linear_svm"],
        "tfidf_linear_svm_model": model_path,
        "metrics": metrics_path,
    }
    artifacts = {key: sha256_file(path) for key, path in artifact_files.items()}
    artifact_paths = {
        key: _relative_or_absolute(path) for key, path in artifact_files.items() if key != "metrics"
    }
    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "task": "T2.3",
        "status": "complete",
        "protocol_version": configuration["run"]["protocol_version"],
        "snapshot_id": pseudo_provenance["source_snapshot_id"],
        "dataset_hash": pseudo_provenance["source_snapshot_dataset_hash"],
        "source_cutoff_timestamp": pseudo_provenance["source_cutoff_timestamp"],
        "seed": seed,
        "config_path": _relative_or_absolute(config_path),
        "configuration_hash": config_hash,
        "code_hash": metrics["code_hash"],
        "code_revision": metrics["code_revision"],
        "metrics_path": _relative_or_absolute(metrics_path),
        "artifact_paths": artifact_paths,
        "artifact_sha256": artifacts,
        "frozen_pseudo_test": True,
        "pseudo_test_tuning_allowed": False,
    }
    write_json(manifest_path, manifest)
    write_report(
        report_path,
        metrics,
        manifest_path,
        pilot_path,
        pseudo_test_path,
    )
    return manifest


def write_report(
    path: Path,
    metrics: dict[str, Any],
    manifest_path: Path,
    pilot_path: Path,
    pseudo_test_path: Path,
) -> None:
    """Write a concise task report from one completed run."""

    dictionary = metrics["metrics"]["dictionary"]
    svm = metrics["metrics"]["tfidf_linear_svm"]
    sentiment_classes = ", ".join(metrics["models"]["tfidf_linear_svm"]["sentiment_classes"])
    pilot_rel = _relative_or_absolute(pilot_path)
    pseudo_rel = _relative_or_absolute(pseudo_test_path)
    text = "\n".join(
        [
            "# T2.3: aspect and sentiment baselines",
            "",
            "Status: done.",
            "",
            (
                "This task implements a fixed dictionary baseline and a TF-IDF plus "
                "Linear SVM baseline."
            ),
            (
                "The learned model uses `development_pilot` records with `llm_silver` "
                "status for fitting."
            ),
            "The frozen `llm_pseudo_test` records are assessment-only.",
            "The results measure consistency with frozen LLM pseudo-labels.",
            "They are not human agreement, ground-truth accuracy, or recommendation metrics.",
            "",
            (
                f"The run uses snapshot `{metrics['snapshot_id']}`, dataset hash "
                f"`{metrics['dataset_hash']}`, and cutoff "
                f"`{metrics['source_cutoff_timestamp']}`."
            ),
            f"The seed is `{metrics['seed']}`.",
            f"The configuration hash is `{metrics['configuration_hash']}`.",
            f"The code hash is `{metrics['code_hash']}`.",
            (f"The learned model hash is `{metrics['models']['tfidf_linear_svm']['model_hash']}`."),
            (
                "The frozen pseudo-test labels use model "
                f"`{metrics['pseudo_test_label_provenance']['model_id']}`, revision "
                f"`{metrics['pseudo_test_label_provenance']['model_revision']}`, prompt "
                f"`{metrics['pseudo_test_label_provenance']['prompt_version']}`, temperature "
                f"`{metrics['pseudo_test_label_provenance']['temperature']}`, and sampling seed "
                f"`{metrics['pseudo_test_label_provenance']['seed']}`."
            ),
            "",
            f"The pilot input is `{pilot_rel}`.",
            f"It contains {metrics['inputs']['development_pilot']['unit_count']} records.",
            f"The frozen evaluation input is `{pseudo_rel}`.",
            f"It contains {metrics['inputs']['llm_pseudo_test']['unit_count']} records.",
            (
                "The learned baseline used "
                f"{metrics['models']['tfidf_linear_svm']['training_unit_count']} usable "
                "pilot records."
            ),
            f"The pilot sentiment classes are {sentiment_classes}.",
            "The learned sentiment classifier uses only these pilot classes.",
            "Neutral pseudo-test labels remain visible in the error samples and metrics.",
            "",
            (
                "The dictionary baseline has aspect micro F1 "
                f"{dictionary['aspect']['micro_f1']:.4f} and aspect macro F1 "
                f"{dictionary['aspect']['macro_f1']:.4f}."
            ),
            (
                "Its aspect-polarity pair micro F1 is "
                f"{dictionary['aspect_polarity_pair']['micro_f1']:.4f}."
            ),
            (
                "Its sentiment macro F1 on detected aspects is "
                f"{dictionary['sentiment_on_detected_aspects']['macro_f1']:.4f}."
            ),
            (
                "Its sentiment coverage is "
                f"{dictionary['sentiment_on_detected_aspects']['coverage']:.4f}."
            ),
            f"Its evidence offset validity is {dictionary['evidence']['valid_rate']:.4f}.",
            (
                "Its excluded-record evidence validity is "
                f"{dictionary['excluded_evidence']['valid_rate']:.4f}."
            ),
            (
                "Its sentiment F1 on supplied reference aspect spans is "
                f"{dictionary['sentiment_on_reference_aspects']['covered_macro_f1']:.4f}."
            ),
            "",
            (
                "The TF-IDF plus Linear SVM baseline has aspect micro F1 "
                f"{svm['aspect']['micro_f1']:.4f} and aspect macro F1 "
                f"{svm['aspect']['macro_f1']:.4f}."
            ),
            (
                "Its aspect-polarity pair micro F1 is "
                f"{svm['aspect_polarity_pair']['micro_f1']:.4f}."
            ),
            (
                "Its sentiment macro F1 on detected aspects is "
                f"{svm['sentiment_on_detected_aspects']['macro_f1']:.4f}."
            ),
            (f"Its sentiment coverage is {svm['sentiment_on_detected_aspects']['coverage']:.4f}."),
            f"Its evidence offset validity is {svm['evidence']['valid_rate']:.4f}.",
            (
                "Its excluded-record evidence validity is "
                f"{svm['excluded_evidence']['valid_rate']:.4f}."
            ),
            (
                "Its sentiment F1 on supplied reference aspect spans is "
                f"{svm['sentiment_on_reference_aspects']['covered_macro_f1']:.4f}."
            ),
            "",
            (
                f"The metrics exclude {svm['out_of_scope_units']} out-of-scope pseudo-test records "
                f"and {svm['needs_review_units']} records marked for review."
            ),
            (
                "The run keeps refusal counts and up to "
                f"{len(svm['error_samples'])} error samples per model."
            ),
            "Evidence spans remain slices of the original source text.",
            "The learned model uses grouped cross-validation for sigmoid calibration.",
            "It uses calibration only when each fold supports the required classes.",
            "A calibrated probability is an estimated class probability from that step.",
            "",
            "Run the command from the repository root:",
            "",
            "```bash",
            "PYTHONPATH=src python3 scripts/run_nlp_baselines.py",
            "PYTHONPATH=src python3 scripts/validate_nlp_baselines.py",
            "```",
            "",
            (f"The completed run manifest is `{_relative_or_absolute(manifest_path)}`."),
            "It is beside the generated artifacts in `artifacts/t2_3/`.",
            "The manifest blocks pseudo-test tuning and records artifact hashes.",
            "",
            (
                "Related documents are [the NLP baseline contract](../specs/"
                "nlp-baseline-contract.md), "
                "[the annotation contract](../specs/nlp-annotation-contract.md), "
                "[ADR-0013](../adr/0013-llm-only-annotation-policy.md), and "
                "[the phase 2 plan](../plans/phase-2-nlp.md)."
            ),
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=Path, default=DEFAULT_PILOT)
    parser.add_argument("--pseudo-test", type=Path, default=DEFAULT_PSEUDO_TEST)
    parser.add_argument("--pseudo-labels", type=Path, default=DEFAULT_PSEUDO_LABELS)
    parser.add_argument("--pseudo-manifest", type=Path, default=DEFAULT_PSEUDO_MANIFEST)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--run-id", default="t2.3-nlp-baselines-v1")
    args = parser.parse_args()
    run_baselines(
        pilot_path=args.pilot,
        pseudo_test_path=args.pseudo_test,
        pseudo_labels_path=args.pseudo_labels,
        pseudo_manifest_path=args.pseudo_manifest,
        config_path=args.config,
        output_dir=args.output_dir,
        manifest_path=args.manifest,
        report_path=args.report,
        run_id=args.run_id,
    )
    print(f"Wrote T2.3 run manifest to {args.manifest}")


if __name__ == "__main__":
    main()
