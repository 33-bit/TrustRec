"""Build and verify deterministic report and reproducibility bundles."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any


class BundleError(ValueError):
    """Raised when a report bundle input or output breaks its contract."""


IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
CLAIM_KINDS = {"measured", "design", "limitation", "synthetic"}
EVIDENCE_ROLES = {"snapshot", "recommendation_test", "llm_pseudo_test", "synthetic", "design"}
AVAILABILITY = {"required", "external"}
RUN_AVAILABILITY = {"available", "external_pending"}
MISSING_HASH_VALUES = {None, "", "unavailable"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _identifier(value: Any, field: str) -> str:
    if not isinstance(value, str) or not IDENTIFIER_RE.fullmatch(value):
        raise BundleError(f"{field} must use letters, numbers, periods, underscores, or hyphens")
    return value


def _as_mapping(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise BundleError(f"{field} must be an object")
    return dict(value)


def _load_json(path: Path, field: str) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise BundleError(f"{field} JSON values must be finite: {value}")

    try:
        value = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)
    except (OSError, json.JSONDecodeError) as error:
        raise BundleError(f"{field} is not a readable JSON object: {path}") from error
    return _as_mapping(value, field)


def _resolve_path(root: Path, value: Any, field: str) -> tuple[str, Path]:
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise BundleError(f"{field} must be a relative path inside the repository root")
    relative = Path(value)
    if ".." in relative.parts:
        raise BundleError(f"{field} must stay inside the repository root")
    unresolved = root / relative
    cursor = root
    for part in relative.parts:
        cursor /= part
        if cursor.is_symlink():
            raise BundleError(f"{field} cannot use a symbolic link")
    candidate = unresolved.resolve()
    root_resolved = root.resolve()
    try:
        candidate.relative_to(root_resolved)
    except ValueError as error:
        raise BundleError(f"{field} must stay inside the repository root") from error
    return relative.as_posix(), candidate


def _path_value(root: Path, value: Any, field: str) -> tuple[str, Path]:
    relative, resolved = _resolve_path(root, value, field)
    return relative, resolved


def _load_spec(spec_path: Path) -> dict[str, Any]:
    if not spec_path.is_file():
        raise FileNotFoundError(f"report specification does not exist: {spec_path}")
    spec = _load_json(spec_path, "report specification")
    if spec.get("schema_version") != 1:
        raise BundleError("report specification schema_version must be 1")
    _identifier(spec.get("bundle_id"), "bundle_id")
    if not isinstance(spec.get("title"), str) or not spec["title"].strip():
        raise BundleError("title must be a non-empty string")
    return spec


def _records(spec: Mapping[str, Any], field: str) -> list[dict[str, Any]]:
    value = spec.get(field, [])
    if not isinstance(value, list):
        raise BundleError(f"{field} must be a list")
    return [_as_mapping(item, f"{field}[{index}]") for index, item in enumerate(value)]


def _unique_ids(records: Sequence[Mapping[str, Any]], field: str, id_field: str) -> None:
    ids: list[str] = []
    for index, record in enumerate(records):
        ids.append(_identifier(record.get(id_field), f"{field}[{index}].{id_field}"))
    if len(ids) != len(set(ids)):
        raise BundleError(f"{field} {id_field} values must be unique")


def _artifact_records(
    spec: Mapping[str, Any], root: Path
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    environment = _records(spec, "environment_files")
    artifacts = _records(spec, "artifacts")
    _unique_ids(environment, "environment_files", "artifact_id")
    _unique_ids(artifacts, "artifacts", "artifact_id")
    all_records: list[dict[str, Any]] = []
    for is_environment, records in ((True, environment), (False, artifacts)):
        for raw in records:
            record = dict(raw)
            artifact_id = _identifier(record.get("artifact_id"), "artifact_id")
            if record.get("availability", "required") not in AVAILABILITY:
                raise BundleError(f"artifact {artifact_id} has an invalid availability")
            if is_environment and record.get("availability", "required") != "required":
                raise BundleError(f"environment file {artifact_id} must be required")
            relative, resolved = _path_value(
                root, record.get("path"), f"artifact {artifact_id}.path"
            )
            expected = record.get("sha256")
            if (
                expected in MISSING_HASH_VALUES
                and record.get("availability", "required") == "required"
            ):
                raise BundleError(f"artifact {artifact_id} requires a sha256")
            if expected not in MISSING_HASH_VALUES and (
                not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected)
            ):
                raise BundleError(
                    f"artifact {artifact_id}.sha256 must be a lowercase SHA-256 value"
                )
            record.update(
                {
                    "artifact_id": artifact_id,
                    "path": relative,
                    "resolved_path": resolved,
                    "is_environment": is_environment,
                    "availability": record.get("availability", "required"),
                    "expected_sha256": expected,
                }
            )
            all_records.append(record)
    by_id: dict[str, dict[str, Any]] = {}
    for record in all_records:
        artifact_id = record["artifact_id"]
        if artifact_id in by_id:
            raise BundleError(f"artifact_id {artifact_id} is used more than once")
        path = record["resolved_path"]
        exists = path.is_file()
        if not exists and record["availability"] == "required":
            raise FileNotFoundError(f"required report artifact does not exist: {path}")
        actual = _sha256(path) if exists else None
        expected = record["expected_sha256"]
        if exists and expected not in MISSING_HASH_VALUES and actual != expected:
            raise BundleError(f"artifact {artifact_id} sha256 does not match: {record['path']}")
        record["exists"] = exists
        record["sha256"] = actual
        record["hash_status"] = (
            "matched"
            if exists and expected not in MISSING_HASH_VALUES
            else ("available_without_expected_hash" if exists else "unavailable")
        )
        by_id[artifact_id] = record
    return by_id, all_records


def _json_payload(record: Mapping[str, Any]) -> dict[str, Any] | None:
    if not record["exists"] or Path(record["path"]).suffix.lower() != ".json":
        return None
    return _load_json(record["resolved_path"], f"artifact {record['artifact_id']}")


def _json_pointer(value: Any, pointer: Any, field: str) -> Any:
    if not isinstance(pointer, str) or (pointer and not pointer.startswith("/")):
        raise BundleError(f"{field}.pointer must be an RFC 6901 JSON pointer")
    current = value
    if pointer == "":
        return current
    for raw_token in pointer[1:].split("/"):
        token = raw_token.replace("~1", "/").replace("~0", "~")
        if isinstance(current, Mapping) and token in current:
            current = current[token]
        elif isinstance(current, list) and token.isdigit() and int(token) < len(current):
            current = current[int(token)]
        else:
            raise BundleError(f"{field}.pointer does not resolve: {pointer}")
    return current


def _scalar(value: Any, field: str) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        raise BundleError(f"{field}.value must be finite")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise BundleError(f"{field}.pointer must resolve to a scalar value")


def _first_cutoff(payload: Mapping[str, Any] | None) -> Any:
    if not payload:
        return None
    cutoffs = payload.get("cutoffs")
    if isinstance(cutoffs, Mapping):
        return cutoffs.get("t1") or cutoffs.get("t0")
    return (
        payload.get("cutoff_timestamp")
        or payload.get("source_cutoff_timestamp")
        or payload.get("t1")
        or payload.get("t0")
        or payload.get("cutoff")
    )


def _manifest_summary(
    run: Mapping[str, Any], manifest: Mapping[str, Any] | None, metrics: Mapping[str, Any] | None
) -> dict[str, Any]:
    source = manifest or {}
    metric_source = metrics or {}
    snapshot_id = source.get("snapshot_id") or metric_source.get("snapshot_id")
    dataset_hash = source.get("dataset_hash") or metric_source.get("dataset_hash")
    cutoffs = source.get("cutoffs") or metric_source.get("cutoffs")
    if cutoffs is None:
        cutoff = source.get("source_cutoff_timestamp") or metric_source.get(
            "source_cutoff_timestamp"
        )
        cutoffs = {"cutoff": cutoff} if cutoff else {}
    configuration_hash = source.get("configuration_hash") or source.get("config_sha256")
    configuration_hash = configuration_hash or metric_source.get("configuration_hash")
    model_hashes = source.get("model_hashes") or metric_source.get("model_hashes") or {}
    metric_models = metric_source.get("models")
    if metric_models and not isinstance(metric_models, Mapping):
        raise BundleError(f"run {run['run_id']} metrics models must be an object")
    if metric_models and not model_hashes:
        model_hashes = {
            model_id: payload.get("metadata", {}).get("model_hash") or payload.get("model_hash")
            for model_id, payload in metric_models.items()
            if isinstance(payload, Mapping)
            and (payload.get("metadata", {}).get("model_hash") or payload.get("model_hash"))
        }
    metric_evaluation = metric_source.get("evaluation", {})
    if not isinstance(metric_evaluation, Mapping):
        raise BundleError(f"run {run['run_id']} metrics evaluation must be an object")
    metric_bootstrap = metric_source.get("bootstrap", {})
    if not isinstance(metric_bootstrap, Mapping):
        raise BundleError(f"run {run['run_id']} metrics bootstrap must be an object")
    llm_provenance = metric_source.get("pseudo_test_label_provenance") or source.get(
        "pseudo_test_label_provenance", {}
    )
    if not isinstance(llm_provenance, Mapping):
        raise BundleError(f"run {run['run_id']} LLM provenance must be an object")
    return {
        "run_id": str(run["run_id"]),
        "evidence_role": run["evidence_role"],
        "command_id": run["command_id"],
        "manifest_artifact": run["manifest_artifact"],
        "metrics_artifact": run.get("metrics_artifact"),
        "availability": run.get("availability", "available"),
        "status": source.get("status") or metric_source.get("status") or "unavailable",
        "snapshot_id": snapshot_id or "unavailable",
        "dataset_hash": dataset_hash or "unavailable",
        "cutoffs": cutoffs,
        "protocol_version": source.get("protocol_version")
        or metric_source.get("protocol_version")
        or "unavailable",
        "seed": source.get("seed", metric_source.get("seed", "unavailable")),
        "seeds": source.get("seeds") or metric_source.get("model_seeds") or [],
        "configuration_hash": configuration_hash or "unavailable",
        "model_hashes": model_hashes,
        "code_revision": source.get("code_revision")
        or metric_source.get("code_revision")
        or "unavailable",
        "llm_provenance": dict(llm_provenance),
        "cutoff_timestamp": metric_source.get("cutoff_timestamp")
        or source.get("cutoff_timestamp")
        or source.get("source_cutoff_timestamp")
        or _first_cutoff(source),
        "evaluation": {
            key: value
            for key, value in metric_evaluation.items()
            if key in {"candidate_set_hash", "eligible_users", "exclusions"}
        },
        "bootstrap": metric_bootstrap,
        "candidate_rule": metric_source.get("candidate_rule", "unavailable"),
        "target_rule": metric_source.get("target_rule", "unavailable"),
        "model_results": {
            model_id: {
                key: model[key]
                for key in ("metrics", "bootstrap", "counts", "slices", "resource")
                if key in model
            }
            for model_id, model in (metric_models or {}).items()
            if isinstance(model, Mapping) and run["evidence_role"] == "recommendation_test"
        },
    }


def _validate_run(
    run: Mapping[str, Any],
    artifacts: Mapping[str, Mapping[str, Any]],
    commands: Mapping[str, Mapping[str, Any]],
    repository_root: Path,
) -> tuple[dict[str, Any], dict[str, Any] | None, dict[str, Any] | None]:
    run_id = _identifier(run.get("run_id"), "run_id")
    role = run.get("evidence_role")
    if role not in EVIDENCE_ROLES:
        raise BundleError(f"run {run_id} has an invalid evidence_role")
    command_id = _identifier(run.get("command_id"), f"run {run_id}.command_id")
    if command_id not in commands:
        raise BundleError(f"run {run_id} refers to an unknown command")
    manifest_id = _identifier(run.get("manifest_artifact"), f"run {run_id}.manifest_artifact")
    if manifest_id not in artifacts:
        raise BundleError(f"run {run_id} refers to an unknown manifest artifact")
    manifest_record = artifacts[manifest_id]
    availability = run.get("availability", "available")
    if availability not in RUN_AVAILABILITY:
        raise BundleError(f"run {run_id} has an invalid availability")
    manifest = _json_payload(manifest_record)
    if manifest is None:
        if availability != "external_pending":
            raise BundleError(f"run {run_id} needs an available JSON manifest")
        metrics_id = run.get("metrics_artifact")
        if metrics_id is not None:
            metrics_id = _identifier(metrics_id, f"run {run_id}.metrics_artifact")
            if metrics_id not in artifacts:
                raise BundleError(f"run {run_id} refers to an unknown metrics artifact")
            if artifacts[metrics_id]["exists"]:
                raise BundleError(f"run {run_id} is pending but its metrics artifact exists")
        return _manifest_summary(run, None, None), None, None
    metrics_id = run.get("metrics_artifact")
    metrics: dict[str, Any] | None = None
    if metrics_id is not None:
        metrics_id = _identifier(metrics_id, f"run {run_id}.metrics_artifact")
        if metrics_id not in artifacts:
            raise BundleError(f"run {run_id} refers to an unknown metrics artifact")
        metrics = _json_payload(artifacts[metrics_id])
        if metrics is None:
            raise BundleError(f"run {run_id} needs an available JSON metrics artifact")
    if manifest is not None:
        actual_run_id = manifest.get("run_id")
        if role == "snapshot":
            actual_run_id = actual_run_id or manifest.get("snapshot_id")
        if role == "synthetic" and actual_run_id is None:
            metric_run_ids = {
                row.get("run_id") for row in manifest.get("metrics", []) if isinstance(row, Mapping)
            }
            actual_run_id = run_id if metric_run_ids == {run_id} else None
        if actual_run_id != run_id:
            raise BundleError(f"run {run_id} does not match its manifest run_id")
        _identifier(manifest.get("snapshot_id"), f"run {run_id}.snapshot_id")
        if not re.fullmatch(r"[0-9a-f]{64}", str(manifest.get("dataset_hash", ""))):
            raise BundleError(f"run {run_id} needs a SHA-256 dataset_hash")
        if role != "snapshot" and manifest.get("status") != "complete":
            raise BundleError(f"run {run_id} manifest status is not complete")
        if metrics is not None:
            metric_run_id = metrics.get("run_id")
            if role == "synthetic" and metric_run_id is None:
                metric_ids = {row.get("run_id") for row in metrics.get("metrics", [])}
                metric_run_id = run_id if metric_ids == {run_id} else None
            if metric_run_id != run_id:
                raise BundleError(f"run {run_id} does not match its metrics run_id")
            for field in ("snapshot_id", "dataset_hash"):
                if manifest.get(field) and metrics.get(field) and manifest[field] != metrics[field]:
                    raise BundleError(f"run {run_id} has conflicting {field}")
            for field in (
                "cutoffs",
                "cutoff_name",
                "cutoff_timestamp",
                "source_cutoff_timestamp",
                "protocol_version",
                "seed",
                "configuration_hash",
            ):
                if field in manifest and field in metrics and manifest[field] != metrics[field]:
                    raise BundleError(f"run {run_id} has conflicting {field}")
            if "seeds" in manifest and manifest["seeds"] != metrics.get("model_seeds"):
                raise BundleError(f"run {run_id} has conflicting model seeds")
            declared_metrics_hash = manifest.get("artifact_sha256", {}).get("metrics")
            if (
                not declared_metrics_hash
                or artifacts[metrics_id]["sha256"] != declared_metrics_hash
            ):
                raise BundleError(f"run {run_id} metrics hash does not match its manifest")
            metrics_path = manifest.get("metrics_path")
            if not isinstance(metrics_path, str):
                raise BundleError(f"run {run_id} is missing metrics_path")
            _, manifest_metrics_path = _resolve_path(
                repository_root, metrics_path, f"run {run_id}.metrics_path"
            )
            if manifest_metrics_path != artifacts[metrics_id]["resolved_path"]:
                raise BundleError(f"run {run_id} metrics_path does not match its metrics artifact")
        if role == "llm_pseudo_test":
            frozen = manifest.get(
                "frozen_pseudo_test", metrics.get("frozen_pseudo_test") if metrics else None
            )
            tuning = manifest.get(
                "pseudo_test_tuning_allowed",
                metrics.get("pseudo_test_tuning_allowed") if metrics else None,
            )
            if frozen is not True or tuning is not False:
                raise BundleError(f"run {run_id} does not lock the frozen pseudo-test")
            required_provenance = {
                "model_id",
                "model_revision",
                "prompt_version",
                "temperature",
                "seed",
            }
            if not required_provenance <= set(
                _manifest_summary(run, manifest, metrics)["llm_provenance"]
            ):
                raise BundleError(f"run {run_id} is missing LLM label provenance")
        if role == "recommendation_test" and metrics is not None:
            if not metrics.get("candidate_rule") or not metrics.get("target_rule"):
                raise BundleError(f"run {run_id} metrics must record candidate and target rules")
            evaluation = metrics.get("evaluation", {})
            if not isinstance(evaluation, Mapping) or not evaluation.get("candidate_set_hash"):
                raise BundleError(f"run {run_id} metrics must record candidate_set_hash")
            metric_models = metrics.get("models")
            if not isinstance(metric_models, Mapping) or not metric_models:
                raise BundleError(f"run {run_id} metrics must contain model records")
            eligible_counts = {
                model.get("counts", {}).get("eligible_users")
                for model in metric_models.values()
                if isinstance(model, Mapping)
            }
            if None in eligible_counts or not eligible_counts or len(eligible_counts) != 1:
                raise BundleError(f"run {run_id} models do not share eligible-user count")
            candidate_hashes = {
                model.get("metadata", {}).get("candidate_set_hash")
                for model in metric_models.values()
                if isinstance(model, Mapping)
            }
            if candidate_hashes != {evaluation["candidate_set_hash"]}:
                raise BundleError(f"run {run_id} models do not share candidate set")
            if evaluation.get("eligible_users") not in eligible_counts:
                raise BundleError(
                    f"run {run_id} top-level eligible-user count does not match models"
                )
    return _manifest_summary(run, manifest, metrics), manifest, metrics


def _validate_claims(
    claims: Sequence[Mapping[str, Any]],
    artifacts: Mapping[str, Mapping[str, Any]],
    runs: Mapping[str, Mapping[str, Any]],
    run_payloads: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    _unique_ids(claims, "claims", "claim_id")
    resolved_claims: list[dict[str, Any]] = []
    result_rows: list[dict[str, Any]] = []
    for index, raw_claim in enumerate(claims):
        claim = dict(raw_claim)
        claim_id = _identifier(claim.get("claim_id"), f"claims[{index}].claim_id")
        kind = claim.get("kind")
        if kind not in CLAIM_KINDS:
            raise BundleError(f"claim {claim_id} has an invalid kind")
        if not isinstance(claim.get("statement"), str) or not claim["statement"].strip():
            raise BundleError(f"claim {claim_id} needs a statement")
        run_id = _identifier(claim.get("run_id"), f"claim {claim_id}.run_id")
        if run_id not in runs:
            raise BundleError(f"claim {claim_id} refers to an unknown run")
        summary = run_payloads[run_id]
        if summary["availability"] == "external_pending" and kind != "limitation":
            raise BundleError(f"claim {claim_id} cannot use an unavailable run")
        role = summary["evidence_role"]
        if role in {"synthetic", "design"} and kind not in {role, "limitation"}:
            raise BundleError(f"claim {claim_id} cannot use {role} evidence as a measurement")
        if kind == "synthetic" and role != "synthetic":
            raise BundleError(f"claim {claim_id} must use a synthetic run")
        if kind == "design" and role != "design":
            raise BundleError(f"claim {claim_id} must use a design run")
        evidence = claim.get("evidence")
        if not isinstance(evidence, list) or not evidence:
            raise BundleError(f"claim {claim_id} needs evidence")
        resolved_evidence: list[dict[str, Any]] = []
        for evidence_index, raw_evidence in enumerate(evidence):
            item = _as_mapping(raw_evidence, f"claim {claim_id}.evidence[{evidence_index}]")
            artifact_id = _identifier(item.get("artifact_id"), "evidence artifact_id")
            if artifact_id not in artifacts:
                raise BundleError(f"claim {claim_id} refers to an unknown artifact")
            if artifact_id not in {summary["manifest_artifact"], summary["metrics_artifact"]}:
                raise BundleError(f"claim {claim_id} evidence does not belong to its run")
            label = item.get("label")
            if not isinstance(label, str) or not label.strip():
                raise BundleError(f"claim {claim_id} evidence needs a label")
            record = artifacts[artifact_id]
            pointer = item.get("pointer", "")
            value: Any = "unavailable"
            evidence_status = "unavailable"
            if record["exists"]:
                payload = _json_payload(record)
                if payload is None:
                    if pointer != "":
                        raise BundleError(f"claim {claim_id} points into a non-JSON artifact")
                    value = record["path"]
                else:
                    value = _scalar(
                        _json_pointer(payload, pointer, f"claim {claim_id}.evidence"),
                        f"claim {claim_id}.evidence",
                    )
                evidence_status = "available"
            elif kind != "limitation":
                raise BundleError(f"claim {claim_id} uses an unavailable artifact")
            copied_path = record.get("bundle_path")
            if copied_path:
                artifact_link = copied_path
            else:
                artifact_link = f"bundle.manifest.json#artifact-{artifact_id}"
            resolved = {
                "artifact_id": artifact_id,
                "label": label,
                "pointer": pointer,
                "value": value,
                "status": evidence_status,
                "sha256": record["sha256"] or record["expected_sha256"] or "unavailable",
                "artifact_path": record["path"],
                "artifact_link": artifact_link,
                "run_id": run_id,
            }
            resolved_evidence.append(resolved)
            result_rows.append(
                {
                    "claim_id": claim_id,
                    "claim_kind": kind,
                    "statement": claim["statement"],
                    "evidence_label": label,
                    "artifact_id": artifact_id,
                    "artifact_path": record["path"],
                    "pointer": pointer,
                    "value": value,
                    "evidence_status": evidence_status,
                    "run_id": run_id,
                    "evidence_role": role,
                    "model_id": claim.get("model_id", ""),
                    "metric": claim.get("metric", ""),
                    "snapshot_id": summary["snapshot_id"],
                    "cutoff": summary.get("cutoff_timestamp") or "unavailable",
                    "seed": summary["seed"],
                    "protocol_version": summary["protocol_version"],
                    "configuration_hash": summary["configuration_hash"],
                    "dataset_hash": summary["dataset_hash"],
                    "model_hash": summary["model_hashes"].get(claim.get("model_id"), "unavailable"),
                    "eligible_users": summary["model_results"]
                    .get(claim.get("model_id"), {})
                    .get("counts", {})
                    .get("eligible_users", "unavailable"),
                    "candidate_set_hash": summary["evaluation"].get(
                        "candidate_set_hash", "unavailable"
                    ),
                    "ci_lower": summary["model_results"]
                    .get(claim.get("model_id"), {})
                    .get("bootstrap", {})
                    .get(claim.get("metric"), {})
                    .get("lower", "unavailable"),
                    "ci_upper": summary["model_results"]
                    .get(claim.get("model_id"), {})
                    .get("bootstrap", {})
                    .get(claim.get("metric"), {})
                    .get("upper", "unavailable"),
                    "uncertainty_method": summary["bootstrap"].get("method", "unavailable"),
                    "units": claim.get("units", "ratio" if claim.get("metric") else "count"),
                    "declared_model_seeds": summary["seeds"],
                    "candidate_rule": summary["candidate_rule"],
                    "target_rule": summary["target_rule"],
                    "slices": summary["model_results"]
                    .get(claim.get("model_id"), {})
                    .get("slices", "unavailable"),
                    "resource": summary["model_results"]
                    .get(claim.get("model_id"), {})
                    .get("resource", "unavailable"),
                    "exclusions": summary["evaluation"].get("exclusions", "unavailable"),
                    "llm_model_id": summary["llm_provenance"].get("model_id", "unavailable"),
                    "llm_model_revision": summary["llm_provenance"].get(
                        "model_revision", "unavailable"
                    ),
                    "llm_prompt_version": summary["llm_provenance"].get(
                        "prompt_version", "unavailable"
                    ),
                    "llm_temperature": summary["llm_provenance"].get("temperature", "unavailable"),
                    "llm_seed": summary["llm_provenance"].get("seed", "unavailable"),
                }
            )
        rendered = dict(claim)
        rendered["evidence"] = resolved_evidence
        rendered["manifest_link"] = f"bundle.manifest.json#run-{run_id}"
        rendered["command_link"] = f"commands.md#{summary['command_id']}"
        rendered["command_id"] = summary["command_id"]
        rendered["evidence_role"] = role
        rendered["run"] = summary
        resolved_claims.append(rendered)
    return resolved_claims, result_rows


def _source_link(record: Mapping[str, Any]) -> str:
    return str(
        record.get("bundle_path") or f"bundle.manifest.json#artifact-{record['artifact_id']}"
    )


def _render_commands(commands: Sequence[Mapping[str, Any]]) -> str:
    lines = ["# Reproduction commands", "", "Run each command from the repository root.", ""]
    for command in commands:
        command_id = command["command_id"]
        lines.extend([f"## {command_id}", "", str(command["command"]), ""])
        prerequisites = command.get("prerequisites", [])
        if prerequisites:
            lines.append("Prerequisites:")
            lines.extend(f"- {item}" for item in prerequisites)
            lines.append("")
    return "\n".join(lines)


def _render_report(
    title: str,
    claims: Sequence[Mapping[str, Any]],
    commands: Sequence[Mapping[str, Any]],
    source_records: Sequence[Mapping[str, Any]],
) -> str:
    sections = {
        "measured": "Measured results",
        "design": "Design claims",
        "limitation": "Limits",
        "synthetic": "Synthetic demonstration",
    }
    lines = [f"# {title}", "", "This report uses only records in the reproducibility bundle.", ""]
    for kind in ("measured", "design", "limitation", "synthetic"):
        selected = [claim for claim in claims if claim["kind"] == kind]
        if not selected:
            continue
        lines.extend([f"## {sections[kind]}", ""])
        for claim in selected:
            lines.extend([f"### {claim['claim_id']}", "", claim["statement"], ""])
            links = [
                f"[run manifest]({claim['manifest_link']})",
                f"[reproduction command]({claim['command_link']})",
            ]
            evidence_links = []
            for evidence in claim["evidence"]:
                evidence_links.append(
                    f"[{evidence['label']}]({evidence['artifact_link']}) = "
                    f"`{json.dumps(evidence['value'], ensure_ascii=False)}`"
                )
            lines.append("Evidence: " + "; ".join(evidence_links) + ".")
            lines.append("Lineage: " + ", ".join(links) + ".")
            provenance = claim.get("run", {}).get("llm_provenance", {})
            if provenance:
                lines.append(
                    "LLM label provenance: "
                    f"model `{provenance.get('model_id', 'unavailable')}`, "
                    f"revision `{provenance.get('model_revision', 'unavailable')}`, "
                    f"prompt `{provenance.get('prompt_version', 'unavailable')}`, "
                    f"temperature `{provenance.get('temperature', 'unavailable')}`, "
                    f"seed `{provenance.get('seed', 'unavailable')}`."
                )
            lines.append("")
    lines.extend(["## Sources and environment", ""])
    for record in source_records:
        status = "available" if record["exists"] else "external and unavailable"
        lines.append(
            f"- `{record['artifact_id']}`: `{record['path']}` ({status}, "
            f"SHA-256 `{record['sha256'] or record['expected_sha256'] or 'unavailable'}`)."
        )
    lines.extend(
        [
            "",
            "## Reproduction",
            "",
            "Read [the command list](commands.md) from the repository root.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_results(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    columns = (
        "claim_id",
        "claim_kind",
        "statement",
        "evidence_label",
        "artifact_id",
        "artifact_path",
        "pointer",
        "value",
        "evidence_status",
        "run_id",
        "evidence_role",
        "model_id",
        "metric",
        "snapshot_id",
        "cutoff",
        "seed",
        "protocol_version",
        "configuration_hash",
        "dataset_hash",
        "model_hash",
        "eligible_users",
        "candidate_set_hash",
        "ci_lower",
        "ci_upper",
        "uncertainty_method",
        "units",
        "declared_model_seeds",
        "candidate_rule",
        "target_rule",
        "slices",
        "resource",
        "exclusions",
        "llm_model_id",
        "llm_model_revision",
        "llm_prompt_version",
        "llm_temperature",
        "llm_seed",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    column: json.dumps(row[column], ensure_ascii=False)
                    if isinstance(row[column], (dict, list))
                    else row[column]
                    for column in columns
                }
            )


def _copy_sources(
    output_dir: Path, records: Iterable[dict[str, Any]], manifest_ids: set[str]
) -> None:
    source_dir = output_dir / "sources"
    for record in records:
        copy = record["is_environment"] or record["artifact_id"] in manifest_ids
        if not copy or not record["exists"]:
            continue
        suffix = Path(record["path"]).suffix
        destination = source_dir / f"{record['artifact_id']}{suffix}"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(record["resolved_path"], destination)
        record["bundle_path"] = destination.relative_to(output_dir).as_posix()


def _file_hashes(root: Path, *, exclude: set[str]) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): _sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.relative_to(root).as_posix() not in exclude
    }


def _resolve_bundle_path(bundle_dir: Path, value: Any, field: str) -> Path:
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise BundleError(f"{field} must be a relative path inside the bundle")
    relative = Path(value)
    if ".." in relative.parts:
        raise BundleError(f"{field} must stay inside the bundle")
    cursor = bundle_dir
    for part in relative.parts:
        cursor /= part
        if cursor.is_symlink():
            raise BundleError(f"{field} cannot use a symbolic link")
    candidate = (bundle_dir / relative).resolve()
    try:
        candidate.relative_to(bundle_dir.resolve())
    except ValueError as error:
        raise BundleError(f"{field} must stay inside the bundle") from error
    return candidate


def _write_checksums(output_dir: Path) -> None:
    hashes = _file_hashes(output_dir, exclude={"SHA256SUMS"})
    if not hashes:
        raise BundleError("bundle has no files to checksum")
    lines = [f"{digest}  {relative}" for relative, digest in sorted(hashes.items())]
    (output_dir / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")


def package_report(
    spec_path: Path,
    output_dir: Path,
    *,
    repository_root: Path,
    code_revision: str = "unavailable",
) -> dict[str, Any]:
    """Validate the specification and publish one deterministic bundle."""

    spec_path = Path(spec_path)
    output_dir = Path(output_dir)
    repository_root = Path(repository_root).resolve()
    spec = _load_spec(spec_path)
    if output_dir.exists():
        raise FileExistsError(f"report bundle output already exists: {output_dir}")
    artifacts, records = _artifact_records(spec, repository_root)
    commands = _records(spec, "commands")
    _unique_ids(commands, "commands", "command_id")
    for command in commands:
        if not isinstance(command.get("command"), str) or not command["command"].strip():
            raise BundleError(f"command {command['command_id']} must contain command text")
        prerequisites = command.get("prerequisites", [])
        if not isinstance(prerequisites, list) or any(
            not isinstance(item, str) for item in prerequisites
        ):
            raise BundleError(
                f"command {command['command_id']} prerequisites must be a list of strings"
            )
    command_map = {record["command_id"]: record for record in commands}
    runs = _records(spec, "runs")
    _unique_ids(runs, "runs", "run_id")
    run_payloads: dict[str, dict[str, Any]] = {}
    for run in runs:
        summary, _, _ = _validate_run(run, artifacts, command_map, repository_root)
        run_payloads[summary["run_id"]] = summary
    run_map = {run_id: dict(summary) for run_id, summary in run_payloads.items()}
    claims = _records(spec, "claims")
    output_parent = output_dir.parent
    output_parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_parent))
    try:
        _copy_sources(temp_dir, records, {run["manifest_artifact"] for run in runs})
        resolved_claims, result_rows = _validate_claims(claims, artifacts, run_map, run_payloads)
        _write_json(temp_dir / "specification.json", spec)
        _write_json(temp_dir / "claims.json", {"schema_version": 1, "claims": resolved_claims})
        _write_results(temp_dir / "results.csv", result_rows)
        (temp_dir / "commands.md").write_text(_render_commands(commands), encoding="utf-8")
        (temp_dir / "report.md").write_text(
            _render_report(spec["title"], resolved_claims, commands, records), encoding="utf-8"
        )
        spec_hash = _sha256(spec_path)
        source_payload = {}
        for record in records:
            source_payload[record["artifact_id"]] = {
                key: value
                for key, value in record.items()
                if key
                in {
                    "artifact_id",
                    "path",
                    "availability",
                    "exists",
                    "sha256",
                    "expected_sha256",
                    "hash_status",
                    "is_environment",
                    "bundle_path",
                    "copy",
                }
            }
        bundle_manifest = {
            "schema_version": 1,
            "bundle_id": spec["bundle_id"],
            "title": spec["title"],
            "status": "complete",
            "code_revision": code_revision,
            "specification_path": "specification.json",
            "specification_sha256": spec_hash,
            "sources": source_payload,
            "commands": commands,
            "runs": list(run_payloads.values()),
            "claims": resolved_claims,
            "generated_files": {},
            "checksums_path": "SHA256SUMS",
        }
        _write_json(temp_dir / "bundle.manifest.json", bundle_manifest)
        bundle_manifest["generated_files"] = _file_hashes(
            temp_dir, exclude={"bundle.manifest.json", "SHA256SUMS"}
        )
        _write_json(temp_dir / "bundle.manifest.json", bundle_manifest)
        _write_checksums(temp_dir)
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temp_dir, output_dir)
        return json.loads((output_dir / "bundle.manifest.json").read_text(encoding="utf-8"))
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise


def verify_bundle(bundle_dir: Path) -> dict[str, Any]:
    """Verify hashes in a saved bundle and return its manifest."""

    bundle_dir = Path(bundle_dir)
    manifest_path = bundle_dir / "bundle.manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"bundle manifest does not exist: {manifest_path}")
    manifest = _load_json(manifest_path, "bundle manifest")
    if manifest.get("schema_version") != 1 or manifest.get("status") != "complete":
        raise BundleError("bundle manifest is not a complete schema version 1 bundle")
    generated = manifest.get("generated_files")
    if not isinstance(generated, Mapping):
        raise BundleError("bundle manifest generated_files must be an object")
    for relative, expected in generated.items():
        path = _resolve_bundle_path(bundle_dir, relative, "generated file path")
        if not path.is_file() or _sha256(path) != expected:
            raise BundleError(f"bundle generated file hash does not match: {relative}")
    source_records = manifest.get("sources", {})
    if not isinstance(source_records, Mapping):
        raise BundleError("bundle manifest sources must be an object")
    for artifact_id, record in source_records.items():
        if not isinstance(record, Mapping):
            raise BundleError(f"bundle source record is invalid: {artifact_id}")
        bundle_path = record.get("bundle_path")
        if bundle_path:
            source = _resolve_bundle_path(bundle_dir, bundle_path, "source bundle_path")
            if not source.is_file() or _sha256(source) != record.get("sha256"):
                raise BundleError(f"bundle source hash does not match: {artifact_id}")
    checksum_path = _resolve_bundle_path(
        bundle_dir, manifest.get("checksums_path", "SHA256SUMS"), "checksums_path"
    )
    if not checksum_path.is_file():
        raise FileNotFoundError(f"bundle checksums do not exist: {checksum_path}")
    lines = [
        line for line in checksum_path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    if not lines:
        raise BundleError("bundle checksum inventory is empty")
    inventory: dict[str, str] = {}
    for line in lines:
        if not line.strip():
            continue
        digest, separator, relative = line.partition("  ")
        if not separator or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise BundleError("bundle checksums contains an invalid line")
        if relative in inventory:
            raise BundleError(f"bundle checksum inventory repeats: {relative}")
        inventory[relative] = digest
        path = _resolve_bundle_path(bundle_dir, relative, "checksum path")
        if not path.is_file() or _sha256(path) != digest:
            raise BundleError(f"bundle checksum does not match: {relative}")
    expected_inventory = {
        path.relative_to(bundle_dir).as_posix()
        for path in bundle_dir.rglob("*")
        if path.is_file() and path.relative_to(bundle_dir).as_posix() != "SHA256SUMS"
    }
    if set(inventory) != expected_inventory:
        raise BundleError("bundle checksum inventory is incomplete")
    return manifest


def git_revision(repository_root: Path) -> str:
    """Return the current Git revision without making it a package dependency."""

    try:
        result = subprocess.run(
            ["git", "-C", str(repository_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"
    return result.stdout.strip() or "unavailable"
