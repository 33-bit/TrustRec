"""TF-IDF and Linear SVM aspect and sentiment baseline."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from .dictionary import aspect_mentions, is_out_of_scope
from .records import ASPECTS, SENTIMENTS, parse_cutoff_timestamp, validate_annotation_records
from .text import split_text_units


@dataclass
class _ConstantClassifier:
    """A deterministic classifier for a class absent from development data."""

    value: Any
    classes_: np.ndarray | None = None
    calibration_status: str = "constant"

    def fit(self, _x: Any, _y: Any) -> _ConstantClassifier:
        self.classes_ = np.asarray(sorted(set(_y))) if len(set(_y)) else np.asarray([self.value])
        return self

    def predict(self, x: Any) -> np.ndarray:
        return np.full(x.shape[0], self.value)

    def predict_proba(self, x: Any) -> np.ndarray | None:
        if isinstance(self.value, (int, float, np.integer, np.floating)):
            value = float(self.value)
            return np.tile(np.array([[1.0 - value, value]]), (x.shape[0], 1))
        return None

    def decision_function(self, x: Any) -> np.ndarray:
        return np.full(x.shape[0], 1.0 if self.value else -1.0)


class _CalibratedLinearClassifier:
    """Linear SVM with grouped cross-validated sigmoid calibration."""

    def __init__(self, *, seed: int, c_value: float, folds: int) -> None:
        self.seed = seed
        self.c_value = c_value
        self.folds = folds
        self.base = LinearSVC(C=c_value, random_state=seed)
        self.calibrator: LogisticRegression | None = None
        self.calibration_status = "unavailable"
        self.classes_: np.ndarray | None = None

    def fit(
        self,
        features: Any,
        labels: Sequence[Any],
        groups: Sequence[str],
        *,
        raw_texts: Sequence[str] | None = None,
    ) -> _CalibratedLinearClassifier:
        y = np.asarray(labels)
        self.classes_ = np.unique(y)
        self.base.fit(features, y)
        if len(self.classes_) < 2 or len(set(groups)) < self.folds:
            self.calibration_status = "unavailable_insufficient_classes_or_groups"
            return self
        try:
            splitter = GroupKFold(n_splits=self.folds)
            folds = list(splitter.split(features, y, groups))
            expected_classes = set(self.classes_)
            if any(
                set(np.unique(y[train_index])) != expected_classes
                or set(np.unique(y[test_index])) != expected_classes
                for train_index, test_index in folds
            ):
                self.calibration_status = "unavailable_fold_class_support"
                return self
            if raw_texts is None:
                scores = cross_val_predict(
                    LinearSVC(C=self.c_value, random_state=self.seed),
                    features,
                    y,
                    cv=folds,
                    groups=groups,
                    method="decision_function",
                )
            else:
                scores = np.empty(len(y), dtype=float)
                for train_index, test_index in folds:
                    pipeline = Pipeline(
                        [
                            (
                                "tfidf",
                                TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True),
                            ),
                            ("svm", LinearSVC(C=self.c_value, random_state=self.seed)),
                        ]
                    )
                    pipeline.fit([raw_texts[index] for index in train_index], y[train_index])
                    scores[test_index] = pipeline.decision_function(
                        [raw_texts[index] for index in test_index]
                    )
            score_matrix = np.asarray(scores)
            if score_matrix.ndim == 1:
                score_matrix = score_matrix.reshape(-1, 1)
            self.calibrator = LogisticRegression(max_iter=1000, random_state=self.seed)
            self.calibrator.fit(score_matrix, y)
            self.calibration_status = "grouped_cv_sigmoid"
        except ValueError:
            self.calibrator = None
            self.calibration_status = "unavailable_fold_class_support"
        return self

    def predict(self, features: Any) -> np.ndarray:
        return self.base.predict(features)

    def predict_proba(self, features: Any) -> np.ndarray | None:
        if self.calibrator is None:
            return None
        scores = np.asarray(self.base.decision_function(features))
        if scores.ndim == 1:
            scores = scores.reshape(-1, 1)
        return self.calibrator.predict_proba(scores)


def _hash_model(config: dict[str, Any], records: Sequence[dict[str, Any]]) -> str:
    payload = {
        "config": config,
        "training_units": [
            {
                "unit_id": row["unit_id"],
                "source_text_sha256": row["source_text_sha256"],
                "labels": row["labels"],
            }
            for row in records
        ],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _classifier_fingerprint(classifier: Any) -> dict[str, Any]:
    """Serialize fitted classifier parameters for the model hash."""

    payload: dict[str, Any] = {"type": type(classifier).__name__}
    if hasattr(classifier, "value"):
        payload["value"] = classifier.value
        return payload
    payload["status"] = classifier.calibration_status
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


def _fitted_model_hash(
    config: dict[str, Any],
    records: Sequence[dict[str, Any]],
    aspect_vectorizer: TfidfVectorizer,
    aspect_classifiers: dict[str, Any],
    sentiment_vectorizer: TfidfVectorizer,
    sentiment_classifier: Any,
) -> str:
    payload = {
        "config": config,
        "training_units": [
            {
                "unit_id": row["unit_id"],
                "source_text_sha256": row["source_text_sha256"],
                "labels": row["labels"],
            }
            for row in records
        ],
        "aspect_vocabulary": sorted(aspect_vectorizer.vocabulary_.items()),
        "aspect_idf": aspect_vectorizer.idf_.tolist(),
        "aspect_classifiers": {
            aspect: _classifier_fingerprint(aspect_classifiers[aspect])
            for aspect in sorted(aspect_classifiers)
        },
        "sentiment_vocabulary": sorted(sentiment_vectorizer.vocabulary_.items()),
        "sentiment_idf": sentiment_vectorizer.idf_.tolist(),
        "sentiment_classifier": _classifier_fingerprint(sentiment_classifier),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _evidence_span(text: str, aspect: str, *, offset: int = 0) -> tuple[int, int]:
    mentions = aspect_mentions(text).get(aspect, [])
    if mentions:
        start, end = mentions[0]
        return start + offset, end + offset
    return offset, offset + len(text)


def _aspect_context(text: str, aspect: str) -> str:
    """Return the clause that contains the first target aspect mention."""

    mentions = aspect_mentions(text).get(aspect, [])
    if not mentions:
        return text
    start, end = mentions[0]
    for unit in split_text_units(text):
        if unit.start <= start and end <= unit.end:
            return unit.text
    return text


@dataclass
class TfidfSvmBaseline:
    """Fitted multi-label aspect and target-conditioned sentiment model."""

    aspect_vectorizer: TfidfVectorizer
    aspect_classifiers: dict[str, _ConstantClassifier | _CalibratedLinearClassifier]
    sentiment_vectorizer: TfidfVectorizer
    sentiment_classifier: _CalibratedLinearClassifier | _ConstantClassifier
    seed: int
    aspect_threshold: float
    sentiment_threshold: float
    config: dict[str, Any]
    training_provenance: dict[str, Any]

    name = "tfidf_linear_svm"

    @property
    def model_hash(self) -> str:
        return self.training_provenance["model_hash"]

    def _predict_aspects(self, text: str) -> list[tuple[str, float | None]]:
        features = self.aspect_vectorizer.transform([text])
        found: list[tuple[str, float | None]] = []
        for aspect in ASPECTS:
            classifier = self.aspect_classifiers[aspect]
            if classifier.calibration_status == "constant":
                continue
            probabilities = classifier.predict_proba(features)
            if probabilities is None:
                continue
            positive_probability = float(probabilities[0, 1])
            if positive_probability >= self.aspect_threshold:
                found.append((aspect, positive_probability))
        return found

    def predict_target_sentiment(self, text: str, aspect: str) -> dict[str, Any]:
        """Predict sentiment for a supplied aspect without aspect detection."""

        if aspect not in ASPECTS:
            raise ValueError(f"unsupported aspect: {aspect}")
        features = self.sentiment_vectorizer.transform(
            [f"aspect_{aspect} {_aspect_context(text, aspect)}"]
        )
        classifier = self.sentiment_classifier
        if classifier.calibration_status != "grouped_cv_sigmoid":
            return {
                "aspect": aspect,
                "polarity": None,
                "confidence": None,
                "confidence_kind": "unavailable",
                "sentiment_score": 0.0,
                "refusal_reason": "uncalibrated_sentiment",
            }
        polarity = str(classifier.predict(features)[0])
        probabilities = classifier.predict_proba(features)
        if probabilities is None:
            return {
                "aspect": aspect,
                "polarity": polarity,
                "confidence": None,
                "confidence_kind": "unavailable",
                "sentiment_score": 0.0,
                "refusal_reason": "uncalibrated_sentiment",
            }
        class_names = list(classifier.classes_)  # type: ignore[union-attr]
        confidence = float(probabilities[0, class_names.index(polarity)])
        if confidence < self.sentiment_threshold:
            return {
                "aspect": aspect,
                "polarity": None,
                "confidence": confidence,
                "confidence_kind": "calibrated_probability",
                "sentiment_score": 0.0,
                "refusal_reason": "uncertain_sentiment",
            }
        sentiment_score = float(
            probabilities[0, class_names.index("positive")] if "positive" in class_names else 0.0
        ) - float(
            probabilities[0, class_names.index("negative")] if "negative" in class_names else 0.0
        )
        return {
            "aspect": aspect,
            "polarity": polarity,
            "confidence": confidence,
            "confidence_kind": "calibrated_probability",
            "sentiment_score": sentiment_score,
            "refusal_reason": None,
        }

    def predict(
        self,
        text: str,
        *,
        review_id: str = "",
        unit_id: str = "",
    ) -> dict[str, Any]:
        if not isinstance(text, str) or not text.strip():
            return {
                "model": self.name,
                "review_id": review_id,
                "unit_id": unit_id,
                "text": text,
                "labels": [],
                "refusal_reason": "empty_text",
            }
        if is_out_of_scope(text):
            return {
                "model": self.name,
                "review_id": review_id,
                "unit_id": unit_id,
                "text": text,
                "labels": [],
                "refusal_reason": "out_of_scope",
            }

        clause_predictions: list[tuple[str, int, int, float | None]] = []
        for clause in split_text_units(text, review_id=review_id, unit_id_prefix=unit_id or "unit"):
            for aspect, probability in self._predict_aspects(clause.text):
                clause_predictions.append((aspect, clause.start, clause.end, probability))
        if not clause_predictions:
            return {
                "model": self.name,
                "review_id": review_id,
                "unit_id": unit_id,
                "text": text,
                "labels": [],
                "refusal_reason": "no_aspect_prediction",
            }

        labels: list[dict[str, Any]] = []
        sentiment_features = self.sentiment_vectorizer.transform(
            [
                f"aspect_{aspect} {text[start:end]}"
                for aspect, start, end, _probability in clause_predictions
            ]
        )
        sentiment_classifier = self.sentiment_classifier
        if sentiment_classifier.calibration_status != "grouped_cv_sigmoid":
            return {
                "model": self.name,
                "review_id": review_id,
                "unit_id": unit_id,
                "text": text,
                "labels": [],
                "refusal_reason": "uncalibrated_sentiment",
            }
        probabilities = sentiment_classifier.predict_proba(sentiment_features)
        if probabilities is None:
            return {
                "model": self.name,
                "review_id": review_id,
                "unit_id": unit_id,
                "text": text,
                "labels": [],
                "refusal_reason": "uncalibrated_sentiment",
            }
        predicted_polarities = sentiment_classifier.predict(sentiment_features)

        for index, (aspect, clause_start, clause_end, aspect_probability) in enumerate(
            clause_predictions
        ):
            polarity = str(predicted_polarities[index])
            sentiment_probability: float | None = None
            sentiment_score = 0.0
            row = probabilities[index]
            class_names = list(sentiment_classifier.classes_)  # type: ignore[union-attr]
            sentiment_probability = float(row[class_names.index(str(polarity))])
            if sentiment_probability < self.sentiment_threshold:
                continue
            sentiment_score = float(
                row[class_names.index("positive")] if "positive" in class_names else 0.0
            ) - float(row[class_names.index("negative")] if "negative" in class_names else 0.0)
            start, end = _evidence_span(text[clause_start:clause_end], aspect, offset=clause_start)
            label: dict[str, Any] = {
                "aspect": aspect,
                "polarity": polarity,
                "start": start,
                "end": end,
                "evidence": text[start:end],
                "sentiment_score": sentiment_score,
            }
            if aspect_probability is not None:
                label["aspect_probability"] = aspect_probability
            if sentiment_probability is not None:
                label["confidence"] = sentiment_probability
                label["confidence_kind"] = "calibrated_probability"
            else:
                label["confidence_kind"] = "decision_label"
            labels.append(label)

        refusal_reason = None if labels else "uncertain_sentiment"
        if not labels and not clause_predictions:
            refusal_reason = (
                "unsupported_aspect_classifier"
                if len(self.training_provenance.get("unsupported_aspects", [])) == len(ASPECTS)
                else "no_aspect_prediction"
            )
        return {
            "model": self.name,
            "review_id": review_id,
            "unit_id": unit_id,
            "text": text,
            "labels": labels,
            "refusal_reason": refusal_reason,
        }


def train_tfidf_svm(
    records: Sequence[dict[str, Any]],
    *,
    seed: int = 7,
    cutoff_timestamp: str | None = None,
    c_value: float = 1.0,
    calibration_folds: int = 3,
    aspect_threshold: float = 0.5,
    sentiment_threshold: float = 0.6,
) -> TfidfSvmBaseline:
    """Fit the learned baseline from development-pilot records only."""

    validated = validate_annotation_records(
        records,
        expected_role="development_pilot",
        expected_status="llm_silver",
    )
    if cutoff_timestamp is not None:
        cutoff = parse_cutoff_timestamp(cutoff_timestamp)
        if any(row["timestamp"] >= cutoff for row in validated):
            raise ValueError("training record timestamp must be before the cutoff")
    usable = [
        row for row in validated if not row.get("out_of_scope") and not row.get("needs_review")
    ]
    if len(usable) < 2:
        raise ValueError("at least two usable development records are required")
    if calibration_folds < 2:
        raise ValueError("calibration_folds must be at least 2")

    config: dict[str, Any] = {
        "family": "tfidf_linear_svm",
        "seed": seed,
        "ngram_range": [1, 2],
        "min_df": 1,
        "sublinear_tf": True,
        "c_value": c_value,
        "calibration": "grouped_cv_sigmoid",
        "calibration_folds": calibration_folds,
        "aspect_threshold": aspect_threshold,
        "sentiment_threshold": sentiment_threshold,
    }
    clauses: list[dict[str, Any]] = []
    for row in usable:
        for clause in split_text_units(
            row["text"], review_id=row["review_id"], unit_id_prefix=row["unit_id"]
        ):
            clause_labels = [
                label
                for label in row["labels"]
                if clause.start <= label["start"] <= label["end"] <= clause.end
            ]
            clauses.append(
                {
                    "text": clause.text,
                    "labels": clause_labels,
                    "group": (
                        f"duplicate|{row.get('item_id')}|{row['duplicate_group']}"
                        if row.get("duplicate_group") not in (None, "")
                        else f"review|{row['review_id']}"
                    ),
                }
            )
    if len(clauses) < 2:
        raise ValueError("at least two usable development clauses are required")
    aspect_vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
    aspect_features = aspect_vectorizer.fit_transform([clause["text"] for clause in clauses])
    aspect_classifiers: dict[str, _ConstantClassifier | _CalibratedLinearClassifier] = {}
    groups = [clause["group"] for clause in clauses]
    for aspect_index, aspect in enumerate(ASPECTS):
        labels = np.asarray(
            [
                int(any(label["aspect"] == aspect for label in clause["labels"]))
                for clause in clauses
            ]
        )
        if len(np.unique(labels)) < 2:
            aspect_classifiers[aspect] = _ConstantClassifier(int(labels[0])).fit(
                aspect_features, labels
            )
            continue
        aspect_classifiers[aspect] = _CalibratedLinearClassifier(
            seed=seed + aspect_index,
            c_value=c_value,
            folds=calibration_folds,
        ).fit(
            aspect_features,
            labels,
            groups,
            raw_texts=[clause["text"] for clause in clauses],
        )

    sentiment_rows = [
        (clause, label)
        for clause in clauses
        for label in clause["labels"]
        if label["polarity"] in SENTIMENTS
    ]
    sentiment_texts = [
        f"aspect_{label['aspect']} {clause['text']}" for clause, label in sentiment_rows
    ]
    sentiment_labels = [label["polarity"] for _, label in sentiment_rows]
    sentiment_groups = [clause["group"] for clause, _ in sentiment_rows]
    if not sentiment_labels:
        raise ValueError("development_pilot has no sentiment labels")
    sentiment_vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
    sentiment_features = sentiment_vectorizer.fit_transform(sentiment_texts)
    if len(set(sentiment_labels)) < 2:
        sentiment_classifier = _ConstantClassifier(sentiment_labels[0])
        sentiment_classifier.fit(sentiment_features, sentiment_labels)
    else:
        sentiment_classifier = _CalibratedLinearClassifier(
            seed=seed,
            c_value=c_value,
            folds=calibration_folds,
        ).fit(
            sentiment_features,
            sentiment_labels,
            sentiment_groups,
            raw_texts=sentiment_texts,
        )
        sentiment_classifier.classes_ = np.asarray(sorted(set(sentiment_labels)))

    training_provenance = {
        "split_role": "development_pilot",
        "label_status": "llm_silver",
        "source_snapshot_ids": sorted({row["source_snapshot_id"] for row in usable}),
        "source_snapshot_dataset_hashes": sorted(
            {row["source_snapshot_dataset_hash"] for row in usable}
        ),
        "source_cutoff_timestamps": sorted({row["source_cutoff_timestamp"] for row in usable}),
        "training_unit_count": len(usable),
        "training_clause_count": len(clauses),
        "training_unit_ids": [row["unit_id"] for row in usable],
        "absent_sentiment_classes": sorted(set(SENTIMENTS) - set(sentiment_labels)),
        "unsupported_aspects": sorted(
            aspect
            for aspect, classifier in aspect_classifiers.items()
            if getattr(classifier, "calibration_status", "constant") != "grouped_cv_sigmoid"
        ),
        "calibration": {
            **{
                aspect: {
                    "status": getattr(classifier, "calibration_status", "constant"),
                    "supported": getattr(classifier, "calibration_status", "constant")
                    == "grouped_cv_sigmoid",
                }
                for aspect, classifier in aspect_classifiers.items()
            },
            "sentiment": {
                "status": getattr(sentiment_classifier, "calibration_status", "constant"),
                "supported": getattr(sentiment_classifier, "calibration_status", "constant")
                == "grouped_cv_sigmoid",
            },
        },
    }
    training_provenance["model_hash"] = _fitted_model_hash(
        config,
        usable,
        aspect_vectorizer,
        aspect_classifiers,
        sentiment_vectorizer,
        sentiment_classifier,
    )
    return TfidfSvmBaseline(
        aspect_vectorizer=aspect_vectorizer,
        aspect_classifiers=aspect_classifiers,
        sentiment_vectorizer=sentiment_vectorizer,
        sentiment_classifier=sentiment_classifier,
        seed=seed,
        aspect_threshold=aspect_threshold,
        sentiment_threshold=sentiment_threshold,
        config=config,
        training_provenance=training_provenance,
    )
