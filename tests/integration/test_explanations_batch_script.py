import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from scripts.run_explanations_batch import run_explanations
from scripts.run_hybrid_batch import run_hybrids

from trustrec.explanations.claims import ClaimThresholds
from trustrec.recommenders.hybrid_batch import HybridConfig

CUTOFF = "1970-01-01T00:00:00.100000+00:00"
ASPECTS = ("story", "graphics", "value")
SETTINGS = {
    "normalization": "minmax",
    "cold_start": "fallback",
    "cold_window_days": 1.0,
    "recency_window_days": 1.0,
}
H0 = HybridConfig(
    "h0_fixed_hybrid", mf_weight=0.1, graph_weight=0.2, recency_weight=0.5, aspect_weight=0.2
)
COMPONENTS = {
    "mf_parameters": {"epochs": 3, "embedding_dimensions": 4},
    "graph_parameters": {"max_iter": 10},
}


def _snapshot(tmp_path: Path) -> tuple[Path, Path]:
    rows, texts, evidence = [], [], []
    for i in range(300):
        row = {
            "review_id": f"r{i}",
            "user_id": f"u{i % 40:02d}",
            "item_id": f"i{(i * 7) % 23:02d}",
            "rating": float(1 + i % 5),
            "timestamp": i % 90,
        }
        rows.append(row)
        text = f"Review {i}. The {ASPECTS[i % 3]} is {('great', 'poor')[i % 2]} overall."
        texts.append({**row, "text": text})
        if i % 3:
            start = text.index("The")
            evidence.append(
                {
                    "review_id": row["review_id"],
                    "item_id": row["item_id"],
                    "author_id": row["user_id"],
                    "timestamp": row["timestamp"],
                    "aspect": ASPECTS[i % 3],
                    "sentiment": ("positive", "negative")[i % 2],
                    "confidence": 1.0,
                    "start": start,
                    "end": len(text),
                    "evidence": text[start:],
                }
            )
    targets = [
        {
            "review_id": f"t{i}",
            "user_id": f"u{i % 45:02d}",
            "item_id": f"i{(i * 3) % 23:02d}",
            "rating": 5.0,
            "timestamp": 100 + i,
        }
        for i in range(60)
    ]
    pq.write_table(pa.Table.from_pylist(rows), tmp_path / "train.parquet")
    pq.write_table(pa.Table.from_pylist(texts), tmp_path / "texts.parquet")
    pq.write_table(pa.Table.from_pylist(targets), tmp_path / "targets.parquet")
    evidence_path = tmp_path / "evidence.jsonl"
    evidence_path.write_text("\n".join(json.dumps(row) for row in evidence) + "\n")
    manifest = tmp_path / "snapshot.json"
    manifest.write_text(
        json.dumps(
            {
                "snapshot_id": "snap-1",
                "dataset_hash": "d" * 64,
                "cutoffs": {"t0": CUTOFF, "t1": CUTOFF},
                "artifact_paths": {
                    "train_interactions": "train.parquet",
                    "fit_interactions": "train.parquet",
                    "validation_targets": "targets.parquet",
                    "test_targets": "targets.parquet",
                    "review_texts": "texts.parquet",
                },
            }
        )
    )
    return manifest, evidence_path


def test_explanations_follow_the_saved_ranking_and_report_coverage(tmp_path: Path) -> None:
    manifest, evidence = _snapshot(tmp_path)
    ranking = tmp_path / "h0.jsonl"
    run_hybrids(
        manifest,
        cutoff_name="t1",
        evidence_path=evidence,
        configs=[H0],
        output_paths=[ranking],
        k=10,
        scorer_settings=SETTINGS,
        **COMPONENTS,
    )

    summary = run_explanations(
        manifest,
        evidence_path=evidence,
        ranking_path=ranking,
        config=H0,
        scorer_settings=SETTINGS,
        output_dir=tmp_path / "explanations",
        k=5,
        thresholds=ClaimThresholds(min_distinct_authors=2),
        workers=2,
        **COMPONENTS,
    )

    metrics = summary["metrics"]
    assert summary["checks"]["ranking_matches_saved_file"] is True
    assert summary["checks"]["max_profile_gap_vs_ranker"] < 1e-9
    assert metrics["recommendations"] == 5 * metrics["users"]
    assert sum(metrics["status_share"].values()) == pytest.approx(1.0)
    assert "correctness_pending" in metrics
    assert len(metrics["threshold_sensitivity"]) == 9
    faith = metrics["faithfulness"]["counts"]["all"]
    assert faith["not_applicable_aspect_not_in_score"] > 0
    assert faith.get("tested", 0) > 0
    rows = [
        json.loads(line)
        for line in (tmp_path / "explanations" / "recommendations.jsonl").read_text().splitlines()
    ]
    cold = [row for row in rows if not row["aspect_in_score"]]
    warm = [row for row in rows if row["aspect_in_score"]]
    assert cold and warm
    assert all(row["score_parts"].keys() == {"recent_popularity"} for row in cold)
    assert all(c["score_contribution"] == 0.0 for row in cold for c in row["claims"])
    tested = [row for row in warm if row["faithfulness"] and "faithful" in row["faithfulness"]]
    assert tested and all("random_score_delta" in row["faithfulness"] for row in tested)
    assert all(row["score"] == pytest.approx(sum(row["score_parts"].values())) for row in warm)
    claims = [
        json.loads(line)
        for line in (tmp_path / "explanations" / "claims.jsonl").read_text().splitlines()
    ]
    texts = {
        row["review_id"]: row["text"]
        for row in pq.read_table(tmp_path / "texts.parquet").to_pylist()
    }
    for claim in claims:
        for cited in claim["cited_evidence"]:
            assert texts[cited["review_id"]][cited["start"] : cited["end"]] == cited["text"]


def test_explanations_refuse_a_ranking_they_do_not_reproduce(tmp_path: Path) -> None:
    manifest, evidence = _snapshot(tmp_path)
    ranking = tmp_path / "h0.jsonl"
    run_hybrids(
        manifest,
        cutoff_name="t1",
        evidence_path=evidence,
        configs=[H0],
        output_paths=[ranking],
        k=10,
        scorer_settings=SETTINGS,
        **COMPONENTS,
    )
    rows = [json.loads(line) for line in ranking.read_text().splitlines()]
    rows[0]["ranked_item_ids"] = list(reversed(rows[0]["ranked_item_ids"]))
    ranking.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

    with pytest.raises(ValueError, match="differs"):
        run_explanations(
            manifest,
            evidence_path=evidence,
            ranking_path=ranking,
            config=H0,
            scorer_settings=SETTINGS,
            output_dir=tmp_path / "explanations",
            k=5,
            **COMPONENTS,
        )


def test_judge_audits_a_stratified_sample_without_showing_labels(tmp_path: Path) -> None:
    from scripts.run_explanation_judge import run_judge

    from trustrec.explanations.judge import JUDGE_PROMPT
    from trustrec.nlp.llm_labeler import ChatClient

    manifest, evidence = _snapshot(tmp_path)
    ranking = tmp_path / "h0.jsonl"
    run_hybrids(
        manifest,
        cutoff_name="t1",
        evidence_path=evidence,
        configs=[H0],
        output_paths=[ranking],
        k=10,
        scorer_settings=SETTINGS,
        **COMPONENTS,
    )
    run_explanations(
        manifest,
        evidence_path=evidence,
        ranking_path=ranking,
        config=H0,
        scorer_settings=SETTINGS,
        output_dir=tmp_path / "explanations",
        k=5,
        thresholds=ClaimThresholds(min_distinct_authors=2),
        **COMPONENTS,
    )
    seen = []

    def post(*args, **kwargs):
        body = kwargs["json"]
        assert body["messages"][0]["content"] == JUDGE_PROMPT
        message = json.loads(body["messages"][1]["content"])
        seen.append(message)
        answer = {
            "passages": [
                {"id": p["id"], "about_aspect": True, "polarity": "positive", "about_product": True}
                for p in message["passages"]
            ],
            "claim_supported": "yes",
        }

        class Response:
            status_code = 200

            def json(self):
                return {"choices": [{"message": {"content": json.dumps(answer)}}], "usage": {}}

        return Response()

    result = run_judge(
        manifest,
        explanations_dir=tmp_path / "explanations",
        output_path=tmp_path / "judge.json",
        client=ChatClient(
            api_key="k",
            model_id="judge",
            post=post,
            reasoning_effort=None,
            system_prompt=JUDGE_PROMPT,
        ),
        per_stratum=3,
        workers=2,
    )

    summary = result["summary"]["all_direction_strata"]
    assert result["traceability"]["passages"] == result["traceability"]["traceable"] > 0
    assert summary["judged"] == summary["claims"] == len(seen) > 0
    assert summary["claim_supported"]["yes"] == 1.0
    assert all("sentiment" not in passage for message in seen for passage in message["passages"])
    assert result["judge"]["prompt_version"] == "t7.3-claim-judge-v1"
