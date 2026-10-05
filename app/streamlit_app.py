"""Streamlit presenter view for the prepared TrustRec demo bundle."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import streamlit as st
from app.demo import (
    DemoBundle,
    DemoLoadResult,
    DemoRecommendation,
    DemoValidationError,
    compare_rankings,
    evidence_for_item,
    load_demo_bundle_with_fallback,
    rank_recommendations,
)

MODEL_LABELS = {
    "b0_most_popular": "Baseline: most popular",
    "t0_trustrec": "TrustRec: adaptive gate",
}


@st.cache_resource(show_spinner=False)
def _load_bundle() -> DemoLoadResult:
    return load_demo_bundle_with_fallback()


def _model_label(model_id: str) -> str:
    return MODEL_LABELS.get(model_id, model_id)


def _aspect_names(bundle: DemoBundle, user_id: str) -> tuple[str, ...]:
    aspects = set(bundle.users[user_id].learned_aspect_weights)
    for item in bundle.items.values():
        aspects.update(item.aspect_scores)
    return tuple(sorted(aspects))


def _score_text(score_parts: Mapping[str, float]) -> str:
    return ", ".join(f"{name} {value:.3f}" for name, value in sorted(score_parts.items()))


def _ranking_rows(
    recommendations: Sequence[DemoRecommendation], changed_items: set[str]
) -> list[dict[str, object]]:
    return [
        {
            "Rank": recommendation.rank,
            "Item": recommendation.title,
            "Item ID": recommendation.item_id,
            "Score": round(recommendation.total_score, 4),
            "Score parts": _score_text(recommendation.score_parts),
            "Evidence": recommendation.evidence_state,
            "Moved": "Yes" if recommendation.item_id in changed_items else "No",
        }
        for recommendation in recommendations
    ]


def _weight_rows(
    learned: Mapping[str, float], priorities: Mapping[str, float]
) -> list[dict[str, object]]:
    aspects = sorted(set(learned) | set(priorities))
    return [
        {
            "Aspect": aspect,
            "Learned weight": round(learned.get(aspect, 0.0), 4),
            "Temporary priority": round(priorities.get(aspect, 0.0), 4),
        }
        for aspect in aspects
    ]


def _metric_rows(bundle: DemoBundle, model_id: str) -> list[dict[str, object]]:
    rows = [metric for metric in bundle.metrics if metric.model_id == model_id]
    if not rows:
        rows = list(bundle.metrics)
    return [
        {
            "Metric": metric.metric,
            "Value": round(metric.value, 4),
            "Split": metric.split_role,
            "Run ID": metric.run_id,
            "Model hash": metric.model_hash,
        }
        for metric in rows
    ]


def _render_evidence(bundle: DemoBundle, recommendations: Sequence[DemoRecommendation]) -> None:
    st.subheader("Source evidence")
    for recommendation in recommendations:
        rows = evidence_for_item(bundle, recommendation.item_id)
        with st.expander(
            f"{recommendation.rank}. {recommendation.title} ({recommendation.item_id})"
        ):
            st.write(f"Support state: {recommendation.evidence_state}")
            if not rows:
                st.info("Evidence is unavailable for this item.")
                continue
            for row in rows:
                st.markdown(f"{row.aspect} · {row.sentiment}")
                st.write(row.text)
                st.caption(
                    f"Review ID: {row.review_id} · Source timestamp: {row.timestamp} · "
                    f"Support: {row.support:.2f}"
                )


def _render_lineage(bundle: DemoBundle, model_id: str) -> None:
    with st.expander("Lineage and prepared metric"):
        st.table(
            [
                {"Field": "Snapshot ID", "Value": bundle.snapshot_id},
                {"Field": "Dataset hash", "Value": bundle.dataset_hash},
                {"Field": "Cutoff timestamp", "Value": bundle.cutoff_timestamp},
                {"Field": "Configuration hash", "Value": bundle.configuration_hash},
                {
                    "Field": "Model hash",
                    "Value": bundle.model_hashes.get(model_id, "unavailable"),
                },
            ]
        )
        st.table(_metric_rows(bundle, model_id))


def main() -> None:
    """Render the five-minute TrustRec presenter flow."""

    st.set_page_config(page_title="TrustRec demo", layout="wide")
    st.title("TrustRec evidence-aware recommendations")
    st.caption("Prepared snapshot demo with source evidence and temporary priorities")

    try:
        load_result = _load_bundle()
    except DemoValidationError as error:
        st.error(f"The demo bundle is unavailable: {error}")
        st.stop()

    bundle = load_result.bundle
    if load_result.used_fallback:
        st.warning(
            "Using the synthetic fallback bundle. "
            f"The configured bundle could not load: {load_result.load_error}"
        )

    user_options = sorted(
        bundle.users,
        key=lambda user_id: (bundle.users[user_id].history_count, user_id),
    )
    with st.sidebar:
        st.header("Demo controls")
        st.selectbox("Snapshot", [bundle.snapshot_id], disabled=True)
        selected_user_id = st.selectbox(
            "User",
            user_options,
            format_func=lambda user_id: (
                f"{user_id} · {bundle.users[user_id].history_count} history events"
            ),
        )
        user = bundle.users[selected_user_id]
        model_options = [
            model_id
            for model_id in bundle.rankings
            if selected_user_id in bundle.rankings[model_id]
        ]
        selected_model_id = st.selectbox(
            "Model",
            model_options,
            format_func=_model_label,
        )
        candidate_count = len(bundle.rankings[selected_model_id][selected_user_id])
        k = st.slider(
            "K",
            min_value=1,
            max_value=min(10, candidate_count),
            value=min(3, candidate_count),
        )
        st.subheader("Temporary aspect priorities")
        priorities: dict[str, float] = {}
        for aspect in _aspect_names(bundle, selected_user_id):
            priority = st.slider(
                aspect.replace("_", " ").title(),
                min_value=0.0,
                max_value=2.0,
                value=0.0,
                step=0.1,
                key=f"priority-{aspect}",
            )
            if priority > 0.0:
                priorities[aspect] = priority

    try:
        baseline = rank_recommendations(
            bundle,
            user_id=selected_user_id,
            model_id="b0_most_popular",
            k=k,
        )
        default_selected = rank_recommendations(
            bundle,
            user_id=selected_user_id,
            model_id=selected_model_id,
            k=k,
        )
        selected = rank_recommendations(
            bundle,
            user_id=selected_user_id,
            model_id=selected_model_id,
            k=k,
            priorities=priorities,
        )
    except DemoValidationError as error:
        st.error(f"The selected prepared records are invalid: {error}")
        st.stop()

    model_changes = set(compare_rankings(baseline, default_selected))
    priority_changes = set(compare_rankings(default_selected, selected))
    st.info(
        f"Selected {selected_user_id} with {user.history_count} history events. "
        "The learned weights stay fixed while temporary priorities change the current view."
    )

    st.subheader("Learned preferences and temporary priorities")
    st.table(_weight_rows(user.learned_aspect_weights, priorities))

    first_column, second_column = st.columns(2)
    with first_column:
        st.subheader("Baseline ranking")
        st.table(_ranking_rows(baseline, model_changes))
    with second_column:
        st.subheader(_model_label(selected_model_id))
        st.table(_ranking_rows(selected, model_changes | priority_changes))

    change_column, priority_column = st.columns(2)
    change_column.metric("Baseline to model rank changes", len(model_changes))
    priority_column.metric("Rank changes after priorities", len(priority_changes))
    _render_evidence(bundle, selected)
    _render_lineage(bundle, selected_model_id)


if __name__ == "__main__":
    main()
