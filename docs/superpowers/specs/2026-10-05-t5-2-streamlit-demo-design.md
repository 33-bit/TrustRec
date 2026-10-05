# T5.2 Streamlit Demo Design

## Goal

Build a deterministic five-minute Streamlit demo for the TrustRec serving contract. The demo reads prepared records, compares a baseline with TrustRec, shows source evidence, applies temporary aspect priorities, and reports a measured result.

## Scope

The demo accepts a user, snapshot, model, K, and temporary aspect priorities. It shows rank, item ID, item title when available, total score, score parts, key aspects, support state, source evidence text, and source timestamps. It does not show reviewer identities.

The demo does not download raw data, fit a recommender, fit an NLP model, or build a graph during a request. It reads one JSON bundle that contains prepared user, ranking, evidence, lineage, and metric records. The bundle path comes from `TRUSTREC_DEMO_BUNDLE`. A small synthetic bundle in `app/demo_bundle.json` keeps the five-minute flow usable when external artifacts are absent.

## Architecture

`app/demo.py` owns bundle loading, lineage validation, ranking views, temporary priority application, and display records. It has no Streamlit dependency. The loader rejects missing identity fields, invalid score values, mismatched candidate IDs, invalid timestamps, and incomplete lineage. The ranking view keeps the stored learned user weights and the current priority overrides in separate mappings.

`app/streamlit_app.py` owns controls and rendering. It loads the bundle once per process, displays the selected user history, renders baseline and TrustRec tables, highlights rank changes, and opens evidence details for one item. It renders support states for supported, insufficient, conflicting, and unavailable evidence. It renders lineage fields and prepared metric rows so a presenter can name the snapshot, cutoff, model hash, and metric source.

`app/demo_bundle.json` is a checked-in synthetic fallback. It contains one low-history user, three items, baseline and TrustRec score parts, learned aspect weights, evidence with positive and negative passages, one weak-support item, and one measured metric. It is a demonstration fixture, not a production model artifact.

## Data flow

The loader reads the configured bundle and validates its schema version, snapshot ID, dataset hash, cutoff timestamp, configuration hash, model hashes, users, items, rankings, evidence, and metrics. The service selects the requested user and model from the validated records. It returns at most K candidates in stable score order. It removes no prepared candidate at request time except the requested K limit.

The default TrustRec ranking uses the stored score parts. A temporary priority changes only the aspect contribution. For an item, the service computes an adjusted aspect score from the item aspect scores and the sum of the learned user weights plus the temporary priority map. It keeps the learned map unchanged and records both maps in the response. The non-aspect contributions remain fixed. The service reorders items by adjusted total score and item ID for ties.

The service maps evidence rows to each item and aspect. It keeps review IDs for traceability but the view omits author fields. A row with both positive and negative evidence has the `conflicting` state. A row below the support threshold has the `insufficient_support` state and no strong claim text. Missing evidence has the `unavailable` state.

## User flow

The sidebar selects the snapshot, low-history user, model, K, and aspect priorities. The main panel shows the user history count and learned weights. The ranking panel shows the baseline and selected model with rank, item title, item ID, total score, and score parts. A change indicator names items that moved after a priority change.

Each recommendation has an evidence expander. The expander shows the aspect, support state, sentiment direction, evidence text, source review ID, and source timestamp. It does not show reviewer identities. A metric panel shows the prepared metric name, value, split role, run ID, and model hash. A lineage panel shows the snapshot ID, dataset hash, cutoff, and configuration hash.

The fallback fixture supports this sequence: select the low-history user, view the baseline, select TrustRec, observe a rank change, open source evidence, increase one aspect priority, observe a second rank change, and open weak or conflicting evidence.

## Error handling

If the configured bundle path does not exist or fails validation, the app shows the validation message and loads the synthetic fallback. The app labels the fallback as synthetic. If both bundles fail, the app stops with a user-readable message and does not attempt a download or fit.

If a requested user, model, or item does not exist, the service raises a typed validation error. The view shows the error and keeps the controls available. If K is outside the prepared range, the view clamps it to the available candidate count.

## Testing

Tests in `tests/unit/test_demo.py` cover valid bundle loading, lineage rejection, stable ranking, baseline comparison, temporary priority isolation, rank changes, conflicting evidence, weak support, and fallback selection. Tests use the checked-in synthetic bundle and do not read production data or call Streamlit.

The final validation runs the focused demo tests, the full pytest suite, Ruff formatting and lint checks, and a Streamlit smoke import. The task record names these commands and the fallback fixture hash.

## Out of scope

The task does not add an online model service, a new ranking algorithm, a report pipeline, a full snapshot scan, or a production artifact registry. T5.1 owns final evaluation metrics. T5.3 owns the report and reproducibility bundle.
