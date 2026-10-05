# Demo Application

Run the demo from the repository root:

```bash
streamlit run app/streamlit_app.py
```

The app reads one prepared JSON bundle. Set `TRUSTREC_DEMO_BUNDLE` to use an external bundle:

```bash
TRUSTREC_DEMO_BUNDLE=/path/to/demo_bundle.json streamlit run app/streamlit_app.py
```

If the configured bundle is missing or invalid, the app loads the checked-in
synthetic bundle at `app/demo_bundle.json`. The app shows a warning when it
uses this fallback. The fallback supports the full presenter flow, but its
metric is synthetic and does not support a production result claim.

The sidebar accepts a snapshot, user, model, K, and temporary aspect
priorities. The page shows the baseline and selected model rankings, score
parts, rank changes, learned weights, source review IDs, evidence text,
support states, source timestamps, lineage fields, and a prepared metric.
Temporary priorities affect only the current view. The learned user weights
remain unchanged.

The app does not download raw data or fit a recommender, NLP model, or graph
during a request. Prepare production rankings, evidence, and metrics before
you start the app. Use `streamlit.testing.v1.AppTest` for a local smoke run
when the environment cannot bind a server port.

Use this five-minute sequence:

1. Select the low-history user.
2. Show the baseline ranking.
3. Select TrustRec.
4. Point to the rank change.
5. Open source evidence.
6. Increase one aspect priority.
7. Show the new rank change and conflicting or weak support.
8. Open the prepared metric and lineage panel.
