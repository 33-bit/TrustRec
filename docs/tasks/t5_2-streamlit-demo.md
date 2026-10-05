# T5.2: Five-minute Streamlit demo

Status: done.

The demo reads a prepared JSON bundle. It lets a presenter select a user,
snapshot, model, K, and temporary aspect priorities. It compares a most
popular baseline with TrustRec, shows score parts, opens source evidence, and
shows supported, conflicting, insufficient, and unavailable evidence states.

## Related records

- Backlog: [`backlog.md`](backlog.md)
- Task matrix: [`task-matrix.md`](task-matrix.md)
- Design spec: [`../superpowers/specs/2026-10-05-t5-2-streamlit-demo-design.md`](../superpowers/specs/2026-10-05-t5-2-streamlit-demo-design.md)
- Implementation plan: [`../superpowers/plans/2026-10-06-t5-2-streamlit-demo.md`](../superpowers/plans/2026-10-06-t5-2-streamlit-demo.md)
- Serving contract: [`../specs/serving-contract.md`](../specs/serving-contract.md)
- Evidence contract: [`../specs/evidence-and-explanation.md`](../specs/evidence-and-explanation.md)
- Abstention decision: [`../adr/0010-explanation-abstention.md`](../adr/0010-explanation-abstention.md)
- Lineage decision: [`../adr/0011-experiment-and-claim-lineage.md`](../adr/0011-experiment-and-claim-lineage.md)

## Source and cutoff

The production source snapshot is `video_games-full-d6c4efeb74aa`. Its dataset
hash is
`0b990d349f917c8b864a17a729f73f83314829b78d7307073d6bd0a644404043`.
The training cutoff is `2019-01-13T04:24:39.377000+00:00`. The fallback bundle
uses these values as display lineage. The fallback data is synthetic and does
not contain the production snapshot rows.

## Implementation

`app/demo.py` validates the bundle schema, lineage, IDs, timestamps, scores,
candidate IDs, evidence rows, and metric rows. It returns typed records for
the view. It does not import Streamlit and does not fit a model.

`app/streamlit_app.py` renders the sidebar controls, baseline and selected
model rankings, rank changes, learned weights, temporary priorities, source
evidence, lineage, and prepared metrics. The view shows review IDs and exact
source text but does not show reviewer identities.

`app/demo_bundle.json` is the checked-in synthetic fallback. Its SHA-256 is
`841ba49dd421b1f350bf370b15dfceff5550c1edd5bbc7d73f80c5daefea3285`.
The bundle contains one low-history user and three items. A graphics priority
changes the TrustRec order. It also contains positive evidence, conflicting
evidence, and weak evidence for the presenter flow.

Set `TRUSTREC_DEMO_BUNDLE` to use an external prepared bundle. If that file is
missing or invalid, the app loads the fallback and shows the load error.

## Validation

The focused checks passed:

```text
ruff format --check app tests/unit/test_demo.py
4 files already formatted

ruff check app tests/unit/test_demo.py
All checks passed!

python3 -m pytest tests/unit/test_demo.py -q
10 passed in 0.16s

python3 -c "import app.streamlit_app"
exit 0
```

The in-process Streamlit smoke run passed:

```text
python3 - <<'PY'
from streamlit.testing.v1 import AppTest

app = AppTest.from_file("app/streamlit_app.py").run(timeout=10)
assert not app.exception, app.exception
assert app.title[0].value == "TrustRec evidence-aware recommendations"
assert any("Baseline ranking" in item.value for item in app.subheader)
print("AppTest passed")
PY
AppTest passed
```

The full repository gate passed:

```text
make check
171 files already formatted
All checks passed!
166 passed in 2.24s

make validate
Validated 31 repository foundation files
```

The direct Streamlit server smoke command could not bind a local port in the
managed sandbox. AppTest provided the page smoke validation without a socket.

## Limits

The fallback metric is synthetic. Use an external prepared bundle for a
production presentation. T5.1 owns final evaluation values. T5.3 owns the
report and reproducibility package.
