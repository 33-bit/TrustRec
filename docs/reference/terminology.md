# TrustRec Terminology

| Term | Meaning in this repository |
| --- | --- |
| aspect | A product dimension such as gameplay, story, performance, or value. |
| evidence | A source sentence/clause and offsets that support an aspect claim. |
| support | Weighted volume and diversity of past evidence; not a confidence interval. |
| confidence | A calibrated or model-reported NLP decision measure; never review authenticity. |
| trust | Ability to inspect evidence and disagreement in a recommendation. |
| snapshot | A time-bounded view of interactions, text, graph, profiles, and models. |
| candidate | A known item before cutoff that the user has not previously reviewed. |
| target | A future observed item with rating ≥4 under the locked evaluation protocol. |
| silver label | AI-generated label used for development or triage, not gold truth. |
| gold label | Human-reviewed label with a frozen guideline and adjudication record. |
| abstention | A deliberate refusal to emit a strong label, explanation, or prediction when support is insufficient. |

Never use “trust” as a synonym for reviewer honesty or fake-review detection in reports.

