# T2.1 — Laya Integration

Laya is integrated as an optional local labeling backend. The adapter asks two typed decisions per aspect: whether the text discusses the aspect and which polarity applies. It also asks whether the unit is out of scope and whether a human should adjudicate it.

Install with `make install-laya`. The first real prediction downloads the selected Hugging Face checkpoint; later runs reuse the local cache. No hosted LLM API or API key is needed. `laya[serve]` is not required for this repository because the Python runner calls the local `Router` directly.

Output is written to `docs/tasks/t2_1_laya_pilot_labels.jsonl` and is always marked `ai_preliminary`. The default presence threshold is 0.85; tune it only on development labels and record the value. Because Laya returns decisions rather than text spans, the runner uses the already selected sentence/clause as the evidence span and preserves its exact offsets. Human review remains required for gold labels and for validating the model's confidence threshold.
