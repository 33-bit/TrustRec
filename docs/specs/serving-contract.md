# Serving and Demo Contract

The serving path reads prepared snapshot artifacts. A snapshot is a versioned view for time-based analysis. The service accepts `user_id`, `snapshot_id`, `model_id`, K, and optional aspect priorities. It returns ranked items, score parts, explanation status, claims, evidence references, and refusal reasons.

Keep learned user weights separate from temporary priorities. Show score parts separately. Show source text and timestamps. Do not show reviewer identities. Do not fit NLP or recommender models during a request. Prepare a video or static artifact for demo failures.
