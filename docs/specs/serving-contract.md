# Serving and Demo Contract

The serving path reads prepared snapshot artifacts. It accepts `user_id`, `snapshot_id`, `model_id`, K, and optional aspect priorities. It returns ranked items, component scores, explanation status, claims, evidence references, and a reason for any abstention.

The UI must distinguish learned user weights from temporary priorities, show score components separately, display source text and timestamp, and avoid reviewer identity. It must not fit NLP or recommender models during a request. A demo failure must have a prepared video or static artifact fallback.

