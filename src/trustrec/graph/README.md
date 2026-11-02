# Graph Module

This module builds positive interaction edges and sparse transition matrices. It owns personalized PageRank, dangling-node behavior, and graph diagnostics. Compute graph scores from the relevant snapshot only.

`PersonalizedPageRankRanker` builds a positive user-item graph before the
cutoff. It starts at the requested user, sends dangling mass back to that
user, and reports `no_positive_history` or `no_graph_path` when it cannot
produce a personalized item score. It falls back to positive popularity in
both cases.

The ranker iterates item mass only. Two walk steps form the item-space
operator `M = B^T D_u^-1 B D_i^-1`, and users with one positive edge add a
diagonal term. This gives the same scores and stopping step as the node-by-node
power iteration up to rounding. `prepare(user_ids)` scores up to `batch_size`
users per sparse product, and `clear_cache()` frees those scores.
