# Graph Module

This module builds positive interaction edges and sparse transition matrices. It owns personalized PageRank, dangling-node behavior, and graph diagnostics. Compute graph scores from the relevant snapshot only.

`PersonalizedPageRankRanker` builds a positive user-item graph before the
cutoff. It starts at the requested user, sends dangling mass back to that
user, and reports `no_positive_history` or `no_graph_path` when it cannot
produce a personalized item score. It falls back to positive popularity in
both cases.
