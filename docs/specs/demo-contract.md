# Demo Contract

The Streamlit demo lets a presenter select a user, snapshot, K, and model, then adjust aspect priorities. It displays rank, item ID, component scores, prominent aspects, evidence text, evidence timestamps, and support limits. It separates learned user weights from priorities entered during the current interaction.

The five-minute path is: sparse-history user → baseline → TrustRec → rank change → source evidence → aspect-priority change → conflicting or insufficient evidence → measured baseline/ablation result. The demo must use internal IDs and prepared artifacts; it must not download or fit the full corpus during a request.

