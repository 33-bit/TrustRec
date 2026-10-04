# Demo Contract

The Streamlit demo lets a presenter select a user, snapshot, K, and model. The presenter can change aspect priorities. An aspect is a product property. The app shows rank, item ID, score parts, key aspects, review evidence, timestamps, and support limits.

Keep learned user weights separate from priorities entered during the current interaction. Use internal IDs and prepared artifacts. Do not download or fit the full corpus during a request.

Use this five-minute sequence:

1. Select a user with little history.
2. Show the baseline ranking.
3. Switch to TrustRec.
4. Show a rank change.
5. Open the source evidence.
6. Change an aspect priority.
7. Show conflicting or weak evidence.
8. Show a measured baseline or ablation result.
