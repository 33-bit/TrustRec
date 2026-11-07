# Task: T7.3: LLM-judge audit of real explanations

Status: done for the ADR-0017 H0 top-10 on `t1` (T7.1 second run). These
are LLM agreement results, not human judgment or ground truth.

## Method

```bash
PYTHONPATH=src python3 scripts/run_explanation_judge.py --snapshot-manifest data/manifests/video_games-full-d6c4efeb74aa.json --explanations-dir artifacts/explanations/t1_h0_adr0017_k10 --output reports/generated/t7-3-explanation-judge-deepseek-flash.json --model deepseek-flash --reasoning-effort none --per-stratum 100 --top-exposure 30 --workers 8
```

- **Judge.** `deepseek-flash` (DeepSeek-V4.1-Flash), with reasoning off,
  temperature 0, and prompt `t7.3-claim-judge-v1`. It differs from the
  labeler (`gpt-6-luna`). The judge never sees the labeler's polarities.
- **Sample.** Distinct supported (item, aspect) claims: 100 positive, 98
  negative (all that exist), and 100 mixed, drawn with seed 7. The 30 claims
  shown most often form a separate stratum.
- **Per passage.** The judge rates whether the passage covers the aspect, its
  polarity, and whether it is about the reviewed product.
- **Per claim.** The judge rates whether the passages support the claim
  (`claim_supported`).
- **Derived support.** A claim counts as supported when every passage on its
  own side covers the aspect with that polarity, or, for a mixed claim, when
  on-aspect passages show both sides. This ignores the opposing passage that
  ADR-0019 cites on purpose; the judge's own `claim_supported` usually marks
  that passage as `partly`.
- **Cost.** 328 requests, 124k input and 30k output tokens, with no failures.
- Result SHA-256: `46adb70c9a42d0b28afb0b100b3e13af03c1d8c494415021ad8e9632ea0b9877`.

## Results

All 984 cited passages match their source review text at the stored offsets.

| Stratum | Claims | Aspect correct (passages) | About the product | Polarity agrees with label | Derived support | Judge `yes` / `partly` / `no` |
| --- | --- | --- | --- | --- | --- | --- |
| Positive | 100 | 85.7% | 98.7% | 87.3% | 70.0% | 14% / 81% / 5% |
| Negative | 98 | 88.4% | 99.0% | 86.4% | 77.6% | 11% / 82% / 7% |
| Mixed | 100 | 90.7% | 99.7% | 87.7% | 77.0% | 73% / 20% / 7% |
| Top 30 by exposure | 30 | 75.6% | 98.9% | 81.1% | 60.0% | 13% / 70% / 17% |

**Exposure.** The 30 most-shown claims cover 97.9% of all displayed
supported claims, and the top 10 cover 94.9%. Cold-start users share one
B5 list, so they see the same claims. Weighted by exposure, the top-30
derived support is 60.3%. That is the best estimate of what users see.

## Findings

- **Traceability and product.** Every cited passage traces to its source
  text, and 99% are about the reviewed product.
- **Main failure: off-aspect passages.** About 12% of sampled passages, and
  24% among the top-30 claims, do not discuss the claimed aspect. Most are
  generic praise or complaints that the labeler tagged as gameplay, such as
  "LOVE IT", "very good", and "Hate the game". Gameplay has the highest
  off-aspect rate: 20% of its sampled passages, versus 4–8% for multiplayer,
  value, graphics, controls, and performance. This also helps explain why 97%
  of claims are about gameplay.
- **Exposure concentration.** Quality per claim (70–78%) is higher than
  quality per view (about 60%). The single most-shown claim, `B07V8YSBFG`
  gameplay, is used 59,567 times. It cites "LOVE IT" and "Hate the game" as
  gameplay evidence and fails.
- **Coverage and correctness together (ADR-0010).** 99.6% of recommendations
  carry a supported claim, but the judge supports only about 60% of the claims
  users see. Coverage alone overstates explanation quality.

## After LLM-verified display passages (ADR-0020, 2026-11-07)

[ADR-0020](../adr/0020-verified-display-passages.md) shows only passages
that a verifier model accepts, and abstains when none passes on the claim
side. The verifier was `deepseek-flash` with reasoning off and prompt
`adr0020-passage-verify-v1`. It checked 2,354 displayed claims in 2,354
requests (1.29M input and 0.69M output tokens) in 10 minutes.

```bash
PYTHONPATH=src python3 scripts/verify_explanation_passages.py --explanations-dir artifacts/explanations/t1_h0_adr0017_k10 --output-dir artifacts/explanations/t1_h0_adr0017_k10_verified --cache artifacts/explanations/verifier_cache_deepseek-flash.jsonl --model deepseek-flash --reasoning-effort none --workers 8
PYTHONPATH=src python3 scripts/run_explanation_judge.py --snapshot-manifest data/manifests/video_games-full-d6c4efeb74aa.json --explanations-dir artifacts/explanations/t1_h0_adr0017_k10_verified --output reports/generated/t7-3-explanation-judge-deepseek-flash-verified.json --model deepseek-flash --reasoning-effort none --per-stratum 100 --top-exposure 30 --workers 8
```

Verification summary SHA-256:
`6f10aaee7a94c0e6808361ac65867fa8d357181e6be8068d9f5ca755620c6bec`.
Audit SHA-256:
`cf3ba9da7b60364a2a81265a3358d6f71ad5c597c17a94b61da790b607688659`.

| | Before verification | After verification |
| --- | --- | --- |
| Displayed claims kept | 2,354 | 2,293 (97.4%) |
| Views (recommendations showing a claim) | 615,572 | 497,688 (80.9%) |
| Recommendations with a supported claim | 99.6% | 80.6% |
| Passage aspect correct (direction strata) | 88.3% | 98.6% |
| Polarity agrees with label | 87.1% | 97.0% |
| Derived support per claim (direction strata) | 74.8% | 94.3% |
| Derived support, top-30 claims | 60.0% | 93.3% |
| Derived support weighted by exposure (top 30) | 60.3% | 99.7% |
| Cited passages traceable to source text | 984 / 984 | 947 / 947 |

### Findings

- **What users see improves.** Weighted by exposure, the judge supports
  almost all displayed claims after verification, up from about 60%.
- **Coverage falls.** Coverage drops from 99.6% to 80.6% of recommendations.
  Most of the drop comes from two claims that cold-start users saw over and
  over. Both are rejected gameplay claims:
  - `B07V8YSBFG` (a Roblox currency card), 59,567 views;
  - `B017V6YVDC` (an online subscription with games), 55,704 views.
- **Gift cards and subscriptions.** The most-dropped claims belong to gift
  cards, currency codes, and subscriptions (`B07V8YSBFG`, `B017V6YVDC`,
  `B004RMK5QG`, `B000N5Z2L4`). Their reviews praise code delivery and
  redemption, not gameplay, controls, or performance. B5 recommends them
  often because they are popular. The verifier rejects their aspect claims,
  which is correct. Whether such items belong in the recommendation catalog
  is a scope question for ADR-0006.
- **Self-agreement.** The same model (`deepseek-flash`) verified the passages
  and judged them. Part of the improvement may be the model agreeing with
  itself. A human spot check, or a judge from another model family, is
  needed before this result is reported as an improvement.
