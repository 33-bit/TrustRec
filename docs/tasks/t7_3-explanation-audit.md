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
