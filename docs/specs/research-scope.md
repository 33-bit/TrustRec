# Research Scope and Questions

Given history before cutoff `τ`, rank known products that the user did not review. Attach evidence that the user can inspect. Evidence is source text that supports a claim. The benchmark uses rating ≥4 as a positive future target. Missing feedback is not a known negative.

[ADR-0018](../adr/0018-explanation-first-research-focus.md) makes evidence-grounded explanation the primary contribution. Ranking accuracy is a guardrail: the explained recommender must stay on par with the strongest baseline, B5 recent popularity.

## Primary research questions (explanations)

| ID | Question | Required comparison |
| --- | --- | --- |
| RQ4 | Do claims match their sources and score contributions? | LLM explanation audit and evidence-removal test against random removal |
| RQ5 | How many recommendations can carry a supported claim, and how often does the system abstain? | Coverage and abstention for the explained top-K, grouped by history, item popularity, and item age |

## Accuracy guardrail

| ID | Question | Required comparison | Status |
| --- | --- | --- | --- |
| G1 | Is the explained recommender on par with B5? | ADR-0017 H0 versus B5 on `t1`, paired bootstrap, ±0.005 NDCG@10 margin | measured in T6.4 |

## Answered accuracy questions

These questions were answered in [T6.4](../tasks/t6_4-baseline-results.md). Keep the results as findings.

| ID | Question | Finding |
| --- | --- | --- |
| RQ1 | Do aspect opinions and graph data add value beyond collaborative filtering? | Graph adds value on top of recency. Aspect evidence adds no robust accuracy. |
| RQ2 | Do adaptive weights help more than fixed weights? | No. The T0 gate equals its no-aspect ablation. |
| RQ3 | Do evidence weights reduce changes from duplicates and polarity noise? | Not tested at full scale. It is lower priority after ADR-0018. |

Video Games is the primary domain after T1.2 profiling. Core work includes top-K ranking at K ∈ {5, 10, 20}, BPR MF, PPR, recent popularity, and aspect sentiment. It also includes evidence aggregation, explanation generation, the faithfulness test, and a Streamlit demo.

Explanations use LLM evidence only. Comparing explanation quality across evidence sources, such as the T2.3 dictionary labels, is an extension.

Node2vec, NCF, stronger NLP, diversification, onboarding, and item cold-start are extensions. Cold-start means that an item has no past interactions. Extensions cannot replace core baselines or the locked temporal evaluation.
