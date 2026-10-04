# Research Scope and Questions

Given history before cutoff `τ`, rank known products that the user did not review. Attach evidence that the user can inspect. Evidence is source text that supports a claim. The benchmark uses rating ≥4 as a positive future target. Missing feedback is not a known negative.

## Research questions

| ID | Question | Required comparison |
| --- | --- | --- |
| RQ1 | Do aspect opinions and graph data add value beyond collaborative filtering? | BPR MF, MF+aspect, MF+graph, full model |
| RQ2 | Do adaptive weights help more than fixed weights? | H0 versus T0 and A4, grouped by history |
| RQ3 | Do evidence weights reduce changes from duplicates and polarity noise? | T0 versus A3 and controlled stress tests |
| RQ4 | Do claims match their sources and score contributions? | LLM explanation audit and evidence-removal test |

Video Games is the primary domain after T1.2 profiling. Core work includes top-K ranking at K ∈ {5, 10, 20}, BPR MF, PPR, and aspect sentiment. It also includes evidence aggregation, the adaptive gate, and a Streamlit demo.

Node2vec, NCF, stronger NLP, diversification, onboarding, and item cold-start are extensions. Cold-start means that an item has no past interactions. Extensions cannot replace core baselines or the locked temporal evaluation.
