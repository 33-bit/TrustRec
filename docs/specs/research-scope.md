# Research Scope and Questions

## Problem

Given a user history before cutoff `τ`, rank known products the user has not reviewed and attach inspectable aspect evidence. The benchmark uses rating ≥4 as a positive future target; missing feedback is not a confirmed negative.

## Research questions

| ID | Question | Required comparison |
| --- | --- | --- |
| RQ1 | Do aspect opinions and graph structure add value beyond collaborative filtering? | BPR MF, MF+aspect, MF+graph, full model |
| RQ2 | Does adaptive weighting help compared with fixed weights? | H0 versus T0 and A4, sliced by history |
| RQ3 | Does evidence weighting improve stability under duplicates and polarity noise? | T0 versus A3 plus controlled stress tests |
| RQ4 | Are explanations source-correct and faithful to score contributions? | Manual claim audit plus evidence-removal recomputation |

## In scope

Video Games as the primary domain after T1.2 profiling; top-K ranking at K ∈ {5, 10, 20}; BPR MF, PPR, aspect sentiment, evidence aggregation, adaptive gate, and Streamlit inspection demo.

## Conditional extensions

Node2vec, NCF, stronger NLP, diversification, onboarding, and item cold-start are extension work. They cannot displace the core baseline ladder or the locked temporal evaluation.

