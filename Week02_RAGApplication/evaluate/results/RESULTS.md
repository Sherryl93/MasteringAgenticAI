# RAGAS Evaluation Results

_Generated 2026-06-11T19:40:57 - 6 questions - judge: meta-llama/Llama-3.3-70B-Instruct via Nebius_

## Score comparison

| Metric | Config A — fixed-size, no rerank | Config B — semantic, no rerank | Config C — semantic + FlashRank rerank | Config A->Config B | Config B->Config C |
|---|---|---|---|---|---|
| Faithfulness | 0.583 | 0.569 | 0.786 | -0.014 | +0.217 |
| Context Precision | 0.442 | 0.475 | 0.304 | +0.033 | -0.171 |
| Context Recall | 0.611 | 1.000 | 0.556 | +0.389 | -0.444 |

**Winner (highest summed mean): Config B — semantic, no rerank**

## Analysis (computed from the numbers above)

- Best Faithfulness: Config C — semantic + FlashRank rerank (0.786).
- Best Context Precision: Config B — semantic, no rerank (0.475).
- Best Context Recall: Config B — semantic, no rerank (1.000).
- Chunking (A->B) reduced Faithfulness by -0.014.
- Chunking (A->B) improved Context Precision by +0.033.
- Chunking (A->B) improved Context Recall by +0.389.
- Reranking (B->C) improved Faithfulness by +0.217.
- Reranking (B->C) reduced Context Precision by -0.171.
- Reranking (B->C) reduced Context Recall by -0.444.
