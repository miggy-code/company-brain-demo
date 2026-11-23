# Retrieval evals: simulator

From `brain eval casa-prado --suite golden --model live --keys-file <keys-file> --repeats 3 --report simulator` at commit `afd0365e3b89` on 2026-11-23.
Answer model and grader `deepseek-flash` (temperature 0.0, reasoning effort none), mode `live`. Index `bge-m3@5617a9f61b02` + `bge-reranker-v2-m3@953dc6f6f85a`.
80 questions, 3 repeats per question and ranking, 240 answers. Synthetic data (Casa Prado is fictional).

| Ranking | Recall@8 | MRR | Citation precision | Correct | Correct not-found | False not-found | Language match | USD/answer | p50 / p95 ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `hybrid_rerank` | 0.99 [0.98, 1.00] n=64 | 0.95 [0.91, 0.99] n=64 | 0.85 [0.79, 0.91] n=64 | 0.96 [0.93, 0.99] n=64 | 1.00 [1.00, 1.00] n=16 | 0.01 [0.00, 0.03] n=64 | 1.00 [1.00, 1.00] n=80 | 0.000242 | 2736 / 16018 |

Means with 95% bootstrap intervals over questions; n is the number of questions scored. The grader is the answer model itself, so grader-based numbers carry a self-preference risk.
