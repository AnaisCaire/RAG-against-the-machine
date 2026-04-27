---
name: RAG Project State
description: Key facts about the RAG-against-the-machine project: current recall scores, critical bugs, tech stack
type: project
---

RAG-against-the-machine: hybrid BM25 + FAISS semantic retrieval over the vLLM-0.10.1 codebase.

Tech stack: bm25s, sentence-transformers (all-MiniLM-L6-v2), faiss-cpu, Qwen/Qwen3-0.6B, Python Fire CLI, pydantic v2, uv package manager.

## Current recall (as of 2026-04-27 latest moulinette run)
- Docs Recall@5: 0.77 (threshold 0.80) — FAIL by 3 questions
- Code Recall@5: 0.43 (threshold 0.50) — FAIL by 7 questions
- Moulinette evaluates at Recall@5 (not @10); uses IoU >= 5% to count a hit

## Index stats
- Docs index: 736 chunks, median 1790 chars, max_chunk_size=2000
- Code index: 46290 chunks across 849 files, max_chunk_size=2000

## Diagnosed root causes (ranked by impact)
1. CodeChunker Case B drops function/class HEADER when recursing into body — creates hard coverage gaps. 9 code questions have zero indexable chunk (irrecoverable miss). Fix: emit header chunk from node_start to first-body-child-start before recursing.
2. /benchmarks/ filter in __main__.py lines 34-38 also catches vllm/benchmarks/throughput.py — 1 code question permanently excluded.
3. CANDIDATE_MULT=4 in indexer.search() means only top-40 BM25 + top-40 FAISS enter RRF out of 46290 chunks. 6 code + 7 docs questions land at rank 6-10 (outside the @5 evaluation window). Fix: raise to 10.
4. setup.py not in docs index (ingest_docs only takes .md/.txt) — 1 docs question miss.
5. 2 docs expected sources are 21-22 chars: with chunk_size=2000 max achievable IoU is ~1%, below the 5% threshold. Structurally irrecoverable unless chunk_size <= 440.
6. 42 code questions have a valid chunk in index but it doesn't rank in top-5 — due to 46k boilerplate-heavy code chunks overwhelming BM25 IDF.

## Moulinette vs evaluate.py discrepancy
- Moulinette: IoU (intersection/union) >= 5%
- evaluate.py: overlap/exp_len >= 5% (one-sided ratio, not IoU)
- For this dataset: zero disagreements found empirically — both agree on every question.

**Why:** Failing moulinette thresholds.
**How to apply:** Use moulinette numbers as ground truth; local evaluate.py scores may differ from production.
