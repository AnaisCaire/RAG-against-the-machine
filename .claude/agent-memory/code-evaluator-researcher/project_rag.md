---
name: RAG Project State
description: Key facts about the RAG-against-the-machine project: current recall scores, critical bugs, tech stack
type: project
---

RAG-against-the-machine: hybrid BM25 + FAISS semantic retrieval over the vLLM-0.10.1 codebase.

Tech stack: bm25s, sentence-transformers (all-MiniLM-L6-v2), faiss-cpu, Qwen/Qwen3-0.6B, Python Fire CLI, pydantic v2, uv package manager.

## Current recall (2026-04-28 analysis)
- Docs Recall@5: 0.87 on private dataset (threshold 0.80) — PASS
- Code Recall@5: PASS (>= 50%)
- Target: ~95% docs Recall@5
- Local simulation on public dataset: 83/100 = 83% (consistent with 87% private)
- CANDIDATE_MULT=20 and docs_chunk_size=1200 are the current live params

## Index stats (current)
- Docs index: 1199 chunks, mean 951 chars, median 1068 chars, max 1200 chars
- File contamination: benchmarks/, examples/, tests/, requirements/, .buildkite/ dirs all included in docs index — 33 noisy files
- Basename collisions: 124 unique basenames shared across multiple paths (README.md = 32 different files, 138 chunks)
- Corpus text format: "{basename}\n{chunk_text}" — LOSES path info, destroys BM25 disambiguation

## Diagnosed failure modes for docs (17 failures on public dataset)
1. RANKING failure (not coverage): 100% of GT spans are covered by at least one chunk in index
2. Small GT spans (74-162 chars) — semantic embedding too dilute for recall; BM25 can find them (rank 1) but RRF fails when semantic rank=None
3. Markdown table GT spans — no semantic signal (pipe-heavy, model name lists); neither BM25 nor semantic can match "tensor parallelism" to a row of model names
4. Noise files (.buildkite, benchmarks) appearing in top-5 for 10% of queries, displacing true results
5. DSE-Qwen2-MRL: BM25 rank=1, semantic rank=6 — RRF fusion not boosting BM25 winner enough when semantic fails
6. Very large GT spans (1849-1997 chars) split across two 1200-char chunks — best chunk only covers 38-39% of GT span. Overlap threshold is 5%, so technically passes; but getting the RIGHT chunk to rank is harder when the relevant content is split

## Key levers (ranked by expected impact)
1. Embedding model upgrade: all-MiniLM-L6-v2 (384d) → all-mpnet-base-v2 (768d): +6-8% semantic recall; benchmarks at 19.9s for 1199 chunks (well within 300s index budget)
2. Relative file path in corpus: "docs/features/lora.md\n{text}" instead of "lora.md\n{text}" — fixes 124 basename collision families for BM25
3. Noise file filtering: exclude benchmarks/, examples/, tests/, .buildkite/, requirements/ from docs index — removes 33 noisy files
4. BM25 boost for small GT spans: lower RRF k from 60 → 30 to amplify rank-1 BM25 hits
5. Chunk overlap: helps with large GT spans (1849-1997 chars split at 1200 boundary)
6. Header injection: prepend nearest Markdown H1/H2/H3 to every chunk — helps table-only chunks get matched

**Why:** Failing moulinette 95% target for docs.
**How to apply:** Every change must stay within 300s index time and 90s search throughput.
