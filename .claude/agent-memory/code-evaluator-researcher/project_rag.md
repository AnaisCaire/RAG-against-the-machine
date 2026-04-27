---
name: RAG Project State
description: Key facts about the RAG-against-the-machine project: current recall scores, critical bugs, tech stack
type: project
---

RAG-against-the-machine: hybrid BM25 + FAISS semantic retrieval over the vLLM-0.10.1 codebase.

Current recall (as of 2026-04-25): docs=93% (4 stars), code=58% (3 stars).
Targets: docs>=80% mandatory, >=95% for 5 stars. Code>=50% mandatory, >=65% for 5 stars.

Tech stack: bm25s, sentence-transformers (all-MiniLM-L6-v2), faiss-cpu, Qwen/Qwen3-0.6B, Python Fire CLI, pydantic v2, uv package manager.

Critical known bugs: faiss crashes on k=0 (AssertionError), sentence-transformers and faiss-cpu not in pyproject.toml (uv sync will fail on fresh machine), batch.py has no FileNotFoundError handling for bad dataset paths.

**Why:** These bugs cause mandatory Yes/No evaluation failures on edge case tests.
**How to apply:** Always check these three first when reviewing this project.
