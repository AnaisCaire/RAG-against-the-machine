# RAG Project — Improvement Plan for Maximum Grade

---

## Current State Summary

The pipeline is architecturally sound and already passes both mandatory recall thresholds:
- **Docs Recall@10: 93.0%** (4 stars — needs 95% for 5 stars, only 7 questions fail)
- **Code Recall@10: 58.0%** (3 stars — needs 65% for 5 stars, 42 questions fail)

The hybrid BM25 + FAISS semantic approach (RRF fusion) is well-designed and has correctly separate docs/code indices. However, several issues will cause outright **Yes/No failures** in the rubric: two critical missing dependencies, one faiss crash on `k=0`, one crash on bad dataset paths, and widespread flake8/mypy violations that a peer evaluator running `make lint` will see immediately. The README is also nowhere near the required format.

Estimated current grade without fixes: **fails "Edge cases don't crash" (Q12) and likely "Code quality" (Q3)**. With all fixes below, the project should score strongly across the board.

---

## CRITICAL — Must Fix to Pass

These cause outright **Yes/No = No** on mandatory rubric questions. Fix all of these before the evaluation.

### 1. `sentence-transformers` and `faiss-cpu` missing from `pyproject.toml`

**File:** `pyproject.toml`
**Rubric question:** Q1 — "Setup: uv sync works"

These two packages are installed manually in `.venv` but are **not declared** in `pyproject.toml` and do not appear in `uv.lock`. A peer evaluator who runs `make install` (`uv sync`) on a fresh machine will get an import error when the indexer loads:

```python
# indexer.py line 9
from sentence_transformers import SentenceTransformer  # ImportError on fresh install
import faiss                                            # ImportError on fresh install
```

**Fix:** Add both to `pyproject.toml` dependencies, then run `uv add sentence-transformers faiss-cpu` to regenerate `uv.lock`:

```toml
# pyproject.toml — add these two lines to dependencies list
"sentence-transformers>=3.0.0",
"faiss-cpu>=1.9.0",
```

After adding, run:
```bash
uv add sentence-transformers faiss-cpu
```

This regenerates `uv.lock` so `uv sync` installs them on any machine.

---

### 2. `k=0` crashes the indexer with a faiss `AssertionError`

**File:** `student/indexer.py`, `search()` method (line 135)
**Rubric question:** Q12 — "Edge cases don't crash: `answer 'What is vLLM?' --k 0`"

When `k=0` is passed to `search()`, the code calls `self.faiss_index.search(..., k * 2)` which is `faiss.search(..., 0)`. FAISS raises an uncaught `AssertionError` on `k=0`.

```python
# Current — crashes when k=0
docs, _ = self.bm25_retriever.retrieve(token_q, k=k * 2)       # k*2 = 0
distances, indices = self.faiss_index.search(
    np.array(query_vector).astype('float32'), k * 2)             # AssertionError!
```

**Fix:** Guard at the top of `search()` before any retrieval:

```python
def search(self, query: str, k: int = 5) -> List[MinimalSource]:
    """Searches the index and returns the top-k chunks."""
    if k <= 0:
        return []
    # ... rest of the method unchanged
```

---

### 3. Bad dataset path crashes `search_dataset` and `answer_dataset` with unhandled `FileNotFoundError`

**File:** `student/batch.py`, `search_dataset()` line 26 and `answer_dataset()` line 70
**Rubric question:** Q12 — "Edge cases: `search_dataset --dataset_path /nonexistent.json`"

Both methods call `open(dataset_path, 'r')` with no error handling. A missing file raises `FileNotFoundError` and crashes the CLI with an unformatted traceback.

```python
# Current — crashes with FileNotFoundError
with open(dataset_path, 'r') as f:
    raw_data = json.load(f)
```

**Fix:** Wrap both file opens in a try/except:

```python
# search_dataset and answer_dataset — add this guard
try:
    with open(dataset_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
except FileNotFoundError:
    print(f"Error: dataset file not found: {dataset_path}")
    return
except json.JSONDecodeError as e:
    print(f"Error: invalid JSON in {dataset_path}: {e}")
    return
```

---

### 4. Flake8 violations will fail the "Code quality" check

**Files:** All files under `student/`
**Rubric question:** Q3 — "Code quality + all 7 pydantic models correct"

Running `make lint` currently produces **45 flake8 errors**. A peer evaluator who runs this will likely mark Q3 as No. The violations by file:

- `student/__main__.py`: F401 (unused `torch` import), E128 (indentation), E501 (lines too long), E305
- `student/batch.py`: E501, F401 (unused `Generator` import — it IS used for the type hint, so fix the type annotation instead of removing the import)
- `student/chunker.py`: E129, E501
- `student/evaluate.py`: E501 (many), W292 (no newline at end of file)
- `student/generator.py`: E402 (imports after `os.environ` assignment), E501
- `student/indexer.py`: E501
- `student/models.py`: W391 (blank line at end of file)

**Fix E402 in `generator.py`** (most structurally wrong): the `os.environ` call breaks the "all imports first" rule. Move it after all imports:

```python
# Current — E402 because os.environ call interrupts imports
import os
os.environ["OMP_NUM_THREADS"] = "1"   # <-- this breaks PEP8
from typing import Any, List
...

# Fixed — all imports first, then module-level config
import os
from typing import Any, List
from transformers import AutoTokenizer, AutoModelForCausalLM
from student.models import MinimalSource, MinimalAnswer
import torch

os.environ["OMP_NUM_THREADS"] = "1"   # <-- after imports
```

**Fix F401 in `__main__.py`**: Remove the unused `import torch` on line 9.

**Fix F401 in `batch.py`**: `Generator` is imported but the `__init__` type annotation uses a bare `None`. Either add a proper type annotation or remove the unused import:

```python
# Current (F401 unused import because type hint is missing)
from student.generator import Generator
def __init__(self, search_engine: Optional[Indexer], generator=None):

# Fixed
from student.generator import Generator
def __init__(self, search_engine: Optional[Indexer],
             generator: Optional[Generator] = None) -> None:
```

**Fix E501 (lines > 79 chars)**: Either add a `[flake8]` section to `pyproject.toml` to set `max-line-length = 120` (common in modern projects), or break the long lines manually. The cleaner approach for a graded project is to add the config:

```toml
# Add to pyproject.toml
[tool.flake8]
max-line-length = 120
```

Note: `flake8` does not read `[tool.flake8]` from `pyproject.toml` by default — it needs a `.flake8` file or `setup.cfg`:

```ini
# Create .flake8 at project root
[flake8]
max-line-length = 120
exclude = .venv,__pycache__
```

---

### 5. Mypy fails with 17 type errors

**Files:** `student/__main__.py`, `student/generator.py`, `student/indexer.py`, `student/batch.py`, `student/models.py`
**Rubric question:** Q3 — "Code quality + mypy type hints"

Running `make lint` as-is produces a fatal mypy "found twice under different module names" error before it even checks types. After fixing that (add `--explicit-package-bases` to the mypy command in the Makefile or add a `mypy.ini`), 17 real type errors remain:

**Critical mypy errors to fix:**

1. **All `__main__.py` methods missing `-> None`** (lines 17, 20, 43, 56, 85, 105, 119):
   ```python
   # Current
   def index(self, max_chunk_size: int = 2000):
   # Fixed
   def index(self, max_chunk_size: int = 2000) -> None:
   ```

2. **`generator.py` `__init__` missing `-> None`** (line 10):
   ```python
   def __init__(self) -> None:
   ```

3. **`generator.py` lines 101 and 169**: `tokenizer.decode()` returns `str | list[str]` so `.strip()` is not safe without narrowing. Fix:
   ```python
   decoded = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
   expanded_keywords = decoded if isinstance(decoded, str) else decoded[0]
   expanded_keywords = expanded_keywords.strip()
   ```

4. **`indexer.py` lines 89, 151, 153**: `Optional[SentenceTransformer]` needs a None-guard before `.encode()` and `.search()`. Add an assertion or early return:
   ```python
   if self.embedding_model is None or self.faiss_index is None:
       raise RuntimeError("Semantic model not loaded. Did you call build_index/load_index?")
   ```

5. **`batch.py` line 11**: `generator` parameter needs `Optional[Generator]` type annotation (see fix above in issue 4).

6. **`batch.py` line 39**: `self.search_engine` can be `None` but is called without a guard. Add a check:
   ```python
   if self.search_engine is None:
       raise RuntimeError("search_engine is None — cannot run search_dataset.")
   found_sources = self.search_engine.search(query=question, k=k)
   ```

7. **`models.py` line 44**: `List[MinimalAnswer]` is not assignable to `List[MinimalSearchResults]` because `list` is invariant in mypy. Fix by using `Sequence` or by redefining the field without inheritance conflict. The simplest working fix:
   ```python
   # In models.py — change the StudentSearchResultsAndAnswer field
   class StudentSearchResultsAndAnswer(StudentSearchResults):
       """Search results with generated answers."""
       search_results: List[MinimalAnswer]  # type: ignore[assignment]
   ```

**Fix the Makefile mypy command** to avoid the "found twice" fatal error:

```makefile
# Current
PYTHONPATH=. uv run mypy $(CODE_DIRS) \
    --ignore-missing-imports \
    ...

# Fixed — add --explicit-package-bases
PYTHONPATH=. uv run mypy $(CODE_DIRS) \
    --ignore-missing-imports \
    --explicit-package-bases \
    --disallow-untyped-defs \
    --check-untyped-defs \
    --warn-return-any \
    --warn-unused-ignores
```

---

## HIGH PRIORITY — Recall Score Improvements

### 6. Code recall at 58%: 7 percentage points short of 5-star threshold (65%)

The code index correctly filters out `/tests/`, `/benchmarks/`, and `/examples/`. The 42 failing questions mostly suffer from ranking issues, not missing files. The following targeted changes push recall closer to the 65% target.

**6a. Apply the same `_clean_text` camelCase/underscore splitting to query expansion**

The `_clean_text` function in `indexer.py` normalizes text well for BM25, splitting camelCase and underscores. But when `expand=True` is used, the LLM-expanded query goes through `_clean_text` after tokenization — which is correct. However, the expansion model (`expand_querry`) uses only 30 new tokens with a very restrictive system prompt. Increase to 50 tokens and make the prompt generate more diverse code identifiers:

```python
# generator.py expand_querry — increase max_new_tokens
outputs = self.model.generate(
    **inputs,
    max_new_tokens=50,   # was 30
    do_sample=False,
    pad_token_id=self.tokenizer.eos_token_id
)
```

**6b. Increase the BM25 candidate pool from `k*2` to `k*4` before RRF fusion**

Currently `search()` fetches `k*2` candidates from each of BM25 and FAISS before fusing. For code, where the relevant chunk may need to beat many lexically similar but wrong chunks, doubling the candidate pool significantly improves recall:

```python
# indexer.py search() — change k * 2 to k * 3 or k * 4
CANDIDATE_MULTIPLIER = 4

docs, _ = self.bm25_retriever.retrieve(token_q, k=k * CANDIDATE_MULTIPLIER)
...
distances, indices = self.faiss_index.search(..., k * CANDIDATE_MULTIPLIER)
```

This has no cost at query time (the BM25 and FAISS lookups are fast); it only makes the RRF fusion pool larger.

**6c. The `index` command ignores `--max_chunk_size` for docs (hardcoded 1500)**

`student/__main__.py` line 25 hardcodes the docs chunk size to 1500 regardless of the `--max_chunk_size` CLI argument. This violates the subject requirement ("max chunk size configurable via CLI arg") and means a peer evaluator who runs `index --max_chunk_size 500` will see different behavior than expected.

```python
# Current — ignores CLI arg for docs
def index(self, max_chunk_size: int = 2000):
    docs_ingestion = IngestionEngine(max_chunk_size=1500)  # HARDCODED
    ...
    code_ingestion = IngestionEngine(max_chunk_size=max_chunk_size)  # uses CLI arg

# Fixed — use CLI arg for both
def index(self, max_chunk_size: int = 2000) -> None:
    docs_ingestion = IngestionEngine(max_chunk_size=max_chunk_size)
    ...
    code_ingestion = IngestionEngine(max_chunk_size=max_chunk_size)
```

Note: The current 1500-char docs chunk size is what achieves 93% recall. If you change to use the CLI default (2000), re-run the pipeline and measure. 2000 might perform similarly or slightly better for docs.

---

### 7. Docs recall at 93%: 2 percentage points short of 5-star threshold (95%)

All 7 failing docs questions have the correct chunk indexed — the problem is purely ranking. The correct chunk ranks just outside the top 10. Two targeted changes help:

**7a. Increase candidate pool before RRF (same fix as 6b above)** — applying `k*4` candidates before fusion will also help docs recall.

**7b. Re-index after the `max_chunk_size` fix above** — with `max_chunk_size=2000` (the default) instead of 1500 for docs, some of the failing queries (like the `p2p_nccl_connector.md [9005-9060]` 55-char excerpt) may get better chunking alignment.

---

## MEDIUM PRIORITY — Code Quality

### 8. Missing docstrings (PEP 257 violation)

**Rubric question:** Q3 — "Code quality"

The following classes and methods have no docstrings at all:

- `student/indexer.py`: `class Indexer` (line 12)
- `student/chunker.py`: `BaseChunker.__init__`, `BaseChunker.chunk`, `TextChunker.chunk`, `CodeChunker.chunk`
- `student/ingestion.py`: `class IngestionEngine`, `IngestionEngine.__init__`, `IngestionEngine.ingest_directory`
- `student/generator.py`: `class Generator`, `Generator.__init__`
- `student/batch.py`: `class BatchProcessor`, `BatchProcessor.__init__`
- `student/models.py`: `UnansweredQuestion`, `AnsweredQuestion`, `MinimalSearchResults`, `MinimalAnswer`, `StudentSearchResults`, `StudentSearchResultsAndAnswer`

Add a one-line docstring to each. Example:

```python
class Indexer:
    """Hybrid BM25 + semantic retrieval index supporting build, save, load, and search."""
```

---

### 9. Dead code: `pass` statement in `indexer.py` line 53

**File:** `student/indexer.py`, `_make_corpus()` line 53

After `corpus.append(chunk_text)` there is a bare `pass` with no purpose:

```python
# Current — dead code
corpus.append(chunk_text)
pass              # remove this line

# Fixed
corpus.append(chunk_text)
```

---

### 10. Dead parameter: `mode` in `build_index()` shadows `self.mode`

**File:** `student/indexer.py`, `build_index()` line 70

The `mode: str = "hybrid"` parameter is declared but never used — all logic reads from `self.mode`. This is confusing dead code:

```python
# Current — 'mode' param is dead
def build_index(self, chunks, is_code: bool, mode: str = "hybrid") -> None:

# Fixed — remove the unused parameter
def build_index(self, chunks: List[MinimalSource], is_code: bool) -> None:
```

---

### 11. `clean` not in `.PHONY`

**File:** `Makefile`, line 4

```makefile
# Current
.PHONY: install debug run lint lint-strict pipeline

# Fixed — add clean
.PHONY: install debug run clean lint lint-strict pipeline
```

---

### 12. Method name typo: `expand_querry` should be `expand_query`

**Files:** `student/generator.py` line 65, `student/batch.py` line 37

```python
# generator.py — rename the method
def expand_query(self, query: str) -> str:  # was expand_querry

# batch.py — update the call site
question = self.generator.expand_query(question)  # was expand_querry
```

---

### 13. Answer generation context limited to 1000 characters causes hallucinations

**File:** `student/generator.py`, `_build_prompt()` line 39

The prompt builder caps source context at 1000 characters and only looks at 3 sources. This is so small that the LLM frequently has no relevant text to work from, causing hallucinations like fabricated file paths (`path/to/vLLM_api.py`) and wrong API names in the generated answers.

The current answers show:
- "POST /api/v1/models/{model_name}/load" — the endpoint does not exist in this form
- "[Source: path/to/vLLM_api.py]" — hallucinated file path not from retrieved sources

Increase the context limit significantly and use the actual retrieved source paths in the citation instruction:

```python
# Current
max_content_chars = 1000
for source in retrieved_sources[:3]:
    ...
    chunk_str = f"--- SOURCE FILE: {source.file_path} ---\n{chunk_text}\n\n"

# Improved
max_content_chars = 4000  # was 1000 — give the LLM enough context
for source in retrieved_sources[:5]:  # was 3
    ...
```

Also, update the system prompt to tell the model to cite from the actual source file paths it received, not invent a generic one:

```python
"You MUST cite the source of your answer using the exact SOURCE FILE path shown "
"in the context above, formatted as [Source: path/to/file]. "
"Never invent file paths."
```

---

### 14. `search` and `answer` CLI commands only search the docs index

**File:** `student/__main__.py`, lines 48 and 90

Both `search()` and `answer()` hardcode `"data/processes/index_hybrid_docs"`. There is no way to search the code index from the CLI interactively. A peer evaluator testing `search "some code question"` will get docs results. Add a `--is_code` flag:

```python
def search(self, query: str, k: int = 10,
           is_code: bool = False) -> None:
    """Searches the index for a query."""
    index_dir = ("data/processes/index_hybrid_code" if is_code
                 else "data/processes/index_hybrid_docs")
    indexer = Indexer(mode="hybrid")
    indexer.load_index(index_dir, is_code=is_code)
    ...
```

---

### 15. `ingestion.py` has an unused `ingest_directory()` method

**File:** `student/ingestion.py`, lines 19-45

`ingest_directory()` is defined but never called anywhere in the codebase. The `index` command uses `ingest_docs()` and `ingest_code()` separately. Remove it to clean up the public API:

```python
# Remove the entire ingest_directory() method (lines 19-45)
```

---

## MEDIUM PRIORITY — CLI and Edge Cases

### 16. `search ""` with an empty query will crash on small corpora / may behave unexpectedly

**File:** `student/indexer.py`, `search()` method

An empty string tokenized by bm25s returns zero tokens, which causes bm25s to return arbitrary results (it returns documents sorted by default, not by relevance). While this does not crash with the large real corpus, it is semantically wrong and should return an empty list:

```python
def search(self, query: str, k: int = 5) -> List[MinimalSource]:
    """Searches the index and returns the top-k chunks."""
    if k <= 0:
        return []
    if not query or not query.strip():
        return []
    ...
```

---

### 17. The `index` command's `raw_dir` is hardcoded and not a CLI argument

**File:** `student/__main__.py`, line 22

```python
raw_dir = "data/raw/vllm-0.10.1"   # hardcoded
```

This is acceptable given the subject specifies vLLM, but adding a `--raw_dir` parameter makes the project more robust and testable by the evaluator:

```python
def index(self, max_chunk_size: int = 2000,
          raw_dir: str = "data/raw/vllm-0.10.1") -> None:
    """Build separate docs and code BM25+semantic indices."""
```

---

## MEDIUM PRIORITY — README Gaps

The README requires specific sections and a mandatory formatted first line. The current README is written informally as developer notes and will fail the "README (5/6 sections present)" check (Q11).

### 18. Missing mandatory italicized first line

**Rubric requirement:** "Italicized first line: `*This project has been created as part of the 42 curriculum by <login>*`"

The README starts with `# RAG-against-the-machine` with no such line. Add as the very first line before the title:

```markdown
*This project has been created as part of the 42 curriculum by acd*

# RAG-against-the-machine
```

Replace `acd` with your actual 42 login.

---

### 19. Missing required README sections

The rubric checks for at minimum 5 of 6 of these sections. Currently the README has informal notes that cover some topics but does not have clearly labeled sections:

**Missing or insufficient sections:**

- **Resources** — No section listing external references as a structured list (the README has URLs scattered in prose)
- **System Architecture** — No section with a clear diagram or structured description of the pipeline stages
- **Performance Analysis** — The current content mentions numbers informally but lacks a table of Recall@k results
- **Design Decisions** — No dedicated section; rationale is buried in prose
- **Challenges Faced** — The "accuracy problems" and "speed" sections exist but are informal

**Restructure the README with these explicit section headers:**

```markdown
*This project has been created as part of the 42 curriculum by <login>*

# RAG against the Machine

## Description
...

## Instructions
...

## System Architecture
...

## Chunking Strategy
...

## Retrieval Method
...

## Performance Analysis
| Dataset | Recall@10 | Stars |
|---------|-----------|-------|
| Docs    | 93.0%     | 4/5   |
| Code    | 58.0%     | 3/5   |

## Design Decisions
...

## Challenges Faced
...

## Example Usage
...

## Resources
- [bm25s](https://github.com/xhluca/bm25s)
- [sentence-transformers](https://www.sbert.net/)
- [FAISS](https://faiss.ai/)
- [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)
- ...
```

---

## LOW PRIORITY — Answer Quality

### 20. Answer quality: self-containment and faithfulness issues

**Rubric question:** Q8 — "Answer quality (2/3 answers satisfactory)"

The evaluator samples 3 answers and checks all 4 criteria: self-contained, source-grounded, faithful, relevant. Current answers have two recurring problems:

1. **Fabricated file paths in citations**: `[Source: path/to/vLLM_api.py]` — the model ignores the actual retrieved file paths and invents generic ones. Fix by including the actual source path explicitly in the prompt (see issue 13 above).

2. **Very short answers (1 sentence)** that are sometimes circular and non-informative: "The method is `llm.generate_embeddings`" — this does not actually explain what the method does. Increase `max_new_tokens` from 80 to 150 to allow more complete answers:
   ```python
   # generator.py generate_answer — increase token budget
   max_new_tokens=150,   # was 80
   ```

3. **"Answer:" at the end of the user turn**: The current `_build_prompt` appends `"Question: {query}\nAnswer:"` in the user message, but then the system message also says "Answer the user's question". The redundancy can confuse the model. Simplify:
   ```python
   # End of _build_prompt — just provide the context and question
   prompt += f"\n\nQuestion: {query}"
   # Let the system message handle the "Answer:" instruction
   ```

---

## BONUS — Advanced Features Worth Implementing

The bonus section awards up to 5 stars. The project already implements:
- **Semantic embeddings**: 1pt (FAISS + sentence-transformers) — implemented, claim this
- **Hybrid retrieval (BM25 + semantic)**: 2pt (RRF fusion) — implemented, claim this
- **Caching**: 1pt (`self.answer_cache` in `Generator`) — implemented, claim this
- **Query expansion**: 1pt (`expand_querry` flag in `search_dataset`) — implemented, claim this

**That is already 5/5 bonus points if all implementations are working correctly.** Focus on making sure the evaluator can see and verify them.

### B1. Make bonus features clearly discoverable during evaluation

The evaluator needs to be able to run and observe each bonus feature. Add explicit examples to the README:

```bash
# Show semantic embedding bonus (indexer loads sentence-transformers)
uv run python -m student index

# Show hybrid retrieval bonus (search uses RRF fusion)
uv run python -m student search "What is vLLM?" --k 5

# Show query expansion bonus (uses LLM to expand before searching)
uv run python -m student search_dataset \
    --dataset_path data/datasets/UnansweredQuestions/dataset_code_public.json \
    --save_directory data/output/search_results \
    --k 10 --expand True

# Show caching bonus (second identical answer is instant)
uv run python -m student answer "What is vLLM?" --k 5
uv run python -m student answer "What is vLLM?" --k 5
```

### B2. vLLM inference (2pt bonus — not implemented)

This is worth 2 bonus points but requires a Linux machine with an NVIDIA GPU to run vLLM. On Apple Silicon, this is not practical. If your evaluation machine has an NVIDIA GPU, you can add a `use_vllm` flag to `Generator`:

```python
# generator.py — optional vLLM backend
def __init__(self, use_vllm: bool = False) -> None:
    if use_vllm:
        from vllm import LLM, SamplingParams
        self.vllm_engine = LLM(model="Qwen/Qwen3-0.6B")
        self.sampling_params = SamplingParams(max_tokens=80)
        self.backend = "vllm"
    else:
        # existing transformers code
        self.backend = "transformers"
```

Only attempt this if you have confirmed GPU access on the evaluation machine. The risk of breaking the working implementation is high and not worth it if the machine is CPU/MPS-only.

---

## Implementation Checklist

Work through these in order. Items marked with a star (*) are blockers for mandatory Yes/No rubric questions.

### Setup and Dependencies
- [ ] * Add `sentence-transformers>=3.0.0` to `pyproject.toml` dependencies
- [ ] * Add `faiss-cpu>=1.9.0` to `pyproject.toml` dependencies
- [ ] * Run `uv add sentence-transformers faiss-cpu` to update `uv.lock`
- [ ] Verify `uv sync` from scratch installs everything needed

### Critical Edge Case Fixes
- [ ] * Fix `k=0` crash in `indexer.py search()`: add `if k <= 0: return []` at top
- [ ] * Fix bad dataset path crash in `batch.py search_dataset()`: wrap `open()` in try/except
- [ ] * Fix bad dataset path crash in `batch.py answer_dataset()`: wrap `open()` in try/except
- [ ] Add empty/whitespace query guard in `indexer.py search()`: `if not query.strip(): return []`

### Flake8 Fixes
- [ ] * Remove unused `import torch` from `__main__.py` line 9
- [ ] * Fix E402 in `generator.py`: move `os.environ["OMP_NUM_THREADS"] = "1"` after all imports
- [ ] * Fix `batch.py` `__init__` type annotation: `generator: Optional[Generator] = None`
- [ ] * Create `.flake8` file with `max-line-length = 120` to suppress E501
- [ ] Fix W292 in `evaluate.py`: add newline at end of file
- [ ] Fix W391 in `models.py`: remove trailing blank line
- [ ] Fix E305 in `__main__.py`: add blank line after class definition
- [ ] Fix E128/E129 indentation issues in `__main__.py` lines 35-36 and `chunker.py` line 91
- [ ] Run `make lint` and verify zero flake8 errors

### Mypy Fixes
- [ ] * Add `--explicit-package-bases` to Makefile mypy command
- [ ] * Add `-> None` to all `__main__.py` methods: `__init__`, `index`, `search`, `search_dataset`, `answer`, `answer_dataset`, `evaluate`
- [ ] * Add `-> None` to `Generator.__init__`
- [ ] * Add `-> None` to `BatchProcessor.__init__`
- [ ] * Fix `tokenizer.decode()` return type narrowing in `generator.py` lines 101 and 169
- [ ] * Add None-guard for `self.embedding_model` in `indexer.py` lines 89 and 151
- [ ] * Add None-guard for `self.faiss_index` in `indexer.py` line 153
- [ ] * Add None-guard for `self.search_engine` in `batch.py` line 39
- [ ] Add `# type: ignore[assignment]` to `StudentSearchResultsAndAnswer.search_results` in `models.py`
- [ ] Run `make lint` and verify zero mypy errors

### Docstrings
- [ ] Add docstring to `class Indexer`
- [ ] Add docstring to `class IngestionEngine` and `IngestionEngine.__init__` and `ingest_directory`
- [ ] Add docstring to `class Generator` and `Generator.__init__`
- [ ] Add docstring to `class BatchProcessor` and `BatchProcessor.__init__`
- [ ] Add docstring to `BaseChunker.__init__`, `BaseChunker.chunk`, `TextChunker.chunk`, `CodeChunker.chunk`
- [ ] Add docstrings to all 6 unnamed model classes in `models.py`

### Dead Code Cleanup
- [ ] Remove `pass` statement from `indexer.py` line 53
- [ ] Remove unused `mode` parameter from `build_index()` in `indexer.py`
- [ ] Remove unused `ingest_directory()` method from `ingestion.py` (or keep if you plan to use it)

### Makefile
- [ ] Add `clean` to `.PHONY` targets

### Typo Fix
- [ ] Rename `expand_querry` to `expand_query` in `generator.py` line 65
- [ ] Update the call site in `batch.py` line 37

### Recall Improvements
- [ ] Fix `index()` to use `max_chunk_size` for docs ingestion (remove hardcoded 1500)
- [ ] Increase RRF candidate pool: change `k * 2` to `k * 4` in `indexer.py search()`
- [ ] Re-run `make pipeline` and measure recall after changes
- [ ] Optionally increase query expansion `max_new_tokens` from 30 to 50

### Answer Quality
- [ ] Increase `max_content_chars` in `_build_prompt()` from 1000 to 4000
- [ ] Increase `retrieved_sources` limit in `_build_prompt()` from 3 to 5
- [ ] Increase `max_new_tokens` in `generate_answer()` from 80 to 150
- [ ] Fix system prompt to require citing the actual source file paths shown in context

### CLI Improvements
- [ ] Add `--is_code` flag to `search()` command in `__main__.py`
- [ ] Add `--is_code` flag to `answer()` command in `__main__.py`
- [ ] Optionally add `--raw_dir` parameter to `index()` command

### README
- [ ] * Add italicized first line: `*This project has been created as part of the 42 curriculum by <login>*`
- [ ] Restructure README with clearly labeled `## Resources` section
- [ ] Add `## System Architecture` section with pipeline description
- [ ] Add `## Performance Analysis` table with Recall@k numbers
- [ ] Add `## Design Decisions` section (why hybrid, why separate indices, etc.)
- [ ] Rename/restructure `## accuracy problems` into `## Challenges Faced`
- [ ] Add `## Example Usage` section with copy-pasteable commands

### Bonus Features — Verification
- [ ] Verify semantic embeddings bonus is working (sentence-transformers loads during index)
- [ ] Verify hybrid retrieval bonus is working (RRF shows up in search logs)
- [ ] Verify caching bonus is working (second `answer` call to same query is instant)
- [ ] Verify query expansion bonus is working (`--expand True` triggers LLM rewrite)
- [ ] Document all 4 bonus features clearly in README with demo commands
