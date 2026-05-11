*This project has been created as part of the 42 curriculum by acaire-d.*

# RAG-against-the-machine

## Description

This project implements a complete Retrieval-Augmented Generation (RAG) system designed specifically for codebases. The goal of the system is to ingest the vLLM repository, intelligently chunk and index both Python code and Markdown documentation, and answer highly specific user queries by retrieving the most relevant context and generating evidence-based responses using the Qwen-0.5B Large Language Model.

---

## Instructions

### Standard Installation

This project uses `uv` as its package and dependency manager. To set up the environment and install all required dependencies (like `transformers`, `pydantic`, `bm25s`, `faiss-cpu`, and `fire`), run:

```bash
make install
```

### 42 School Computer Workaround (GoInfre)

Due to strict storage limits on school workstations, installing large AI models and dependencies directly to the home directory will cause a "No space left on device" error. To successfully run this on a school computer, you must route storage to goinfre:

**Create directories in goinfre:**

```bash
mkdir -p /goinfre/$USER/bin
mkdir -p /goinfre/$USER/uv_cache
mkdir -p /goinfre/$USER/hf_cache
```

**Reroute environment variables:**

```bash
export PATH="/goinfre/$USER/bin:$PATH"
export UV_CACHE_DIR="/goinfre/$USER/uv_cache"
export HF_HOME="/goinfre/$USER/hf_cache"
```

**Install uv directly to goinfre:**

```bash
curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="/goinfre/$USER/bin" sh
```

---

## Example Usage

The entire pipeline is exposed as a Command-Line Interface (CLI) powered by Python Fire. All commands are executed through the src module.

### 1. Build the Index

```bash
uv run python -m src index --max_chunk_size 2000
```

### 2. Answer a Single Query

```bash
uv run python -m src answer "What activation formats does the fused batched MoE layer return in vLLM?" --k 10
```

### 3. Batch Search a Dataset

```bash
uv run python -m src search_dataset --dataset_path data/datasets/UnansweredQuestions/dataset_code_public.json --k 10 --save_directory data/output/search_results
```

### 4. Batch Answer a Dataset

```bash
uv run python -m src answer_dataset --student_search_results_path data/output/search_results/dataset_code_public.json --save_directory data/output/search_results_and_answer
```

### 5. Evaluate Performance

```bash
uv run python -m src evaluate --student_results_path="data/output/search_results/dataset_code_public.json" --dataset_path="data/datasets/AnsweredQuestions/dataset_code_public.json" --k=10
```

---

## System Architecture

The system is divided into four strictly typed, object-oriented components:

- **Ingestion Engine:** Crawls the raw directory using `os.walk()`, identifies file types, routes them to the appropriate chunker, and compiles a master list of `MinimalSource` objects.

- **Chunking Suite:** Segments text using either AST parsing (for code) or semantic boundaries (for Markdown).

- **Indexer/Retriever:** Converts chunks into a readable corpus, tokenizes the text, and trains a BM25 mathematical model (and a FAISS vector database) to retrieve the top k most relevant chunks for a given query.

- **Generator:** Takes the retrieved context, structures it into a strict Prompt Template with source citations, and uses the Qwen LLM via Hugging Face transformers to generate a concise answer.

---

## Chunking Strategy

The system implements two distinct strategies to respect the max_chunk_size limit (default 2000 characters) without destroying semantic meaning:

### Code Chunking (ast module)

Python code is parsed into an Abstract Syntax Tree. The system extracts logical, unbroken blocks (like `ClassDef` and `FunctionDef`). If a class exceeds the character limit, the chunker recursively descends into the class body to chunk individual methods, ensuring signatures and logic blocks remain intact.

### Text Chunking (Graceful Degradation)

For Markdown/text, the chunker searches backward from the 2000-character limit to find the most semantic breaking point: prioritizing double newlines (paragraphs), then single newlines, and finally spaces.

---

## Retrieval Method

The system utilizes a Hybrid Search Engine:

- **Lexical Search (BM25):** Evaluates exact keyword matches and term frequency. It is heavily optimized for code by splitting camelCase and snake_case variables into searchable tokens.

- **Semantic Search (FAISS + Sentence Transformers):** Translates chunks into dense vector embeddings (normalized to length 1) to understand the meaning of natural language documentation via Cosine Similarity.

- **Reciprocal Rank Fusion (RRF):** The mathematical rankings of both BM25 and FAISS are combined using RRF to bubble the most contextually relevant chunks to the top of the search results.

---

## Design Decisions

### Dynamic BM25 Normalization (b parameter)

The length normalization dial was customized based on file type. Documentation uses b=0.75 (to penalize rambling text), while code uses b=0.3 (because long Python classes are logically dense, not rambling).

### Code Text Cleaning (_clean_code_text)

BM25 struggles with rigid programming syntax. A regex-based cleaner was implemented to translate lines like `BlockSpaceManager` into `block space manager`, vastly improving keyword matching.

### Stopword Removal

Words like "how" and "the" were stripped from the tokenizer to prevent the math model from being distracted by grammar during documentation retrieval.

---

## Performance Analysis

The system successfully met and exceeded the required thresholds:

- **Retrieval Quality:** Achieved a Recall@5 of >= 80% on the Documentation dataset and >= 50% on the Code dataset.

- **Generation Speed:** By truncating the retrieved context sent to the LLM (reducing k processing) and setting max_new_tokens, generation time was slashed to meet the < 2 seconds per question requirement. Apple Silicon (MPS) and Bfloat16 precision were utilized where applicable to accelerate inference.

---

## Bonus Features

### LLM-Powered Query Expansion

Before searching, the LLM brainstorms technical synonyms and variable names related to the user's natural language question, augmenting the query and bridging the "lexical gap" between English and code syntax.

### Result Caching (Memory Bank)

An in-memory hash map intercepts duplicate queries during batch processing, instantly returning cached answers (<0.001s) to save massive amounts of GPU compute time.

### Semantic Embeddings

FAISS and the all-MiniLM-L6-v2 transformer were implemented alongside BM25 to allow for dense vector search and Hybrid RRF ranking.

---

## Challenges Faced

### AST Decorator Offsets

Python's AST module calculates the start of a `FunctionDef` at the `def` keyword, ignoring decorators (e.g., `@property`). This caused offset errors with the Moulinette. **Solution:** Intercepted the `decorator_list` to manually recalculate the absolute start index.

### LLM "Thinking" Loops

The small Qwen 0.5B model hallucinated chain-of-thought blocks (`<think>`) and refused to answer concisely. **Solution:** Transitioned from the high-level pipeline wrapper to raw `AutoTokenizer` and `AutoModelForCausalLM` usage, strictly enforcing `enable_thinking=False` and deterministic sampling.

### IoU Mathematics

Initial custom evaluations passed locally but failed the Moulinette. **Solution:** Realized the evaluation required Intersection over Union (IoU) rather than Expected Length, requiring chunk sizes to be tuned down to reduce noise.

---

## Resources

### Libraries

- `bm25s`
- `faiss-cpu`
- `sentence-transformers`
- `transformers`
- `pydantic`

### Documentation

- Python AST module
- Green Tree Snakes AST Guide

### AI Usage

AI was used extensively throughout this project as a Socratic Tech Lead/Tutor. It was utilized to explain complex pipeline architectures (like Reciprocal Rank Fusion and Tokenizer mechanics), debug AST line-numbering behaviors, and guide the optimization of PyTorch hardware acceleration for Apple Silicon and Linux CPUs.