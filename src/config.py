import torch

# ── Chunking ─────────────────────────────────────────────────────────────────
# Maximum character length per code chunk.
# Smaller = more precise retrieval but less context per result.
# Larger = more surrounding context but may dilute relevance.
MAX_CODE_CHUNK_SIZE: int = 2000

# Characters of overlap between consecutive text chunks.
# Prevents recall misses when a relevant passage spans a chunk boundary.
CHUNK_OVERLAP: int = 200

MAX_DOCS_CHUNK_SIZE: int = 1800

# ── Retrieval ─────────────────────────────────────────────────────────────────
# Number of top results returned to the caller after RRF fusion.
# Higher = more recall but noisier results. Typical range: 3–20.
DEFAULT_K: int = 10

# Each retriever (BM25 and FAISS) fetches k * CANDIDATE_MULT candidates
# before Reciprocal Rank Fusion. Higher = better fusion quality at the cost
# of more computation. Typical range: 5–50.
CANDIDATE_MULT: int = 20

# Smoothing constant in the RRF formula: score = 1 / (RRF_CONSTANT + rank).
# The original RRF paper uses 60. Lower values give more weight to top ranks.
# Typical alternatives: 1–100.
RRF_CONSTANT: int = 60

# ── Embedding model ───────────────────────────────────────────────────────────
# Sentence-transformer model used to build dense FAISS vectors.
# Changing this requires a full index rebuild (dimensions differ per model).
#   all-MiniLM-L6-v2   →  384-dim, MTEB ~56, fast
#   all-mpnet-base-v2  →  768-dim, MTEB ~65, medium speed
#   BAAI/bge-base-en-v1.5 → 768-dim, MTEB ~64, medium speed
#   intfloat/e5-large-v2  → 1024-dim, MTEB ~67, slow
EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

# Number of chunks encoded in one forward pass through the embedding model.
# Default is 32 — too small for MPS/GPU. Higher = more parallelism = faster.
# Lower if you run out of memory. Range: 32–512.
EMBEDDING_BATCH_SIZE: int = 256

# ── LLM / Generator ───────────────────────────────────────────────────────────
# Local causal LM used for query expansion and answer generation.
# Must be a HuggingFace model ID compatible with AutoModelForCausalLM.
LLM_MODEL_NAME: str = "Qwen/Qwen3-0.6B"

# Numerical precision for model weights.
# torch.float32  → full precision, higher memory (~2× bfloat16)
# torch.bfloat16 → halves VRAM/RAM with negligible quality loss on modern GPUs
LLM_DTYPE: torch.dtype = torch.float32

# Maximum tokens the model may generate during query expansion.
# Keep short: the goal is a rephrased query, not a full answer. Range: 30–100.
MAX_NEW_TOKENS_EXPAND: int = 50

# Maximum tokens the model may generate for the final answer.
# 40 is tight; answers may be cut off for complex questions. Range: 40–150.
MAX_NEW_TOKENS_ANSWER: int = 40

# Whether to enable Qwen3 chain-of-thought "thinking" tokens.
# True can improve accuracy but significantly increases latency and token cost.
ENABLE_THINKING: bool = False

# Number of CPU threads used by OpenMP during inference.
# Setting to "1" prevents oversubscription when running on a single core.
OMP_NUM_THREADS: str = "4"

# ── Prompt construction ───────────────────────────────────────────────────────
# Maximum total characters of retrieved source text injected into the prompt.
# At ~4 chars/token this is ~375 tokens — raise to 3000–4000 for richer context.
MAX_CONTEXT_CHARS: int = 1500

# Maximum number of retrieved sources spliced into the prompt.
# The loop stops at whichever limit (chars or sources) is hit first.
# Typical range: 3–10.
MAX_CONTEXT_SOURCES: int = 3

# ── Evaluation ────────────────────────────────────────────────────────────────
# k used when computing Recall@k and Precision@k during evaluation.
EVAL_K: int = 10

# Minimum character-overlap ratio for a retrieved chunk to count as a hit.
# 0.05 = at least 5% of the ground-truth answer must appear in the chunk.
OVERLAP_THRESHOLD: float = 0.05

# Minimum Recall@k required to pass the docs benchmark.
DOCS_PASS_THRESHOLD: float = 0.80

# Minimum Recall@k required to pass the code benchmark.
CODE_PASS_THRESHOLD: float = 0.50

# ── Ingestion filtering ───────────────────────────────────────────────────────
# Directories skipped during both docs and code ingestion.
# These contain noise (CI scripts, benchmarks, test fixtures) that pollutes
# the index without appearing in any ground-truth answers.
EXCLUDE_DIRS: frozenset = frozenset({
    'benchmarks', 'examples', 'tests', '.buildkite', 'requirements',
})

# ── Index storage paths ───────────────────────────────────────────────────────
# Directory containing the raw corpus (relative to project root).
RAW_DATA_DIR: str = "data/raw/vllm-0.10.1"

# Where the docs hybrid index (BM25 + FAISS) is saved and loaded from.
DOCS_INDEX_DIR: str = "data/processes/index_hybrid_docs"

# Where the code hybrid index (BM25 + FAISS) is saved and loaded from.
CODE_INDEX_DIR: str = "data/processes/index_hybrid_code"

# ── Logging ───────────────────────────────────────────────────────────────────
# File where ingestion warnings and errors are written.
LOG_FILE: str = "ingestion_errors.log"
