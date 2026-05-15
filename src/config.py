import torch

# ── Chunking

# Larger = more surrounding context but may dilute relevance.
MAX_CODE_CHUNK_SIZE: int = 2000
CHUNK_OVERLAP: int = 200
MAX_DOCS_CHUNK_SIZE: int = 1800

# ── Retrieval

DEFAULT_K: int = 10

# Each retriever (BM25 and FAISS) -> k * CANDIDATE_MULT candidates
# before Reciprocal Rank Fusion. Higher = better fusion quality at the cost
# of more computation. range: 5–50
CANDIDATE_MULT: int = 20

# Smoothing constant in the RRF formula: score = 1 / (RRF_CONSTANT + rank).
# The original RRF paper uses 60. Lower values give more weight to top ranks.
# Typical alternatives: 1–100.
RRF_CONSTANT: int = 20

# ── Embedding model
# Sentence-transformer model used to build dense FAISS vectors.
EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

# Number of chunks encoded in one forward pass through the embedding model.
# Default is 32 — too small for MPS/GPU. Higher = more parallelism = faster.
# Lower if you run out of memory. Range: 32–512.
EMBEDDING_BATCH_SIZE: int = 256

# ── LLM / Generator
# smaller: Qwen/Qwen2.5-0.5B big: Qwen/Qwen3-0.6B
LLM_MODEL_NAME: str = "Qwen/Qwen3-0.6B"

# Numerical precision for model weights.
# torch.float32  → full precision, higher memory (~2× bfloat16)
# torch.bfloat16 → halves VRAM/RAM with negligible quality loss on modern GPUs
LLM_DTYPE: torch.dtype = torch.float16


MAX_NEW_TOKENS_EXPAND: int = 50
MAX_NEW_TOKENS_ANSWER: int = 40

# True can improve accuracy but significantly increases latency and token cost.
ENABLE_THINKING: bool = False
OMP_NUM_THREADS: str = "1"

# ── Prompt construction
# Maximum total characters of retrieved source text injected into the prompt.
MAX_CONTEXT_CHARS: int = 1500

# Maximum number of retrieved sources spliced into the prompt.
MAX_CONTEXT_SOURCES: int = 3

# ── Evaluation
# k used when computing Recall@k and Precision@k during evaluation.
EVAL_K: int = 10

OVERLAP_THRESHOLD: float = 0.05
DOCS_PASS_THRESHOLD: float = 0.80
CODE_PASS_THRESHOLD: float = 0.50

# ── Ingestion filtering

EXCLUDE_DIRS: frozenset = frozenset({
    'benchmarks', 'examples', 'tests', '.buildkite', 'requirements',
})

# ── Index storage paths
# Directory containing the raw corpus
RAW_DATA_DIR: str = "data/raw/vllm-0.10.1"

DOCS_INDEX_DIR: str = "data/processes/index_hybrid_docs"
CODE_INDEX_DIR: str = "data/processes/index_hybrid_code"

# ── Logging
LOG_FILE: str = "ingestion_errors.log"
