import os
import uuid
from src.indexer import Indexer
from src.evaluate import Evaluator
from src.ingestion import IngestionEngine
from src.generator import Generator
from src.batch import BatchProcessor
from src.config import (
    MAX_CODE_CHUNK_SIZE,
    MAX_DOCS_CHUNK_SIZE,
    DEFAULT_K,
    EVAL_K,
    RAW_DATA_DIR,
    DOCS_INDEX_DIR,
    CODE_INDEX_DIR,
)
import fire


class RAGCLI:
    """
    Command-line interface for the RAG pipeline.
    """

    def __init__(self) -> None:
        self.evaluator = Evaluator()

    # ------------------------------------------------------------------ #
    # Indexing                                                           #
    # ------------------------------------------------------------------ #

    def index(
        self,
        max_chunk_size: int = MAX_CODE_CHUNK_SIZE,
        docs_chunk_size: int = MAX_DOCS_CHUNK_SIZE,
        semantic: bool = False,
    ) -> None:
        """Build the docs and code retrieval indices.

        Args:
            max_chunk_size:  Max characters per code chunk (default 2000).
            docs_chunk_size: Max characters per docs chunk (default 1800).
            semantic:        Also build FAISS dense-vector index for semantic
                             search (slow — takes several extra minutes).
                             Use `make bonus` to run the full pipeline with
                             this flag set.
        """
        if semantic:
            print("=== Semantic mode ON: will build BM25 + FAISS indices ===")
        else:
            print("=== BM25-only mode (fast). "
                  "Pass --semantic True for hybrid search. ===")

        print("=== Building docs index (md / txt / setup.py) ===")
        docs_ingestion = IngestionEngine(max_chunk_size=docs_chunk_size)
        docs_data = docs_ingestion.ingest_docs(RAW_DATA_DIR)
        # Pass semantic flag so the indexer knows
        docs_indexer = Indexer(semantic=semantic)
        docs_indexer.build_index(docs_data, is_code=False)
        docs_indexer.save_index(DOCS_INDEX_DIR)

        print("=== Building code index (.py files) ===")
        code_ingestion = IngestionEngine(max_chunk_size=max_chunk_size)
        code_data = code_ingestion.ingest_code(RAW_DATA_DIR)
        code_indexer = Indexer(semantic=semantic)
        code_indexer.build_index(code_data, is_code=True)
        code_indexer.save_index(CODE_INDEX_DIR)

    # ------------------------------------------------------------------ #
    # Retrieval                                                          #
    # ------------------------------------------------------------------ #

    def search(
        self,
        query: str,
        k: int = DEFAULT_K,
        semantic: bool = False,
    ) -> None:
        """Search both docs and code indices for a single query.

        Args:
            query:    The question or keyword string to look up.
            k:        Number of results to return per index.
            semantic: Use FAISS-enhanced search (requires a semantic index).
        """
        print(f"Searching for: '{query}'")

        docs_indexer = Indexer(semantic=semantic)
        docs_indexer.load_index(DOCS_INDEX_DIR, is_code=False)
        docs_chunks = docs_indexer.search(query, k)
        print("\n--- Top Docs Results ---")
        for i, chunk in enumerate(docs_chunks):
            print(
                f"{i + 1}. {chunk.file_path} "
                f"[Chars {chunk.first_character_index}"
                f":{chunk.last_character_index}]")

        code_indexer = Indexer(semantic=semantic)
        code_indexer.load_index(CODE_INDEX_DIR, is_code=True)
        code_chunks = code_indexer.search(query, k)
        print("\n--- Top Code Results ---")
        for i, chunk in enumerate(code_chunks):
            print(
                f"{i + 1}. {chunk.file_path} "
                f"[Chars {chunk.first_character_index}"
                f":{chunk.last_character_index}]")

    def search_dataset(
        self,
        dataset_path: str,
        save_directory: str,
        k: int = DEFAULT_K,
        expand: bool = False,
        semantic: bool = False,
    ) -> None:
        """Process a full dataset of questions and save search results.

        Args:
            dataset_path:    Path to the UnansweredQuestions JSON file.
            save_directory:  Where to write the output JSON.
            k:               Top-k results to retrieve per question.
            expand:          Use the LLM to rewrite the query before searching
                             (query expansion bonus feature).
            semantic:        Use FAISS-enhanced search (requires semantic
                             index).
        """
        # Choose the right index based on whether dataset is for code or docs.
        name = os.path.basename(dataset_path)
        if "code" in name:
            index_dir = CODE_INDEX_DIR
            is_code = True
        else:
            index_dir = DOCS_INDEX_DIR
            is_code = False

        indexer = Indexer(semantic=semantic)
        indexer.load_index(index_dir, is_code=is_code)

        gen = None
        if expand:
            print("Loading Generator for query expansion...")
            gen = Generator()

        batcher = BatchProcessor(search_engine=indexer, generator=gen)
        batcher.search_dataset(
            dataset_path=dataset_path,
            save_directory=save_directory,
            k=k,
            expand=expand,
        )

    # ------------------------------------------------------------------ #
    # Answer generation                                                  #
    # ------------------------------------------------------------------ #

    def answer(
        self,
        query: str,
        k: int = DEFAULT_K,
        semantic: bool = False,
    ) -> None:
        """Search the docs index and generate an LLM answer for one query.

        Args:
            query:    The question to answer.
            k:        Number of context chunks to retrieve.
            semantic: Use FAISS-enhanced search (requires semantic index).
        """
        print(f"Answering query: '{query}'")

        # Retrieve context from the docs index.
        indexer = Indexer(semantic=semantic)
        indexer.load_index(DOCS_INDEX_DIR, is_code=False)
        found_chunks = indexer.search(query, k)

        # Generate the answer using the LLM.
        gen = Generator()
        temp_id = str(uuid.uuid4())  # temporary ID — no dataset here
        answer_obj = gen.generate_answer(
            question_id=temp_id,
            query=query,
            retrieved_sources=found_chunks,
        )

        print("\n--- AI Answer ---")
        print(answer_obj.answer)

    def answer_dataset(
        self,
        student_search_results_path: str,
        save_directory: str,
    ) -> None:
        """Read search results and generate an LLM answer per question.

        Args:
            student_search_results_path: Output of search_dataset.
            save_directory:              Where to write the answers JSON.
        """
        gen = Generator()
        # search_engine=None because answering only reads from the JSON file;
        # it does not need to query the index again.
        batcher = BatchProcessor(search_engine=None, generator=gen)
        batcher.answer_dataset(
            student_search_results_path=student_search_results_path,
            save_directory=save_directory,)

    # ------------------------------------------------------------------ #
    # Evaluation                                                         #
    # ------------------------------------------------------------------ #

    def evaluate(
        self,
        student_results_path: str,
        dataset_path: str,
        k: int = EVAL_K,
        max_context_length: int = 2000,
    ) -> None:
        """Evaluate search results against ground truth using Recall@k.

        Args:
            student_results_path: Path to the student search results JSON.
            dataset_path:         Path to the ground-truth JSON.
            k:                    Recall@k cutoff.
            max_context_length:   Max chunk length for evaluation.
        """
        self.evaluator.evaluate(
            student_results_path, dataset_path, k, max_context_length)


if __name__ == "__main__":
    fire.Fire(RAGCLI)
