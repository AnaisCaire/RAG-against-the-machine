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
    Command line interface for the RAG pipeline
    """

    def __init__(self) -> None:
        self.evaluator = Evaluator()

    def index(
        self,
        max_chunk_size: int = MAX_CODE_CHUNK_SIZE,
        docs_chunk_size: int = MAX_DOCS_CHUNK_SIZE,
    ) -> None:
        """Build separate docs and code BM25 indices."""
        print("=== Building docs index (md/txt/setup.py) ===")
        docs_ingestion = IngestionEngine(max_chunk_size=docs_chunk_size)
        docs_data = docs_ingestion.ingest_docs(RAW_DATA_DIR)
        docs_indexer = Indexer()
        docs_indexer.build_index(docs_data, is_code=False)
        docs_indexer.save_index(DOCS_INDEX_DIR)

        print("=== Building code index (py) ===")
        code_ingestion = IngestionEngine(max_chunk_size=max_chunk_size)
        code_data = code_ingestion.ingest_code(RAW_DATA_DIR)

        code_indexer = Indexer()
        code_indexer.build_index(code_data, is_code=True)
        code_indexer.save_index(CODE_INDEX_DIR)

    # ==== Retrival Phase ====

    def search(self, query: str, k: int = DEFAULT_K) -> None:
        """ Searches both docs and code indices for a query """

        print(f"Searching for: '{query}'")

        docs_indexer = Indexer()
        docs_indexer.load_index(DOCS_INDEX_DIR, is_code=False)
        docs_chunks = docs_indexer.search(query, k)
        print("\n--- Top Docs Results ---")
        for i, chunk in enumerate(docs_chunks):
            print(
                f"{i+1}. {chunk.file_path} "
                f"[Chars {chunk.first_character_index}"
                f":{chunk.last_character_index}]"
            )

        code_indexer = Indexer()
        code_indexer.load_index(CODE_INDEX_DIR, is_code=True)
        code_chunks = code_indexer.search(query, k)
        print("\n--- Top Code Results ---")
        for i, chunk in enumerate(code_chunks):
            print(
                f"{i+1}. {chunk.file_path} "
                f"[Chars {chunk.first_character_index}"
                f":{chunk.last_character_index}]"
            )

    def search_dataset(self,
                       dataset_path: str,
                       save_directory: str,
                       k: int = DEFAULT_K,
                       expand: bool = False) -> None:
        """
        Process a dataset of questions and save the res
        """
        name = os.path.basename(dataset_path)
        if "code" in name:
            index_dir = CODE_INDEX_DIR
            is_code = True
        else:
            index_dir = DOCS_INDEX_DIR
            is_code = False
        indexer = Indexer()
        indexer.load_index(index_dir, is_code=is_code)
        gen = None
        if expand:
            print("Loading Generator for Query Expansion...")
            gen = Generator()
        batcher = BatchProcessor(search_engine=indexer, generator=gen)
        batcher.search_dataset(dataset_path=dataset_path,
                               save_directory=save_directory,
                               k=k,
                               expand=expand)

    # ==== Augmentation Phase ====

    def answer(self, query: str, k: int = DEFAULT_K) -> None:
        """Search the index and generate AI answer for a query"""
        print(f"Answering query: '{query}'")
        # 1. Search
        indexer = Indexer()
        indexer.load_index(DOCS_INDEX_DIR, is_code=False)
        found_chunks = indexer.search(query, k)

        # 2. Generate
        gen = Generator()
        temp_id = str(uuid.uuid4())  # fake UUID; no dataset available
        answer_obj = gen.generate_answer(
            question_id=temp_id,
            query=query,
            retrieved_sources=found_chunks
        )

        print("\n--- AI Answer ---")
        print(answer_obj.answer)

    def answer_dataset(self,
                       student_search_results_path: str,
                       save_directory: str) -> None:
        """
        Reads search results, generates AI answers, and saves the final JSON.
        """
        gen = Generator()
        # We can pass None for search_engine since
        # answering a dataset only reads from the JSON!
        batcher = BatchProcessor(search_engine=None, generator=gen)
        batcher.answer_dataset(
            student_search_results_path=student_search_results_path,
            save_directory=save_directory)

    def evaluate(self, student_results_path: str, dataset_path: str,
                 k: int = EVAL_K, max_context_length: int = 2000) -> None:
        """
        Evaluates student search results against ground truth
        using custom Recall@k.
        """
        self.evaluator.evaluate(student_results_path, dataset_path,
                                k, max_context_length)


if __name__ == "__main__":
    fire.Fire(RAGCLI)
