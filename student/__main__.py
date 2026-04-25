
import os
from student.indexer import Indexer
from student.evaluate import Evaluator
from student.ingestion import IngestionEngine
from student.generator import Generator
from student.batch import BatchProcessor
import fire
import torch
import uuid


class RAGCLI:
    """
    Command line interface for the RAG pipeline
    """
    def __init__(self):
        self.evaluator = Evaluator()

    def index(self, max_chunk_size: int = 2000):
        """Build separate docs and code BM25 indices."""
        raw_dir = "data/raw/vllm-0.10.1"

        print("=== Building docs index (md/txt) ===")
        docs_ingestion = IngestionEngine(max_chunk_size=1500)
        docs_data = docs_ingestion.ingest_docs(raw_dir)
        docs_indexer = Indexer(mode="semantic")
        docs_indexer.build_index(docs_data, is_code=False)
        docs_indexer.save_index("data/processes/index_semantic_docs")

        print("=== Building code index (py) ===")
        code_ingestion = IngestionEngine(max_chunk_size=max_chunk_size)
        code_data = code_ingestion.ingest_code(raw_dir)
        code_data = [c for c in code_data if '/tests/' not in c.file_path
            and '/benchmarks/' not in c.file_path
            and '/examples/' not in c.file_path]
        code_indexer = Indexer(mode="semantic")
        code_indexer.build_index(code_data, is_code=True)
        code_indexer.save_index("data/processes/index_semantic_code")

    # ==== Retrival Phase ====

    def search(self, query: str, k: int = 10):
        """ Searches the index for a query """

        print(f"Searching for: '{query}'")
        indexer = Indexer()
        indexer.load_index("data/processes/index_semantic_docs", is_code=False)
        found_chunks = indexer.search(query, k)
        print("\n--- Top Results ---")
        for i, chunk in enumerate(found_chunks):
            print(
                f"{i+1}. {chunk.file_path} [Chars ",
                f"{chunk.first_character_index}:{chunk.last_character_index}]")

    def search_dataset(self,
                       dataset_path: str,
                       save_directory: str,
                       k: int = 10,
                       expand: bool = False):
        """
        Process a dataset of questions and save the res
        """
        name = os.path.basename(dataset_path)
        if "code" in name:
            index_dir = "data/processes/index_semantic_code"
            is_code = True
        else:
            index_dir = "data/processes/index_semantic_docs"
            is_code = False
        indexer = Indexer(mode="semantic")
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

    def answer(self, query: str, k: int = 10):
        """Search the index and generate AI answer for a query"""
        print(f"Answering query: '{query}'")
        # 1. Search
        indexer = Indexer(mode="semantic")
        indexer.load_index("data/processes/index_semantic_docs", is_code=False)
        found_chunks = indexer.search(query, k)

        # 2. Generate
        gen = Generator()
        temp_id = str(uuid.uuid4())  # Generate a fake UUID since there's no dataset
        answer_obj = gen.generate_answer(
            question_id=temp_id,
            query=query,
            retrieved_sources=found_chunks
        )

        print("\n--- AI Answer ---")
        print(answer_obj.answer)

    def answer_dataset(self,
                       student_search_results_path: str,
                       save_directory: str):
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

    def evaluate(self, student_results_path: str, dataset_path: str, k: int = 10, max_context_length: int = 2000):
        """Evaluates student search results against the ground truth using custom Recall@k."""
        self.evaluator.evaluate(student_results_path, dataset_path,
                                k, max_context_length)

if __name__ == "__main__":
    fire.Fire(RAGCLI)
