
from student.indexer import Indexer
from student.chunker import CodeChunker, TextChunker
from student.ingestion import IngestionEngine
from student.generator import Generator
from student.batch import BatchProcessor
import fire
import pathlib
import subprocess
import uuid
import bm25s

class RAGCLI:
    """
    Command line interface for the RAG pipeline
    """

    def index(self, max_chunk_size: int = 2000):
        """Index a raw rep and save the BM25 model"""
        print("Starting indexing process with",
              f" max_chunk_size={max_chunk_size}...")
        ingestion = IngestionEngine(max_chunk_size=max_chunk_size)
        data = ingestion.ingest_directory("data/raw/vllm-0.10.1")
        indexer = Indexer()
        indexer.build_index(data)
        indexer.save_index("data/processes/index_bm25")

    # ==== Retrival Phase ====

    def search(self, query: str, k: int = 10):
        """ Searches the index for a query """

        print(f"Searching for: '{query}'")
        indexer = Indexer()
        indexer.load_index("data/processes/index_bm25")
        found_chunks = indexer.search(query, k)
        print("\n--- Top Results ---")
        for i, chunk in enumerate(found_chunks):
            print(
                f"{i+1}. {chunk.file_path} [Chars ",
                f"{chunk.first_character_index}:{chunk.last_character_index}]")

    def search_dataset(self,
                       dataset_path: str,
                       save_directory: str,
                       k: int = 10):
        """ Process a dataset of questions and save the res"""
        indexer = Indexer()
        indexer.load_index("data/processes/index_bm25")
        batcher = BatchProcessor(search_engine=indexer)
        batcher.search_dataset(dataset_path=dataset_path,
                               save_directory=save_directory,
                               k=k)

    # ==== Augmentation Phase ====

    def answer(self, query: str, k: int = 10):
        """Search the index and generate AI answer for a query"""
        print(f"Answering query: '{query}'")
        # 1. Search
        indexer = Indexer()
        indexer.load_index("data/processes/index_bm25")
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
       

if __name__ == "__main__":
    fire.Fire(RAGCLI)
