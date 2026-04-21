
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
        import json
        
        print("--- Running Custom Recall@k Evaluation ---")
        
        # 1. Load the JSON data
        with open(student_results_path, 'r') as f:
            student_data = json.load(f)
        with open(dataset_path, 'r') as f:
            ground_truth_data = json.load(f)

        # 2. Map ground truth sources by question_id for quick lookup
        truth_map = {q['question_id']: q['sources'] for q in ground_truth_data['rag_questions']}

        total_recall = 0.0
        num_questions = len(student_data['search_results'])

        # 3. Calculate Recall@k for every question
        for result in student_data['search_results']:
            q_id = result['question_id']
            # Slice the predicted list to ensure we only look at top 'k'
            pred_sources = result['retrieved_sources'][:k] 
            expected_sources = truth_map.get(q_id, [])

            if not expected_sources:
                continue

            found_count = 0
            # For every correct source, check if we found it
            for exp in expected_sources:
                exp_len = exp['last_character_index'] - exp['first_character_index']
                is_found = False

                for pred in pred_sources:
                    # Must be the exact same file
                    if exp['file_path'] == pred['file_path']:
                        # Calculate mathematical overlap
                        overlap_start = max(exp['first_character_index'], pred['first_character_index'])
                        overlap_end = min(exp['last_character_index'], pred['last_character_index'])
                        overlap_len = max(0, overlap_end - overlap_start)

                        # Check if it meets the 5% threshold mandated by the subject
                        if (overlap_len / exp_len) >= 0.05:
                            is_found = True
                            break 

                if is_found:
                    found_count += 1

            # Question Recall = (Sources Found / Total Expected Sources)
            question_recall = found_count / len(expected_sources)
            total_recall += question_recall

        # 4. Final System Score
        final_recall = total_recall / num_questions

        print(f"Questions evaluated: {num_questions}")
        print(f"Recall@{k}: {final_recall:.3f}")

        # Dynamically check the threshold based on the filename
        if "docs" in dataset_path:
            if final_recall >= 0.80:
                print("✅ PASS! You achieved >= 80% on the docs dataset.")
            else:
                print("❌ FAIL. You are below the 80% docs threshold.")
        elif "code" in dataset_path:
            if final_recall >= 0.50:
                print("✅ PASS! You achieved >= 50% on the code dataset.")
            else:
                print("❌ FAIL. You are below the 50% code threshold.")

if __name__ == "__main__":
    fire.Fire(RAGCLI)
