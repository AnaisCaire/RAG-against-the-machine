import os
import json
from typing import Optional
from src.models import (
    RagDataset,
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
    MinimalSearchResults
)
from src.indexer import Indexer
from src.generator import Generator
from tqdm import tqdm


class BatchProcessor:
    """Runs search and answer generation over a full dataset of questions."""

    def __init__(self,
                 search_engine: Optional[Indexer],
                 generator: Optional[Generator] = None) -> None:
        """
        Stores the retrieval engine and optional
        LLM generator for batch operations.
        """
        self.search_engine = search_engine
        self.generator = generator

    def search_dataset(self,
                       dataset_path: str,
                       save_directory: str,
                       k: int = 5,
                       expand: bool = False) -> None:
        """
        Reads a dataset, searches the index for each question
        and saves the results.
        """
        print(f"Processing dataset: {dataset_path}")

        try:
            with open(dataset_path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
        except FileNotFoundError:
            print(f"Error: dataset file not found: {dataset_path}")
            return
        except json.JSONDecodeError as e:
            print(f"Error: invalid JSON in {dataset_path}: {e}")
            return

        if self.search_engine is None:
            raise RuntimeError(
                "search_engine is None — cannot run search_dataset."
            )

        dataset = RagDataset(**raw_data)
        results_list = []

        for q in tqdm(dataset.rag_questions, desc="Searching dataset"):
            question_id = q.question_id
            question = q.question

            if expand and self.generator is not None:
                question = self.generator.expand_query(question)

            found_sources = self.search_engine.search(query=question, k=k)
            search_result_obj = MinimalSearchResults(
                question_id=question_id,
                question_str=q.question,
                retrieved_sources=found_sources)
            results_list.append(search_result_obj)

        final_output = StudentSearchResults(
                    search_results=results_list,
                    k=k)

        filename = os.path.basename(dataset_path)
        full_save_path = os.path.join(save_directory, filename)

        # Ensure the output directory exists so it doesn't crash
        os.makedirs(save_directory, exist_ok=True)

        with open(full_save_path, 'w', encoding='utf-8') as f:
            # .model_dump_json() i= instantly makes a JSON string
            f.write(final_output.model_dump_json(indent=4))

        print(f"Search dataset complete! Saved to {full_save_path}")

    def answer_dataset(
        self,
        student_search_results_path: str,
        save_directory: str
    ) -> None:
        """
        Reads search results, generates an answer for each
        and saves the final output.
        """

        print(f"Generating answers for: {student_search_results_path}")

        try:
            with open(student_search_results_path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
        except FileNotFoundError:
            print(f"Error: file not found: {student_search_results_path}")
            return
        except json.JSONDecodeError as e:
            print(f"Error: invalid JSON in {student_search_results_path}: "
                  f"{e}")
            return

        if self.generator is None:
            raise RuntimeError(
                "generator is None, cannot run answer_dataset.")

        search_data = StudentSearchResults(**raw_data)
        answers_list = []

        for result in tqdm(search_data.search_results,
                           desc="Generating Answers"):

            answer_obj = self.generator.generate_answer(
                question_id=result.question_id,
                query=result.question_str,
                retrieved_sources=result.retrieved_sources
            )
            answers_list.append(answer_obj)

        final_output = StudentSearchResultsAndAnswer(
            search_results=answers_list,
            k=search_data.k
        )

        filename = os.path.basename(student_search_results_path)
        full_save_path = os.path.join(save_directory, filename)
        os.makedirs(save_directory, exist_ok=True)

        with open(full_save_path, 'w', encoding='utf-8') as f:
            f.write(final_output.model_dump_json(indent=4))

        print(f"Answer dataset complete! Saved to {full_save_path}")
