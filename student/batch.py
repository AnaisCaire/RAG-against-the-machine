import os
import json
from student.models import RagDataset, StudentSearchResults, StudentSearchResultsAndAnswer, MinimalSearchResults
from student.indexer import Indexer
from student.generator import Generator
from tqdm import tqdm


class BatchProcessor:
    def __init__(self, search_engine: Indexer, generator=None):
        self.search_engine = search_engine
        self.generator = generator

    def search_dataset(self, dataset_path: str, save_directory: str, k: int = 5) -> None:
        """
        Reads a dataset, searches the index for each question, and saves the results.
        """
        print(f"Processing dataset: {dataset_path}")

        # 1. Open the JSON file at dataset_path and load it into a dictionary
        with open(dataset_path, 'r') as f:
            raw_data = json.load(f)

        dataset = RagDataset(**raw_data)

        # 3. Create an empty list to hold your MinimalSearchResults
        results_list = []

        for q in dataset.rag_questions:
            question_id = q.question_id
            question = q.question
            found_sources = self.search_engine.search(query=question, k=k)
            search_result_obj = MinimalSearchResults(
                question_id=question_id,
                question=question,
                retrieved_sources=found_sources)
            results_list.append(search_result_obj)

        # 5. Package the final results_list and 'k'
        # into a StudentSearchResults object
        final_output = StudentSearchResults(
                    search_results=results_list,
                    k=k)
        # 6. Dynamically build the save path
        filename = os.path.basename(dataset_path)
        full_save_path = os.path.join(save_directory, filename)

        # Ensure the output directory exists so it doesn't crash
        os.makedirs(save_directory, exist_ok=True)

        # Save the Pydantic object to disk as a JSON file
        with open(full_save_path, 'w', encoding='utf-8') as f:
            # .model_dump_json() is a Pydantic magic method that instantly makes a JSON string!
            f.write(final_output.model_dump_json(indent=4))

        print(f"Search dataset complete! Saved to {full_save_path}")

    def answer_dataset(self, student_search_results_path: str, save_directory: str) -> None:
        """
        Reads search results, generates an answer for each, and saves the final output.
        """
        import json
        import os

        print(f"Generating answers for: {student_search_results_path}")

        # 1. Open the JSON file at student_search_results_path
        with open(student_search_results_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)

        # 2. Parse it into your StudentSearchResults Pydantic model
        search_data = StudentSearchResults(**raw_data)

        # 3. Create an empty list to hold your MinimalAnswer objects
        answers_list = []

        # 4. Loop through search_data.search_results and generate answers
        for result in tqdm(search_data.search_results, desc="Generating Answers"):
            # Generate the answer using your AI pipeline
            answer_obj = self.generator.generate_answer(
                question_id=result.question_id,
                query=result.question,
                retrieved_sources=result.retrieved_sources
            )
            answers_list.append(answer_obj)

        # 5. Package answers_list and search_data.k into a StudentSearchResultsAndAnswer object
        final_output = StudentSearchResultsAndAnswer(
            search_results=answers_list,
            k=search_data.k
        )

        # 6. Dynamically build the save path and save to disk
        filename = os.path.basename(student_search_results_path)
        full_save_path = os.path.join(save_directory, filename)

        # Force your OS to create the folder if it doesn't exist
        os.makedirs(save_directory, exist_ok=True)

        # Save the Pydantic object to disk as a nicely indented JSON file
        with open(full_save_path, 'w', encoding='utf-8') as f:
            f.write(final_output.model_dump_json(indent=4))

        print(f"Answer dataset complete! Saved to {full_save_path}")
