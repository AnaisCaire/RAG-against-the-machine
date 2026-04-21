import os
import json
from student.models import RagDataset, StudentSearchResults, StudentSearchResultsAndAnswer, MinimalSearchResults
from student.indexer import Indexer
from student.generator import Generator


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
        print(f"Generating answers for: {student_search_results_path}")
        
        # 1. Open the JSON file at student_search_results_path
        # TODO: raw_data = ...
        
        # 2. Parse it into your StudentSearchResults Pydantic model
        # TODO: search_data = StudentSearchResults(**raw_data)
        
        # 3. Create an empty list to hold your MinimalAnswer objects
        # TODO: answers_list = []
        
        # 4. Loop through search_data.search_results
        # TODO: for result in search_data.search_results:
            # a. Pass result.question_id, result.question, and result.retrieved_sources 
            #    into self.generator.generate_answer(...)
            # b. Append the returned MinimalAnswer to answers_list
            
        # 5. Package answers_list and search_data.k into a StudentSearchResultsAndAnswer object
        # TODO: final_output = StudentSearchResultsAndAnswer(...)
        
        # 6. Save to disk exactly like Step 6 above.
        # TODO: Save final_output.model_dump() to the save_directory.
        
        print("Answer dataset complete!")