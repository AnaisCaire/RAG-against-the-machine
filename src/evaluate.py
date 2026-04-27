import json


class Evaluator:
    """
    Handles the mathematical evaluation of student search
    results against ground truth data.
    """

    def evaluate(self, student_results_path: str, dataset_path: str,
                 k: int = 10, max_context_length: int = 2000) -> None:
        """
        Evaluates student search results against the ground
        truth using Recall@k.
        """
        print(f"--- Running Custom Recall@{k} Evaluation ---")

        # 1. Load the JSON data
        try:
            with open(student_results_path, 'r', encoding='utf-8') as f:
                student_data = json.load(f)
            with open(dataset_path, 'r', encoding='utf-8') as f:
                ground_truth_data = json.load(f)
        except FileNotFoundError as e:
            print(f"❌ Error loading files: {e}")
            return

        # 2. Map ground truth sources by question_id for quick lookup
        truth_map = {q['question_id']: q['sources'] for q in ground_truth_data['rag_questions']}

        total_recall = 0.0
        total_precision = 0.0
        num_questions = len(student_data['search_results'])

        # 3. Calculate Recall@k and Precision@k for every question
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
                is_found = False

                for pred in pred_sources:
                    # Must be the exact same file
                    if exp['file_path'] == pred['file_path']:

                        # 1. Calculate Intersection (Overlap)
                        overlap_start = max(exp['first_character_index'], pred['first_character_index'])
                        overlap_end = min(exp['last_character_index'], pred['last_character_index'])
                        overlap_len = max(0, overlap_end - overlap_start)

                        # 2. Calculate expected source length
                        exp_len = exp['last_character_index'] - exp['first_character_index']

                        # 3. Check against the 5% overlap threshold
                        # (overlap relative to expected source length, as per subject)
                        overlap_ratio = overlap_len / exp_len if exp_len > 0 else 0.0
                        if overlap_ratio >= 0.05:
                            is_found = True
                            break

                if is_found:
                    found_count += 1

            # Question Recall = (Sources Found / Total Expected Sources)
            question_recall = found_count / len(expected_sources)
            total_recall += question_recall

            # Precision@k: how many of the retrieved sources are relevant
            relevant_retrieved = 0
            for pred in pred_sources:
                for exp in expected_sources:
                    if exp['file_path'] == pred['file_path']:
                        overlap_start = max(exp['first_character_index'], pred['first_character_index'])
                        overlap_end = min(exp['last_character_index'], pred['last_character_index'])
                        overlap_len = max(0, overlap_end - overlap_start)
                        exp_len = exp['last_character_index'] - exp['first_character_index']
                        overlap_ratio = overlap_len / exp_len if exp_len > 0 else 0.0
                        if overlap_ratio >= 0.05:
                            relevant_retrieved += 1
                            break

            question_precision = relevant_retrieved / k if k > 0 else 0.0
            total_precision += question_precision

        # 4. Final System Score
        final_recall = total_recall / num_questions
        final_precision = total_precision / num_questions

        print(f"Questions evaluated: {num_questions}")
        print(f"Recall@{k}: {final_recall:.3f}")
        print(f"Precision@{k}: {final_precision:.3f}")

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
