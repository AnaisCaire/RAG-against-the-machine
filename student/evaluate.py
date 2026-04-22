    



def evaluate_answer(self, student_results_path: str, dataset_path: str, k: int = 10, max_context_length: int = 2000):
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