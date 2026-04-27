
CODE_DIRS = src

.PHONY: install debug run clean lint lint-strict pipeline

install:
	uv sync

run:
	 uv run python -m src $(CMD)

clean:
	rm -rf .mypy_cache .pytest_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -f ingestion_errors.log
	rm -rf data/processes/*/
	rm -f data/output/search_results/*
	rm -f data/output/search_results_and_answer/*
	@echo "--- Cleanup Complete ---"

debug:
	PYTHONPATH=. uv run python -m pdb -c continue -m src $(CMD)

lint:
	@echo "--- Running Flake8 ---"
	uv run flake8 $(CODE_DIRS)
	@echo "--- Running Mypy ---"
	PYTHONPATH=. uv run mypy $(CODE_DIRS) \
		--ignore-missing-imports \
		--explicit-package-bases \
		--disallow-untyped-defs \
		--check-untyped-defs \
		--warn-return-any \
		--warn-unused-ignores

lint-strict:
	@echo "--- Running Strict Linting ---"
	uv run flake8
	PYTHONPATH=. uv run mypy $(CODE_DIRS) --strict

pipeline:
	@echo "=== [1/6] Indexing ==="
	uv run python -m src index
	@echo "=== [2/6] Search: docs ==="
	uv run python -m src search_dataset \
		--dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
		--save_directory data/output/search_results --k 10
	@echo "=== [3/6] Search: code ==="
	uv run python -m src search_dataset \
		--dataset_path data/datasets/UnansweredQuestions/dataset_code_public.json \
		--save_directory data/output/search_results --k 10
	@echo "=== [4/6] Answer: docs ==="
	uv run python -m src answer_dataset \
		--student_search_results_path data/output/search_results/dataset_docs_public.json \
		--save_directory data/output/search_results_and_answer
	@echo "=== [5/6] Answer: code ==="
	uv run python -m src answer_dataset \
		--student_search_results_path data/output/search_results/dataset_code_public.json \
		--save_directory data/output/search_results_and_answer
	@echo "=== [6/6] Evaluate ==="
	uv run python -m src evaluate \
		--student_results_path data/output/search_results/dataset_docs_public.json \
		--dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json --k 10
	uv run python -m src evaluate \
		--student_results_path data/output/search_results/dataset_code_public.json \
		--dataset_path data/datasets/AnsweredQuestions/dataset_code_public.json --k 10
	@echo "=== Pipeline complete ==="
