
CODE_DIRS = student

.PHONY: install debug run lint lint-strict

install:
	uv sync

run:
	 uv run python -m student $(CMD)

clean:
	rm -rf .mypy_cache .pytest_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -f ingestion_errors.log
	rm -rf data/processes/*/
	rm -f data/output/search_results/*
	rm -f data/output/search_results_and_answer/*
	@echo "--- Cleanup Complete ---"

debug:
	PYTHONPATH=. uv run python -m pdb -c continue -m student $(CMD)

lint:
	@echo "--- Running Flake8 ---"
	uv run flake8 $(CODE_DIRS)
	@echo "--- Running Mypy ---"
	PYTHONPATH=. uv run mypy $(CODE_DIRS) \
		--ignore-missing-imports \
		--disallow-untyped-defs \
		--check-untyped-defs \
		--warn-return-any \
		--warn-unused-ignores

lint-strict:
	@echo "--- Running Strict Linting ---"
	uv run flake8
	PYTHONPATH=. uv run mypy $(CODE_DIRS) --strict
