import os
from typing import List
import logging
from tqdm import tqdm
from src.models import MinimalSource
from src.chunker import CodeChunker, TextChunker

# Configure logger
logging.basicConfig(filename='ingestion_errors.log',
                    level=logging.WARNING,
                    format='%(asctime)s - %(levelname)s -%(message)s')


class IngestionEngine:
    """Walks a directory tree and produces MinimalSource chunks for docs and code files."""

    def __init__(self, max_chunk_size: int = 2000) -> None:
        """Initializes code and text chunkers with the given max chunk size."""
        self.code_chunker = CodeChunker(max_chunk_size)
        self.text_chunker = TextChunker(max_chunk_size)

    def ingest_docs(self, root_dir: str) -> List[MinimalSource]:
        """Ingests only .md and .txt files."""
        all_chunks: List[MinimalSource] = []

        # FIX 2: exclude noise directories that contain no ground-truth sources
        # but pollute the index (sonnets, pip requirements, test prompts, CI
        # scripts). These were 315 of 1199 chunks (~26% noise) and caused ~10%
        # of queries to have a noise file in their top-5 results.
        _EXCLUDE_DIRS = {
            'benchmarks', 'examples', 'tests', '.buildkite', 'requirements',
        }

        doc_files = [
            os.path.join(dirpath, filename)
            for dirpath, dirnames, filenames in os.walk(root_dir)
            for filename in filenames
            if (
                filename.endswith(('.md', '.txt'))
                or (filename.endswith('.py') and dirpath == root_dir)
            )
            and not any(
                part in _EXCLUDE_DIRS
                for part in dirpath.split(os.sep)
            )
        ]

        for file_path in tqdm(doc_files, desc="Ingesting docs"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                all_chunks.extend(
                    self.text_chunker.chunk(file_path, content)
                )
            except Exception as e:
                logging.warning(f"Failed to ingest: {file_path}: {e}")

        return all_chunks

    def ingest_code(self, root_dir: str) -> List[MinimalSource]:
        """Ingests only .py files."""
        all_chunks: List[MinimalSource] = []

        py_files = [
            os.path.join(dirpath, filename)
            for dirpath, _, filenames in os.walk(root_dir)
            for filename in filenames
            if filename.endswith('.py')
        ]

        for file_path in tqdm(py_files, desc="Ingesting code"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                all_chunks.extend(
                    self.code_chunker.chunk(file_path, content)
                )
            except Exception as e:
                logging.warning(f"Failed to ingest: {file_path}: {e}")

        return all_chunks
