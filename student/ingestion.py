import os
from typing import List
import logging
from tqdm import tqdm
from student.models import MinimalSource
from student.chunker import CodeChunker, TextChunker

# Configure logger
logging.basicConfig(filename='ingestion_errors.log',
                    level=logging.WARNING,
                    format='%(asctime)s - %(levelname)s -%(message)s')


class IngestionEngine:
    def __init__(self, max_chunk_size: int = 2000):
        self.code_chunker = CodeChunker(max_chunk_size)
        self.text_chunker = TextChunker(max_chunk_size)

    def ingest_directory(self, root_dir: str) -> List[MinimalSource]:
        all_chunks: List[MinimalSource] = []

        for dirpath, _, filenames in os.walk(root_dir):
            for filename in filenames:
                file_path = os.path.join(dirpath, filename)

                # We need to try/except the read because some files
                # (like images or binaries) will crash .read_text()
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()

                    if filename.endswith('.py'):
                        all_chunks.extend(
                            self.code_chunker.chunk(file_path, content)
                            )
                    elif filename.endswith(('.md', '.txt')):
                        all_chunks.extend(
                            self.text_chunker.chunk(file_path, content)
                            )

                except Exception as e:
                    logging.warning(f"Failed to ingest: {file_path}: {e}")
                    continue

        return all_chunks

    def ingest_docs(self, root_dir: str) -> List[MinimalSource]:
        """Ingests only .md and .txt files."""
        all_chunks: List[MinimalSource] = []

        doc_files = [
            os.path.join(dirpath, filename)
            for dirpath, _, filenames in os.walk(root_dir)
            for filename in filenames
            if filename.endswith(('.md', '.txt'))
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
