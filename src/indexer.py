import os
import re
import bm25s
import json
import torch
from typing import List, Dict, Optional
from collections import defaultdict
from src.models import MinimalSource
from src.config import (
    EMBEDDING_MODEL,
    EMBEDDING_BATCH_SIZE,
    CANDIDATE_MULT,
    RRF_CONSTANT,
    DEFAULT_K
)
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


class Indexer:
    """Hybrid BM25 + semantic index: build, save, load, search."""

    def __init__(self) -> None:
        """Loads the sentence-transformer model on GPU."""
        self.corpus_chunks: List[MinimalSource] = []
        self.is_code: bool = False

        self.bm25_retriever = bm25s.BM25()

        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading Embedding Model on device: {self.device}")

        self.embedding_model: SentenceTransformer = SentenceTransformer(
            EMBEDDING_MODEL, device=self.device
        )
        self.faiss_index: Optional[faiss.IndexFlatIP] = None

    def _make_corpus(self, chunks: List[MinimalSource]) -> List[str]:
        """
        converts coordiante-based chunks into readable strings
        open the physical files, slice the raw text string with the coords
        of the minimal source object and return a list of string
        """
        corpus: List[str] = []

        # make a dict of {file_path : [Minimalsource.... , ....]}
        # resolves massive bottleneck of opening a file once and not 500 times
        dict_chunks: Dict[str, List[MinimalSource]] = defaultdict(list)
        for chunk in chunks:
            dict_chunks[chunk.file_path].append(chunk)

        for each_path, file_chunks in dict_chunks.items():
            try:
                # open the whole file once
                with open(each_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                filename = os.path.relpath(each_path)
                for chunk in file_chunks:
                    chunk_text = content[
                        chunk.first_character_index:
                        chunk.last_character_index
                    ]
                    if self.is_code:
                        path_prefix = ' '.join(
                            self._extract_path_tokens(filename) * 3
                        )  # we want the filename to be 3 times more likely
                        corpus.append(f"{path_prefix}\n{chunk_text}")
                    else:
                        stem = os.path.splitext(os.path.basename(filename))[0]
                        stem_tokens = [t for t in re.split(r'[_\-]', stem)
                                       if len(t) > 1]
                        stem_boost = (' '.join(stem_tokens * 2) + '\n'
                                      ) if stem_tokens else ''
                        # important so filename dosent forget its title
                        heading = self._find_nearest_heading(
                            content, chunk.first_character_index
                        )
                        header_boost = (f"{heading}\n{heading}\n"
                                        if heading else "")
                        table_header = self._find_table_header(
                            content, chunk.first_character_index, chunk_text
                        )
                        table_boost = (f"{table_header}\n"
                                       if table_header else "")
                        corpus.append(
                            f"{stem_boost}{filename}\n"
                            f"{header_boost}{table_boost}{chunk_text}"
                        )

            except Exception:
                print(
                    f"Warning: Could not read {each_path} for creation."
                )

        return corpus

    def _clean_text(self, text: str) -> str:
        """ Normalize the syntax for better BM25 matching """
        text = re.sub(r'(?<=[a-z0-9])(?=[A-Z])', ' ', text)
        cleaned = (text.replace("_", " ").replace(".", " ")
                   .replace("-", " ").replace("/", " "))
        return cleaned.lower()

    def _clean_code_text(self, text: str) -> str:
        """
        Extract only identifiers from code for cleaner BM25 token signal.
        1. ignore code syntax :
        def my_function(x=5): becomes a list: ['def', 'my_function', 'x']
        2. camelCase splitter:
        BlockSpaceManager becomes ['Block', 'Space', 'Manager']
        3. snake_case splitter:
        get_num_free_blocks becomes ['get', 'num', 'free', 'blocks']
        3. we remove single letters like "i", "j"... to remove noise
        4. the we standarize it
        """
        tokens = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*', text)
        result = []
        for token in tokens:
            # camelCase split BEFORE lowercasing
            parts = re.sub(r'(?<=[a-z0-9])(?=[A-Z])', ' ', token).split()
            for part in parts:
                for subpart in part.split('_'):
                    if len(subpart) > 1:
                        result.append(subpart.lower())
        return ' '.join(result)

    @staticmethod
    def _extract_path_tokens(file_path: str) -> List[str]:
        """Split a file path into lowercase BM25-friendly tokens."""
        return [p.lower() for p in re.split(r'[/_\-\.]', file_path)
                if len(p) > 1]

    @staticmethod
    def _preprocess_markdown(text: str) -> str:
        """
        Strip URL and HTML noise from markdown before BM25 tokenization.
        """
        text = re.sub(r'\[([^\]]*)\]\([^\)]*\)', r'\1', text)
        text = re.sub(r'https?://\S+', ' ', text)               # bare URLs
        text = re.sub(r'<[^>]+>', ' ', text)                    # HTML tags
        return text

    @staticmethod
    def _find_nearest_heading(content: str, char_idx: int) -> str:
        """
        Scan backward from char_idx and
        return the nearest markdown heading text.
        helps giving context to docs
        """
        for line in reversed(content[:char_idx].split('\n')):
            stripped = line.strip()
            if stripped.startswith('#'):
                return stripped.lstrip('#').strip()
        return ''

    @staticmethod
    def _find_table_header(content: str,
                           chunk_start: int,
                           chunk_text: str) -> str:
        """
        If chunk contains table rows without a header,
        return the column header text.
        """
        if '|' not in chunk_text:
            return ''
        # headers are present in this chunk
        if re.search(r'^\s*\|[\s\-:|]+\|', chunk_text, re.MULTILINE):
            return ''
        # Scan backward up to 100 lines for the nearest table separator row
        lines = content[:chunk_start].split('\n')
        for i in range(len(lines) - 1, max(-1, len(lines) - 100), -1):
            if re.match(r'\s*\|[\s\-:|]+\|', lines[i]):
                if i > 0 and '|' in lines[i - 1]:
                    cols = [c.strip() for c in lines[i - 1].split('|')
                            if c.strip() and not re.match(
                                r'^[-:]+$', c.strip())]
                    if cols:
                        return ' '.join(cols)
                break
        return ''

    def build_index(self, chunks: List[MinimalSource], is_code: bool) -> None:
        """
        Extracts text, tokenizes/embeds it, and builds the index.
        """
        self.corpus_chunks = chunks
        self.is_code = is_code

        print("Extracting corpus from chunks...")
        raw_corp: List[str] = self._make_corpus(chunks)

        print("Building BM25 Index...")
        if is_code:
            corpus: List[str] = [self._clean_code_text(text)
                                 for text in raw_corp]
        else:
            corpus = [self._clean_text(self._preprocess_markdown(text))
                      for text in raw_corp]
        b = 0.3 if is_code else 0.75
        self.bm25_retriever = bm25s.BM25(method="bm25+", b=b)
        corpus_tokens = bm25s.tokenize(corpus, stopwords="en")
        self.bm25_retriever.index(corpus_tokens)

        print(
            f"Encoding {len(raw_corp)} chunks into dense vectors on {self.device}..."
            " (This takes a moment)"
        )
        embeddings = self.embedding_model.encode(
            raw_corp,
            batch_size=EMBEDDING_BATCH_SIZE,
            show_progress_bar=True,
            normalize_embeddings=True)
        dimension = embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dimension)
        self.faiss_index.add(np.array(embeddings).astype('float32'))

        print("Indexing Complete!")

    def save_index(self, save_dir: str) -> None:
        """Saves the index engines and chunk metadata to disk."""
        print(f"Saving index to {save_dir}...")
        os.makedirs(save_dir, exist_ok=True)

        stand_chunks = [chunk.model_dump() for chunk in self.corpus_chunks]
        with open(os.path.join(save_dir, 'chunks.json'), 'w') as file:
            json.dump(stand_chunks, file)

        self.bm25_retriever.save(save_dir)
        if self.faiss_index is not None:
            faiss.write_index(
                self.faiss_index,
                os.path.join(save_dir, "faiss.index")
            )
        print("Save complete!")

    def load_index(self, load_dir: str, is_code: bool) -> None:
        """Loads the index engines and chunk metadata from disk."""
        self.is_code = is_code
        chunks_path = os.path.join(load_dir, "chunks.json")
        with open(chunks_path, 'r', encoding='utf-8') as f:
            raw_chunks = json.load(f)
        self.corpus_chunks = [
            MinimalSource(**chunk_dict) for chunk_dict in raw_chunks
        ]

        self.bm25_retriever = bm25s.BM25.load(load_dir, load_corpus=False)
        self.faiss_index = faiss.read_index(
            os.path.join(load_dir, "faiss.index")
        )

        print("Load complete!")

    def search(self, query: str, k: int = DEFAULT_K) -> List[MinimalSource]:
        """
        Searches the index and returns the top-k chunks.
        Fuses BM25 and semantic rankings via Reciprocal Rank Fusion.
        """
        if k <= 0:
            return []
        if not query or not query.strip():
            return []

        if self.is_code:
            clean_q = self._clean_code_text(query)
        else:
            clean_q = self._clean_text(query)

        token_q = bm25s.tokenize(clean_q, stopwords="en")
        docs, _ = self.bm25_retriever.retrieve(token_q, k=k * CANDIDATE_MULT)
        bm25_results = [self.corpus_chunks[ticket] for ticket in docs[0]]

        if self.faiss_index is None:
            raise RuntimeError(
                "FAISS index not loaded. "
                "Call build_index or load_index first."
            )
        query_vector = self.embedding_model.encode(
            [query], normalize_embeddings=True
        )
        distances, indices = self.faiss_index.search(
            np.array(query_vector).astype('float32'), k * CANDIDATE_MULT)
        semantic_results = [
            self.corpus_chunks[idx] for idx in indices[0] if idx != -1
        ]

        # Reciprocal Rank Fusion (RRF)
        def chunk_id(c: MinimalSource) -> str:
            """create a unique ID for every chunk so i can track them"""
            return f"{c.file_path}_{c.first_character_index}"

        rrf_scores: Dict[str, float] = defaultdict(float)
        chunk_lookup: Dict[str, MinimalSource] = {}
        # RRF algo.
        for rank, chunk in enumerate(bm25_results):
            cid = chunk_id(chunk)
            chunk_lookup[cid] = chunk
            rrf_scores[cid] += 1.0 / (RRF_CONSTANT + rank + 1)

        for rank, chunk in enumerate(semantic_results):
            cid = chunk_id(chunk)
            chunk_lookup[cid] = chunk
            rrf_scores[cid] += 1.0 / (RRF_CONSTANT + rank + 1)
        # if chunk was found in both then its at the top
        sorted_ids = sorted(
            rrf_scores, key=lambda x: rrf_scores[x], reverse=True
        )
        return [chunk_lookup[cid] for cid in sorted_ids[:k]]
