import os
import re
import bm25s
import json
from typing import List, Dict, Optional
from collections import defaultdict
from src.models import MinimalSource
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


class Indexer:
    """Hybrid BM25 + semantic retrieval index supporting build, save, load, and search."""

    def __init__(self) -> None:
        """Loads the sentence-transformer embedding model and initialises BM25."""
        self.corpus_chunks: List[MinimalSource] = []
        self.is_code: bool = False

        self.bm25_retriever = bm25s.BM25()

        print("Loading Semantic Embedding Model")
        self.embedding_model: SentenceTransformer = SentenceTransformer('all-MiniLM-L6-v2')
        self.faiss_index: Optional[faiss.IndexFlatIP] = None

    def _make_corpus(self, chunks: List[MinimalSource]) -> List[str]:
        """converts coordiante-based chunks into readable strings"""
        corpus: List[str] = []

        # make a dict of {file_path : [Minimalsource.... , ....]}
        dict_chunks: Dict[str, List[MinimalSource]] = defaultdict(list)
        for chunk in chunks:
            dict_chunks[chunk.file_path].append(chunk)

        for each_path, file_chunks in dict_chunks.items():
            try:
                with open(each_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                filename = os.path.basename(each_path)
                for chunk in file_chunks:
                    chunk_text = content[chunk.first_character_index: chunk.last_character_index]
                    # Prepend the filename so BM25/FAISS can discriminate by file
                    # when many files share boilerplate code patterns.
                    corpus.append(f"{filename}\n{chunk_text}")

            except Exception:
                print(f"Warning: Could not read {each_path} for corpus creation.")

        return corpus

    def _clean_text(self, text: str) -> str:
        """ Normalize the syntax for better BM25 matching"""
        text = re.sub(r'(?<=[a-z0-9])(?=[A-Z])', ' ', text)
        cleaned = text.replace("_", " ").replace(".", " ")
        return cleaned.lower()

    def build_index(self, chunks: List[MinimalSource], is_code: bool) -> None:
        """
        Extracts text, tokenizes/embeds it, and builds the index.
        """
        self.corpus_chunks = chunks
        self.is_code = is_code

        print("Extracting corpus from chunks...")
        raw_corp: List[str] = self._make_corpus(chunks)

        print("Building BM25 Index...")
        corpus: List[str] = [self._clean_text(text) for text in raw_corp]
        stopwords = [] if is_code else "en"
        corpus_tokens = bm25s.tokenize(corpus, stopwords=stopwords)
        self.bm25_retriever.index(corpus_tokens)

        print(f"Encoding {len(raw_corp)} chunks into dense vectors... (This takes a moment)")
        embeddings = self.embedding_model.encode(
            raw_corp,
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
            faiss.write_index(self.faiss_index, os.path.join(save_dir, "faiss.index"))
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
        self.faiss_index = faiss.read_index(os.path.join(load_dir, "faiss.index"))

        print("Load complete!")

    def search(self, query: str, k: int = 5) -> List[MinimalSource]:
        """
        Searches the index and returns the top-k chunks.
        Fuses BM25 and semantic rankings via Reciprocal Rank Fusion.
        """
        if k <= 0:
            return []
        if not query or not query.strip():
            return []

        CANDIDATE_MULT = 20

        clean_q = self._clean_text(query)
        stopwords = [] if self.is_code else "en"
        token_q = bm25s.tokenize(clean_q, stopwords=stopwords)
        docs, _ = self.bm25_retriever.retrieve(token_q, k=k * CANDIDATE_MULT)
        bm25_results = [self.corpus_chunks[ticket] for ticket in docs[0]]

        if self.faiss_index is None:
            raise RuntimeError("FAISS index not loaded. Call build_index or load_index first.")
        query_vector = self.embedding_model.encode([query], normalize_embeddings=True)
        distances, indices = self.faiss_index.search(
            np.array(query_vector).astype('float32'), k * CANDIDATE_MULT)
        semantic_results = [
            self.corpus_chunks[idx] for idx in indices[0] if idx != -1
        ]

        # Reciprocal Rank Fusion (RRF)
        def chunk_id(c: MinimalSource) -> str:
            return f"{c.file_path}_{c.first_character_index}"

        rrf_scores: Dict[str, float] = defaultdict(float)
        chunk_lookup: Dict[str, MinimalSource] = {}

        for rank, chunk in enumerate(bm25_results):
            cid = chunk_id(chunk)
            chunk_lookup[cid] = chunk
            rrf_scores[cid] += 1.0 / (60 + rank + 1)

        for rank, chunk in enumerate(semantic_results):
            cid = chunk_id(chunk)
            chunk_lookup[cid] = chunk
            rrf_scores[cid] += 1.0 / (60 + rank + 1)

        sorted_ids = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)
        return [chunk_lookup[cid] for cid in sorted_ids[:k]]
