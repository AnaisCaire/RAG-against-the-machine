import os
import bm25s
import json
from typing import List, Dict, Optional
from collections import defaultdict
from student.models import MinimalSource
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


class Indexer:
    def __init__(self, mode: str = "semantic"):
        """
        Modes:
            1. bm25
            2. Semantic
        """
        self.mode = mode
        self.corpus_chunks: List[MinimalSource] = []
        self.is_code: bool = False

        # ---- BM25 ----
        self.bm25_retriever = bm25s.BM25()

        # ---- SEMANTIC EMBEDDING ----
        self.embedding_model: Optional[SentenceTransformer] = None
        self.faiss_index: Optional[faiss.IndexFlatIP] = None

        if self.mode == "semantic":
            print("Loading Semantic Embedding Model")
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

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

                # 3. Extract the text for every chunk in this file
                for chunk in file_chunks:
                    chunk_text = content[chunk.first_character_index: chunk.last_character_index]
                    corpus.append(chunk_text)
                    pass

            except Exception:
                print(f"Warning: Could not read {each_path} for corpus creation.")

        return corpus

    def _clean_text(self, text: str) -> str:
        """ Normalize the syntax for better BM25 matching"""
        import re
        text = re.sub(r'(?<=[a-z0-9])(?=[A-Z])', ' ', text)
        cleaned = text.replace("_", " ").replace(".", " ")
        return cleaned.lower()

    def build_index(self,
                    chunks: List[MinimalSource],
                    is_code: bool,
                    mode: str = "semantic") -> None:
        """
        Extracts text, tokenizes it, and builds the BM25 index.
        """
        self.corpus_chunks = chunks
        self.is_code = is_code

        print("Extracting corpus from chunks...")
        raw_corp: List[str] = self._make_corpus(chunks)

        if self.mode == "bm25":
            print("Building BM25 Index:")
            corpus: List[str] = [self._clean_text(text) for text in raw_corp]
            print("Tokenizing corpus...")
            if is_code:
                corpus_tokens = bm25s.tokenize(corpus, stopwords=[])
            else:
                corpus_tokens = bm25s.tokenize(corpus, stopwords="en")
            print("Training BM25 Index...")
            self.bm25_retriever.index(corpus_tokens)
        elif self.mode == "semantic":
            print(f"Encoding {len(raw_corp)} chunks into dense vectors...",
                  " (This will take a moment)")
            # 1 make text into Vectors
            embeddings = self.embedding_model.encode(
                raw_corp,
                show_progress_bar=True,
                normalize_embeddings=True)
            # 2 activate the FAISS Vector database
            dimension = embeddings.shape[1]  # remeber: number of columns
            # cosine similarity = Inner Product
            self.faiss_index = faiss.IndexFlatIP(dimension)
            # add to the vectors database:
            self.faiss_index.add(np.array(embeddings).astype('float32'))

        print("Indexing Complete!")

    def save_index(self, save_dir: str) -> None:
        """
        saves the math model and the chunk metadata to folder
        """
        print(f"Saving index to {save_dir}...")
        os.makedirs(save_dir, exist_ok=True)

        # convert pydantic to standard dict
        stand_chunks = [chunk.model_dump() for chunk in self.corpus_chunks]

        with open(os.path.join(save_dir, 'chunks.json'), 'w') as file:
            json.dump(stand_chunks, file)

        # save Math Engine
        if self.mode == "bm25":
            self.bm25_retriever.save(save_dir)
        elif self.mode == "semantic":
            faiss.write_index(self.faiss_index, os.path.join(save_dir, "faiss.index"))
        print("Save complete!")

    def load_index(self, load_dir: str, is_code: bool) -> None:
        """
        Loads the models and the chunk metadata from disk.
        opposite of save_index
        """
        # 1. Rehydrate the chunks
        chunks_path = os.path.join(load_dir, "chunks.json")
        with open(chunks_path, 'r', encoding='utf-8') as f:
            raw_chunks = json.load(f)
        self.corpus_chunks = [
            MinimalSource(**chunk_dict) for chunk_dict in raw_chunks
            ]

        # 2. Load the Engine
        if self.mode == "bm25":
            self.bm25_retriever = bm25s.BM25.load(load_dir, load_corpus=False)
        elif self.mode == "semantic":
            if self.embedding_model is None:
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            self.faiss_index = faiss.read_index(
                os.path.join(load_dir, "faiss.index")
                )

        print("Load complete!")

        print("Load complete!")

    def search(self, query: str, k: int = 5) -> List[MinimalSource]:
        """
        - Query Encoding
        - similarity search: Term Frequency-Inverse Document Frequency
            Ranking: the top k chunks
        - stop-word = remove noisy sounds
        """
        results: List[MinimalSource] = []

        if self.mode == "bm25":
            clean_q = self._clean_text(query)
            stopwords = [] if self.is_code else "en"
            token_q = bm25s.tokenize(clean_q, stopwords=stopwords)
            docs, _ = self.bm25_retriever.retrieve(token_q, k=k)
            for ticket in docs[0]:
                results.append(self.corpus_chunks[ticket])

        elif self.mode == "semantic":
            # 1. Convert the search query into a dense vector
            query_vector = self.embedding_model.encode(
                [query], normalize_embeddings=True)

            # 2. Search FAISS for the mathematically closest vectors
            distances, indices = self.faiss_index.search(np.array(query_vector).astype('float32'), k)

            # 3. Retrieve the chunks using the returned indices
            for index in indices[0]:
                if index != -1:  # FAISS returns -1 if there aren't enough chunks
                    results.append(self.corpus_chunks[index])

        return results
