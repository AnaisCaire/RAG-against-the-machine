import os
import bm25s
import json
from typing import List, Dict
from collections import defaultdict
from student.models import MinimalSource


class Indexer:
    def __init__(self):
        self.corpus_chunks: List[MinimalSource] = []
        self.is_code: bool = False
        # Analogy: spawning the Librarian
        self.retriever = bm25s.BM25()

    def _make_corpus(self, chunks: List[MinimalSource]) -> List[str]:
        """converts coordiante-based chunks into readable strings"""
        corpus: List[str] = []

        # make a dict of {file_path : [Minimalsource.... , ....]}
        dict_chunks: Dict[str, List[MinimalSource]] = defaultdict(list)
        for chunk in chunks:
            dict_chunks[chunk.file_path].append(chunk)

        # read only once each file
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
                    is_code: bool) -> None:
        """
        Extracts text, tokenizes it, and builds the BM25 index.
        """
        self.corpus_chunks = chunks
        self.is_code = is_code

        print("Extracting corpus from chunks...")
        raw_corp: List[str] = self._make_corpus(chunks)
        corpus: List[str] = [self._clean_text(text) for text in raw_corp]
        print("Tokenizing corpus...")
        if is_code:
            corpus_tokens = bm25s.tokenize(corpus, stopwords=[])
        else:
            corpus_tokens = bm25s.tokenize(corpus, stopwords="en")
        # corpus_tokens is a tuple of (id, token)
        print("Training BM25 Index...")

        # make the libririan read it all
        self.retriever.index(corpus_tokens)

        print("Indexing Complete!")

    def save_index(self, save_dir: str) -> None:
        """
        save the 2 pieces of data
        """
        print(f"Saving index to {save_dir}...")
        self.retriever.save(save_dir)

        # convert pydantic to standard dict
        stand_chunks = [chunk.model_dump() for chunk in self.corpus_chunks]

        with open(os.path.join(save_dir, 'chunks.json'), 'w') as file:
            json.dump(stand_chunks, file)
        print("Save complete!")

    def load_index(self, load_dir: str, is_code: bool) -> None:
        """
        Loads the BM25 model and the chunk metadata from disk.
        opposite of save_index
        """
        self.is_code = is_code
        print(f"Loading index from {load_dir}...")

        # 1. Load the math model
        self.retriever = bm25s.BM25.load(load_dir, load_corpus=False)

        # 2. Rehydrate the chunks
        chunks_path = os.path.join(load_dir, "chunks.json")
        with open(chunks_path, 'r', encoding='utf-8') as f:
            raw_chunks = json.load(f)

        # 3. Convert standard dictionaries back into Pydantic objects
        self.corpus_chunks = [MinimalSource(**chunk_dict) for chunk_dict in raw_chunks]

        print("Load complete!")

    def search(self, query: str, k: int = 5) -> List[MinimalSource]:
        """
        - Query Encoding
        - similarity search: Term Frequency-Inverse Document Frequency
            Ranking: the top k chunks
        - stop-word = remove noisy sounds
        """
        results: List[MinimalSource] = []
        clean_q = self._clean_text(query)
        stopwords = [] if self.is_code else "en"
        token_q = bm25s.tokenize(clean_q, stopwords=stopwords)
        # scores = data relevancy (14.344)
        # tiket is phyiscal position in list
        docs, _ = self.retriever.retrieve(token_q, k=k)
        for ticket in docs[0]:
            winning_chunks = self.corpus_chunks[ticket]
            results.append(winning_chunks)
            pass
        return results
