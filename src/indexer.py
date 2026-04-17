import os
import bm25s
import json
from typing import List, Dict
from collections import defaultdict
from src.models import MinimalSource


class Indexer:
    def __init__(self):
        self.corpus_chunks: List[MinimalSource] = []
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
                    # ??? HOW DO WE EXTRACT THE TEXT? ???
                    chunk_text = content[chunk.first_character_index: chunk.last_character_index]
                    corpus.append(chunk_text)
                    pass

            except Exception:
                print(f"Warning: Could not read {each_path} for corpus creation.")

        return corpus

    def build_index(self, chunks: List[MinimalSource]) -> None:
        """
        Extracts text, tokenizes it, and builds the BM25 index.
        """
        self.corpus_chunks = chunks

        print("Extracting corpus from chunks...")
        corpus: List[str] = self._make_corpus(chunks)

        print("Tokenizing corpus...")
        corpus_tokens = bm25s.tokenize(corpus)

        print("Training BM25 Index...")
        # find a libririan
        retriever = bm25s.BM25()
        # make the libririan read it all
        retriever.index(corpus_tokens)

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
