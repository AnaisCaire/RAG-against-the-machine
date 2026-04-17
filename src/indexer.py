from typing import List, Dict
from collections import defaultdict
from src.models import MinimalSource


class Indexer():
    def __init__(self):
        pass

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

            except Exception as e:
                # Defensive programming: skip unreadable files gracefully
                print(f"Warning: Could not read {each_path} for corpus generation.")

        return corpus

    def build_index(self, chunks: List[MinimalSource]) -> None:

        corpus: List[str] = self._make_corpus(chunks)
        # 2. Tokenize (split into words)
        corpus_tokens = bm25s.tokenize(corpus)

        # 3. Build the Index
        retriever = bm25s.BM25()
        retriever.index(corpus_tokens)

    def save_index(self, save_dir: str) -> None:
        pass