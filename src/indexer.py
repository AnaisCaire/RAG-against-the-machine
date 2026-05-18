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
    """
    Hybrid BM25 + (optional) semantic FAISS index: build/save/load/search.
    """

    def __init__(self, semantic: bool = False) -> None:
        """Sets up the BM25 retriever and, when semantic=True, loads the
        sentence-transformer embedding model.

        Args:
            semantic: If True, load the embedding model and enable FAISS
                      If False (default), only — BM25 will handle all retrieval.
        """
        self.corpus_chunks: List[MinimalSource] = []
        self.is_code: bool = False
        self.semantic: bool = semantic

        self.bm25_retriever = bm25s.BM25()
        self.faiss_index: Optional[faiss.IndexFlatIP] = None


        self.device: str = (
                "cuda" if torch.cuda.is_available()
                else "mps" if torch.backends.mps.is_available()
                else "cpu"
            )
        self.embedding_model: Optional[SentenceTransformer] = None

        if semantic:
            print(f"[Semantic] Loading embedding model on {self.device} ...")
            # SentenceTransformer :
            # tokenisation + padding + creating vector +
            self.embedding_model = SentenceTransformer(
                EMBEDDING_MODEL, device=self.device
            )
            print("[Semantic] Embedding model ready.")

    # ------------------------------------------------------------------ #
    #                               helpers                              #
    # ------------------------------------------------------------------ #

    def _make_corpus(self, chunks: List[MinimalSource]) -> List[str]:
        """Converts coordinate-based chunks into readable strings.

        Opens each file once to find raw text for each chunk.  
        For docs we add nearest heading and file-stem tokens.
        """
        corpus: List[str] = []

        dict_chunks: Dict[str, List[MinimalSource]] = defaultdict(list)
        for chunk in chunks:
            dict_chunks[chunk.file_path].append(chunk)

        for each_path, file_chunks in dict_chunks.items():
            try:
                with open(each_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                filename = os.path.relpath(each_path)
                for chunk in file_chunks:
                    chunk_text = content[
                        chunk.first_character_index:
                        chunk.last_character_index
                    ]
                    if self.is_code:
                        # For code, prepend the file path tokens (×3).
                        path_prefix = ' '.join(
                            self._extract_path_tokens(filename) * 3
                        )
                        corpus.append(f"{path_prefix}\n{chunk_text}")
                    else:
                        # For docs, prepend the file stem and nearest heading
                        stem = os.path.splitext(os.path.basename(filename))[0]
                        stem_tokens = [t for t in re.split(r'[_\-]', stem)
                                       if len(t) > 1]
                        stem_boost = (' '.join(stem_tokens * 2) + '\n'
                                      ) if stem_tokens else ''
                        heading = self._find_nearest_heading(
                            content, chunk.first_character_index)
                        header_boost = (f"{heading}\n{heading}\n"
                                        if heading else "")
                        table_header = self._find_table_header(
                            content, chunk.first_character_index, chunk_text)
                        table_boost = (f"{table_header}\n"
                                       if table_header else "")
                        corpus.append(
                            f"{stem_boost}{filename}\n"
                            f"{header_boost}{table_boost}{chunk_text}")

            except Exception:
                print(f"Warning: could not read {each_path} for corpus.")

        return corpus

    def _clean_text(self, text: str) -> str:
        """Normalise prose text for BM25 tokenisation.

        Splits camelCase, replaces punctuation with spaces, and lowercases
        so that 'BlockManager' and 'block manager' both tokenise the same way.
        """
        text = re.sub(r'(?<=[a-z0-9])(?=[A-Z])', ' ', text)
        cleaned = (text.replace("_", " ").replace(".", " ")
                   .replace("-", " ").replace("/", " "))
        return cleaned.lower()

    def _clean_code_text(self, text: str) -> str:
        """Extract identifier tokens from Python source for BM25.

        Steps:
        1. Keep only identifier characters (letters, digits, underscore).
        2. Split camelCase: BlockSpaceManager -> ['Block', 'Space', 'Manager'].
        3. Split snake_case: get_num_free_blocks -> ['get','num','free',...].
        4. Drop single-character tokens (loop vars like i, j) to reduce noise.
        5. Lowercase everything.
        """
        tokens = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*', text)
        result = []
        for token in tokens:
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
        """Strip markdown syntax and web noise before BM25 tokenisation.

        Order matters: strip structured syntax first (links, fenced blocks),
        then inline decorators, then whitespace noise.
        """
        # [link text](url) → link text  (keep the label, drop the URL)
        text = re.sub(r'\[([^\]]*)\]\([^\)]*\)', r'\1', text)
        # bare URLs
        text = re.sub(r'https?://\S+', ' ', text)
        # HTML tags
        text = re.sub(r'<[^>]+>', ' ', text)
        # fenced code blocks (``` ... ```) — keep the code content, drop fences
        text = re.sub(r'```[^\n]*\n', '', text)
        text = re.sub(r'```', '', text)
        # heading markers: '## My Section' → 'My Section'
        text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
        # bold and italic markers: **word** / *word* / __word__ / _word_
        text = re.sub(r'\*{1,2}([^*\n]+)\*{1,2}', r'\1', text)
        text = re.sub(r'_{1,2}([^_\n]+)_{1,2}', r'\1', text)
        # inline code: `identifier` → identifier
        # keeping the content is important — it often contains function names
        text = re.sub(r'`([^`]*)`', r'\1', text)
        # bullet / numbered list markers at the start of a line
        text = re.sub(r'^[\*\-\+]\s+', '', text, flags=re.MULTILINE)
        text = re.sub(r'^\d+\.\s+', '', text, flags=re.MULTILINE)
        return text

    @staticmethod
    def _find_nearest_heading(content: str, char_idx: int) -> str:
        """Return the markdown heading that gives context to this chunk.
          1. If the chunk itself opens with a '#' line → that IS the heading.
          2. Otherwise scan backward through content[:char_idx] for the nearest
             previous heading.
        """
        first_line = content[char_idx:].split('\n')[0].strip()
        if first_line.startswith('#'):
            return first_line.lstrip('#').strip()
        for line in reversed(content[:char_idx].split('\n')):
            stripped = line.strip()
            if stripped.startswith('#'):
                return stripped.lstrip('#').strip()
        return ''

    @staticmethod
    def _find_table_header(content: str,
                           chunk_start: int,
                           chunk_text: str) -> str:
        """Return column header text when a chunk contains table rows
        without the header separator row.

        Without this, chunks that are only the body rows of a table miss the
        column names entirely, hurting recall for queries about those columns.
        """
        if '|' not in chunk_text:
            return ''
        if re.search(r'^\s*\|[\s\-:|]+\|', chunk_text, re.MULTILINE):
            return ''
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

    # ------------------------------------------------------------------ #
    #       Public                                                       #
    # ------------------------------------------------------------------ #

    def build_index(self, chunks: List[MinimalSource], is_code: bool) -> None:
        """
        Build the BM25 index (always) 
        if semantic=True, build the FAISS dense vector index.

        Args:
            chunks:  List of MinimalSource chunks to index.
            is_code: True for Python source files, False for docs / text.
        """
        self.corpus_chunks = chunks
        self.is_code = is_code

        print(f"Extracting corpus from {len(chunks)} chunks...")
        raw_corp: List[str] = self._make_corpus(chunks)

        print("Building BM25 index...")
        if is_code:
            corpus: List[str] = [self._clean_code_text(t) for t in raw_corp]
        else:
            corpus = [self._clean_text(self._preprocess_markdown(t))
                      for t in raw_corp]
        # b=0.3 for code (shorter functions should not be penalised vs longer
        # files); b=0.75 is the standard BM25+ default for docs.
        b = 0.3 if is_code else 0.75
        self.bm25_retriever = bm25s.BM25(method="bm25+", b=b)
        corpus_tokens = bm25s.tokenize(corpus, stopwords="en")
        self.bm25_retriever.index(corpus_tokens)
        print("BM25 index built.")

        if self.semantic:
            if self.embedding_model is None:
                print("[Semantic] WARNING: no embedding model loaded, "
                      "skipping FAISS build.")
                self.semantic = False
                return

            print(
                f"[Semantic] Encoding {len(raw_corp)} chunks on "
                f"{self.device} ... (this can take several minutes)"
            )
            embeddings = self.embedding_model.encode(
                raw_corp,
                batch_size=EMBEDDING_BATCH_SIZE,
                show_progress_bar=True,
                normalize_embeddings=True,
            )
            dimension = embeddings.shape[1]
            # IndexFlatIP = exact inner-product (cosine) search, no
            # approximation.
            self.faiss_index = faiss.IndexFlatIP(dimension)
            self.faiss_index.add(np.array(embeddings).astype('float32'))
            print("[Semantic] FAISS index built.")

        print("Indexing complete!")

    def save_index(self, save_dir: str) -> None:
        """
        Save the index artefacts to disk.
        """
        print(f"Saving index to {save_dir} ...")
        os.makedirs(save_dir, exist_ok=True)

        # Chunk metadata — needed to map integer BM25/FAISS ranks back to
        # the actual MinimalSource objects at search time.
        stand_chunks = [chunk.model_dump() for chunk in self.corpus_chunks]
        with open(os.path.join(save_dir, 'chunks.json'), 'w') as file:
            json.dump(stand_chunks, file)

        # BM25 index files (vocab + sparse matrix).
        self.bm25_retriever.save(save_dir)

        # FAISS binary — only written when we actually built it.
        if self.semantic and self.faiss_index is not None:
            faiss.write_index(
                self.faiss_index,
                os.path.join(save_dir, "faiss.index")
            )
            print("[Semantic] FAISS index saved.")

        print("Save complete!")

    def load_index(self, load_dir: str, is_code: bool) -> None:
        """Load a previously saved index from disk.
        Args:
            load_dir: Directory that was passed to save_index().
            is_code:  Must match the value used at build time so text-cleaning
                      in search() is applied consistently.
        """
        self.is_code = is_code

        # Load chunk metadata.
        chunks_path = os.path.join(load_dir, "chunks.json")
        with open(chunks_path, 'r', encoding='utf-8') as f:
            raw_chunks = json.load(f)
        self.corpus_chunks = [
            MinimalSource(**chunk_dict) for chunk_dict in raw_chunks
        ]

        # Load BM25 sparse index.
        self.bm25_retriever = bm25s.BM25.load(load_dir, load_corpus=False)

        # Load FAISS dense index — only if requested and available.
        if self.semantic:
            faiss_path = os.path.join(load_dir, "faiss.index")
            if os.path.exists(faiss_path):
                self.faiss_index = faiss.read_index(faiss_path)
                print(f"[Semantic] FAISS index loaded from {load_dir}.")
            else:
                # silently fall back to BM25-only.
                print(
                    f"[Semantic] WARNING: faiss.index not found in {load_dir}."
                    " Re-run `index --semantic True` to build it."
                    " Falling back to BM25-only search."
                )
                self.semantic = False

        print("Load complete!")

    def search(self, query: str, k: int = DEFAULT_K) -> List[MinimalSource]:
        """Retrieve the top-k most relevant chunks for a query.

        1. BM25-only mode: return the BM25 ranking directly.
        2. semantic mode: fuses BM25 and FAISS rankings with
        Reciprocal Rank Fusion (RRF) for better precision.

        Args:
            query: The natural-language question or keyword query.
            k:     Number of chunks to return.
        """
        if k <= 0 or not query or not query.strip():
            return []

        # Apply the same text-cleaning to the query
        if self.is_code:
            clean_q = self._clean_code_text(query)
        else:
            clean_q = self._clean_text(query)

        # ---- BM25 retrieval ------------------------------
        # Fetch k * CANDIDATE_MULT candidates so the fusion step has
        # enough material to work with.
        token_q = bm25s.tokenize(clean_q, stopwords="en")
        docs, _ = self.bm25_retriever.retrieve(
            token_q, k=k * CANDIDATE_MULT)
        bm25_results = [self.corpus_chunks[ticket] for ticket in docs[0]]

        # ---- BM25-only path  -------
        if not self.semantic or self.faiss_index is None:
            return bm25_results[:k]

        # ---- Semantic retrieval  ---------------
        # Encode the raw query (not the cleaned version) so the transformer
        # sees natural language rather than identifier tokens.
        assert self.embedding_model is not None  # guaranteed by __init__ guard
        query_vector = self.embedding_model.encode(
            [query], normalize_embeddings=True)
        _, indices = self.faiss_index.search(
            np.array(query_vector).astype('float32'), k * CANDIDATE_MULT)
        semantic_results = [
            self.corpus_chunks[idx] for idx in indices[0] if idx != -1
        ]

        # ---- Reciprocal Rank Fusion (RRF) --------------------------------
        # Score each chunk by summing 1/(rank + RRF_CONSTANT) across both
        # rankers.  A chunk that appears near the top of both BM25 and FAISS
        # accumulates the highest score.
        def chunk_id(c: MinimalSource) -> str:
            """Unique string key for a chunk — file + start offset."""
            return f"{c.file_path}_{c.first_character_index}"

        rrf_scores: Dict[str, float] = defaultdict(float)
        chunk_lookup: Dict[str, MinimalSource] = {}

        for rank, chunk in enumerate(bm25_results):
            cid = chunk_id(chunk)
            chunk_lookup[cid] = chunk
            rrf_scores[cid] += 1.0 / (RRF_CONSTANT + rank + 1)

        for rank, chunk in enumerate(semantic_results):
            cid = chunk_id(chunk)
            chunk_lookup[cid] = chunk
            rrf_scores[cid] += 1.0 / (RRF_CONSTANT + rank + 1)

        sorted_ids = sorted(
            rrf_scores, key=lambda x: rrf_scores[x], reverse=True
        )
        return [chunk_lookup[cid] for cid in sorted_ids[:k]]
