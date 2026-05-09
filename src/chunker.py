import ast
from typing import List
from src.models import MinimalSource
from src.config import CHUNK_OVERLAP


class BaseChunker:
    """Base class for all chunking strategies."""

    def __init__(self, max_chunk_size: int) -> None:
        """Sets the maximum allowed chunk size in characters."""
        self.max_chunk_size = max_chunk_size

    def chunk(self, file_path: str, content: str) -> List[MinimalSource]:
        """Splits content into chunks. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement this method.")


class TextChunker(BaseChunker):
    """Handles Markdown and standard text chunking."""

    def chunk(self, file_path: str, content: str) -> List[MinimalSource]:
        """Splits text at paragraph/line/word boundaries up to max_chunk_size."""
        chunks: List[MinimalSource] = []
        content_length = len(content)
        current_idx = 0
        break_index = 0

        while current_idx < content_length:
            # get the max end index of the chunk
            max_end = min(current_idx + self.max_chunk_size, content_length)

            if max_end == content_length:
                chunks.append(MinimalSource(
                    file_path=file_path,
                    first_character_index=current_idx,
                    last_character_index=max_end
                ))
                break
            else:
                node_text = content[current_idx:max_end]
                if node_text.rfind('\n\n') >= 0:
                    break_index = node_text.rfind('\n\n') + 2
                elif node_text.rfind('\n') >= 0:
                    break_index = node_text.rfind('\n') + 1
                elif node_text.rfind(' ') >= 0:
                    break_index = node_text.rfind(' ') + 1
                else:
                    break_index = len(node_text)

                absolute_end = current_idx + break_index
                chunks.append(MinimalSource(
                    file_path=file_path,
                    first_character_index=current_idx,
                    last_character_index=absolute_end))
                next_idx = absolute_end - CHUNK_OVERLAP
                current_idx = next_idx if next_idx > current_idx else absolute_end
        return chunks


class CodeChunker(BaseChunker):
    """Intelligently chunks Python code using AST."""

    @staticmethod
    def _line_start_helper(content: str) -> List[int]:
        """
        maps where every line starts in the raw text.
        (lineo and col_offset wont be enough)
        line number is not the absolute character, and thats what we need
        the int is the amount of chars between the 0 and the adding new line
        note: AST line numbers are 1-indexed, so line_starts[0] = line 1.
        """
        starts: List[int] = [0]
        for line in content.splitlines(keepends=True):
            starts.append(starts[-1] + len(line))
        return starts

    def _process_node(self,
                      node: ast.AST,
                      content: str,
                      line_starts: List[int],
                      file_path: str) -> List[MinimalSource]:
        """
        Recursively evaluates AST node and breaks it down
        if it exceeds max_chunk_size.
        """
        chunks: List[MinimalSource] = []

        # for lint:
        lineno = getattr(node, 'lineno', None)
        end_lineno = getattr(node, 'end_lineno', None)
        col_offset = getattr(node, 'col_offset', None)
        end_col_offset = getattr(node, 'end_col_offset', None)
        body = getattr(node, 'body', None)

        # 1. Does this node have coordinates? (Not all AST nodes do!)
        if (lineno is not None and end_lineno is not None and
                col_offset is not None and end_col_offset is not None):

            # AST nodes often point to the `def`/`class` keyword. We must explicitly 
            # check the decorator_list to find the true semantic start of the code block.
            decorator_list = getattr(node, 'decorator_list', [])
            if decorator_list:
                first_dec = decorator_list[0]
                dec_ln = getattr(first_dec, 'lineno', lineno)
                dec_col = getattr(first_dec, 'col_offset', col_offset)
                
                # If the decorator starts earlier, update our target coordinates
                if dec_ln < lineno or (dec_ln == lineno and dec_col < col_offset):
                    lineno = dec_ln
                    col_offset = dec_col

            # 2. Calculate absolute start and end indices
            absolute_start = line_starts[lineno - 1] + col_offset
            absolute_end = line_starts[end_lineno - 1] + end_col_offset

            # In Python's AST, a decorator's col_offset points to its name (e.g., 'property'),
            # missing the physical '@' symbol. We step backward by 1 to capture it.
            if decorator_list and absolute_start > 0 and content[absolute_start - 1] == '@':
                absolute_start -= 1

            node_length = absolute_end - absolute_start

            # The Size Decision Engine
            # Case A: It fits perfectly!
            if node_length <= self.max_chunk_size:
                chunks.append(MinimalSource(
                    file_path=file_path,
                    first_character_index=absolute_start,
                    last_character_index=absolute_end))

            # Case B: It is too big, BUT it has a body we can divide more
            elif body:
                # Emit the header (signature, decorators, leading
                # docstring) before recursing. Without this, chars
                # between `def`/`class` and the first body statement
                # are never stored in any chunk, causing recall misses.
                first_child = body[0]
                fc_ln = getattr(first_child, 'lineno', None)
                fc_col = getattr(first_child, 'col_offset', None)
                if fc_ln is not None and fc_col is not None:
                    first_body_start = line_starts[fc_ln - 1] + fc_col
                    if first_body_start > absolute_start:
                        header_len = first_body_start - absolute_start
                        if header_len <= self.max_chunk_size:
                            chunks.append(MinimalSource(
                                file_path=file_path,
                                first_character_index=absolute_start,
                                last_character_index=first_body_start))
                        else:
                            txt_chunks = TextChunker(self.max_chunk_size)
                            node_text = content[
                                absolute_start:first_body_start
                            ]
                            for c in txt_chunks.chunk(file_path, node_text):
                                start = (
                                    c.first_character_index + absolute_start
                                )
                                end = (
                                    c.last_character_index + absolute_start
                                )
                                chunks.append(MinimalSource(
                                    file_path=file_path,
                                    first_character_index=start,
                                    last_character_index=end))
                for child_node in body:
                    chunks.extend(self._process_node(child_node,
                                                     content,
                                                     line_starts,
                                                     file_path))

            # Case C: It is too big, and it DOES NOT have a body.
            else:
                txt_chunks = TextChunker(self.max_chunk_size)
                node_content = content[absolute_start:absolute_end]
                relative_chunks = txt_chunks.chunk(file_path, node_content)
                # TextChunker indices are relative to node_content
                # (0-based); shift back to absolute file positions.
                chunks = [
                    MinimalSource(
                        file_path=c.file_path,
                        first_character_index=(
                            c.first_character_index + absolute_start
                        ),
                        last_character_index=(
                            c.last_character_index + absolute_start
                        )
                    )
                    for c in relative_chunks
                ]

            return chunks

        # 4. for invisibles nodes e.g(the top-level Module)
        # we iterate through its body.
        if body:
            for child in body:
                chunks.extend(self._process_node(
                    child, content, line_starts, file_path))

        return chunks

    def chunk(self, file_path: str, content: str) -> List[MinimalSource]:
        """Parses file as Python AST and chunks top-level nodes recursively."""
        line_starts = self._line_start_helper(content)
        tree = ast.parse(content)
        return (self._process_node(tree, content, line_starts, file_path))
