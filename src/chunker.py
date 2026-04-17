import ast
from typing import List
from src.models import MinimalSource


class BaseChunker:
    """Base class for all chunking strategies."""
    def __init__(self, max_chunk_size: int):
        self.max_chunk_size = max_chunk_size

    def chunk(self, file_path: str, content: str) -> List[MinimalSource]:
        raise NotImplementedError("Subclasses must implement this method.")


class TextChunker(BaseChunker):
    """Handles Markdown and standard text chunking."""
    def chunk(self, file_path: str, content: str) -> List[MinimalSource]:
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
                current_idx = absolute_end
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

            # 2. Calculate absolute start and end indices using line_starts
            absolute_start = line_starts[lineno - 1] + col_offset
            absolute_end = line_starts[end_lineno - 1] + end_col_offset

            node_length = absolute_end - absolute_start

            # 3. The Size Decision Engine
            # Case A: It fits perfectly!
            if node_length <= self.max_chunk_size:
                chunks.append(MinimalSource(
                    file_path=file_path,
                    first_character_index=absolute_start,
                    last_character_index=absolute_end))

            # Case B: It is too big, BUT it has a body we can divide more
            elif body:
                for child_node in body:
                    chunks.extend(self._process_node(child_node,
                                                     content,
                                                     line_starts,
                                                     file_path))

            # Case C: It is too big, and it DOES NOT have a body.
            else:
                txt_chunks = TextChunker(self.max_chunk_size)
                node_content = content[absolute_start:absolute_end]
                chunks = txt_chunks.chunk(file_path, node_content)

            return chunks

        # 4. for invisibles nodes e.g(the top-level Module)
        # we iterate through its body.
        if body:
            for child in body:
                chunks.extend(self._process_node(
                    child, content, line_starts, file_path))

        return chunks

    def chunk(self, file_path: str, content: str) -> List[MinimalSource]:

        line_starts = self._line_start_helper(content)
        tree = ast.parse(content)
        return (self._process_node(tree, content, line_starts, file_path))
