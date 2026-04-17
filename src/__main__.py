from src.chunker import CodeChunker, TextChunker
import pathlib


def main():
    # ----------- code chunk tests ------------

    file_path = "raw/vllm-0.10.1/vllm/core/placeholder_block_space_manager.py"
    content = pathlib.Path(file_path).read_text()
    chunker = CodeChunker(max_chunk_size=2000)
    code_chunk_result = chunker.chunk(file_path, content)
    print(code_chunk_result)

    # ----------- text chunk tests ------------
    text_file_path = "raw/vllm-0.10.1/RELEASE.md"
    text_content = pathlib.Path(text_file_path).read_text()
    breakpoint()
    txt_chunker = TextChunker(max_chunk_size=2000)
    text_chunk_result = txt_chunker.chunk(text_file_path, text_content)
    print(text_chunk_result)
    breakpoint()


if __name__ == "__main__":
    main()
