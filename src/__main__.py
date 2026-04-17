from src.chunker import CodeChunker
import pathlib


def main():
    file_path = "raw/vllm-0.10.1/vllm/core/placeholder_block_space_manager.py"
    content = pathlib.Path(file_path).read_text()
    chunker = CodeChunker(max_chunk_size=2000)
    chunker.chunk(file_path, content)


if __name__ == "__main__":
    main()
