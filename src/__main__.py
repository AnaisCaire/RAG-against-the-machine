from src.indexer import Indexer
from src.chunker import CodeChunker, TextChunker
from src.ingestion import IngestionEngine
from src.generator import Generator
import pathlib


def main():
    # ----------- code chunk tests ------------

    file_path = "raw/vllm-0.10.1/vllm/core/placeholder_block_space_manager.py"
    content = pathlib.Path(file_path).read_text()
    chunker = CodeChunker(max_chunk_size=2000)
    code_chunk_result = chunker.chunk(file_path, content)
    # print(code_chunk_result)

    # ----------- text chunk tests ------------
    text_file_path = "raw/vllm-0.10.1/RELEASE.md"
    text_content = pathlib.Path(text_file_path).read_text()

    txt_chunker = TextChunker(max_chunk_size=2000)
    text_chunk_result = txt_chunker.chunk(text_file_path, text_content)
    # print(text_chunk_result)

    # --------- test ingestion -------
    ingestion = IngestionEngine(2000)
    big_list = ingestion.ingest_directory("raw/vllm-0.10.1")

    # ----- test indexer -----
    indexer = Indexer()
    # (Note: Since you already saved it, you could technically use
    # indexer.load_index('processes/index_bm25') here to save time!)
    test = indexer.build_index(big_list)

    # ----- test search ----
    query = "What activation formats does the fused batched MoE layer return in vLLM?"
    print("\nSearching index...")
    found_chunks = indexer.search(query, k=3) # Let's just grab the top 3 for a faster test

    # ----- generation -----
    print("\nInitializing AI Generator...")
    gen = Generator()
    answer_test = gen.generate_answer(
        question_id="189c8b8a-e59c-4fca-92ad-c02df42cbe40",
        query=query,
        retrieved_sources=found_chunks  # Pass the Coat Check tickets to the AI!
    )
    print("\n--- FINAL OUTPUT ---")
    print(answer_test.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
