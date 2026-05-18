*This project has been created as part of the 42 cucurriculum by acaire-d*

# RAG-against-the-machine

## Description
Build a Retrieval-Augmented Generation system that answers questions about codebases by retrieving relevant information and generating evidence-based responses, implementing intelligent chunking, efficient retrieval (TF-IDF/BM25)

## Installation

This project uses `uv` as its package and dependency manager. To set up the environment and install all required dependencies (like `transformers`, `pydantic`, `bm25s`, `fire`, etc.), ensure `uv` is installed on your system and run:

```bash
make install
```

## Usage and Execution

The entire Retrieval-Augmented Generation (RAG) pipeline is exposed as a Command-Line Interface (CLI) powered by Python Fire. All commands are executed through the src module module.
### 1. build the index
```bash
uv run python -m src index --max_chunk_size 2000
```
### 2. Answer a single Query
if you want to ask a specific question, use this.
```bash
uv run python -m src answer "What activation formats does the fused batched MoE layer return in vLLM?" --k 10
```

### 3. Batch Search a Dataset
For processing the entire dataset of questions and output the retrieved source coordinates to a JSON file, use this.
```bash
uv run python -m src search_dataset --dataset_path data/datasets/UnansweredQuestions/dataset_code_public.json --k 10 --save_directory data/output/search_results
```

### 4. Batch Answer a Dataset
To generate all the AI answers for the searched dataset, and save the final result, use this.
```bash
uv run python -m src answer_dataset --student_search_results_path data/output/search_results/dataset_code_public.json --save_directory data/output/search_results_and_answer
```

### 5. Evaluate Performance
to evaluate the accuracy of the answers, with the ground truth dataset, use this.
```bash
uv run python -m src evaluate --student_results_path="data/output/search_results/dataset_code_public.json" --dataset_path="data/datasets/AnsweredQuestions/dataset_code_public.json" --k=10
```

# Development Log: How I Built This

## 1. Setting Up the Foundation

Step one i, I set up the `pyproject.toml` and the `Makefile`. For the data structures, I copied the models exactly as they were defined in the subject PDF so everything aligns perfectly from the start.

## 2. Figuring Out the Chunking (Code & Text)

I had to split the files up, and I realized chunking code works best if I use an Abstract Syntax Tree (AST). Basically, I'm grabbing the code right after it gets parsed, but before it compiles down to bytecode. (checking out this to understand [AST Explorer](https://astexplorer.net/) or the [Green Tree Snakes docs](https://greentreesnakes.readthedocs.io/en/latest/) it really helped me).

Once the tree is built, I have to make the cuts:

* If a class object is under 2000 characters, I keep it as one solid chunk.
* If it's bigger than that, I break it down into the class's individual functions.
* If a function *still* exceeds 2000 characters, I fall back to my text chunking method.

For text chunking, my logic is straightforward: I find the 200th character. If that lands in the middle of a word, I look forward to find the nearest title break. then, the paragraph break. If there isn't one, I look for the next line break (`\n`), and if all else fails, I just snap to the nearest space.

After cutting everything up, I pack these chunks into my `MinimalSource` class. This is super important because it lets me track the exact `first_index` and `last_index` of where that chunk originally lived.

## 3. The Ingestion Engine

The subject clearly states: *"Read and process all files from the VLLM repository... and create a searchable index"*.

Obviously, I'm not doing that manually. I built an `Ingestion` class to handle all the logistics and routing. It crawls through every single folder and subfolder using Python's `os.walk()`.
For every file it picks up, the script asks, "Which tool handles this format?" It routes the file's text to the correct chunker, and then collects every single resulting chunk into one massive master list.

4. Indexing & The BM25 Problem

Models like BM25 or TF-IDF are purely statistical text models. They have no idea what a file path is, and they don't understand the first_index/last_index tracking in my MinimalSource objects. They only ingest raw tokens.

So, you need a function like the make_corpus function to strip the metadata away and extract just the raw text back out.

For the indexing, I used the bm25s library (look at their quickstart guide on GitHub). BM25 relies on two concepts: it counts the frequency of a word in a specific chunk (Term Frequency), but it also puts a mathematical penalty on words that show up in every chunk (Inverse Document Frequency).

What gets saved? the model builds an inverted index. It maps every unique token to a list of the exact chunks where that word appears, with its frequency stats. (The final "score" for a search is actually calculated dynamically at query time using these saved stats).

The subject also asks to "Store the index for fast retrieval". So, I created a directory specifically to save two things to disk: the serialized BM25 inverted index (so I don't have to recalculate the document frequencies every run), and my actual chunks so I can map the mathematical results back to my code.

**How I made the search actually precise:**
making them return good results when I call `self.retriever.retrieve(query_tokens, k=num_results)` required two massive adjustments:

1. **Cleaning the code text:** Standard search is bad at reading code. I wrote a function so that something like `def my_function(x=5):` gets broken down into a clean list: `['def', 'my_function', 'x']`. I also force it to split camelCase (`BlockSpaceManager` becomes `['Block', 'Space', 'Manager']`) and snake_case (`get_num_free_blocks` becomes `['get', 'num', 'free', 'blocks']`). This makes a world of difference for search accuracy.
2. **Tuning the BM25 Length Normalization ($b$):** In BM25, the $b$ parameter penalizes long documents because it assumes length equals rambling. The research suggests setting $b = 0.75$ for normal text (a 100-word essay about coffee is probably more relevant than a 2000-word essay that mentions coffee once). **But for code, this is a trap.** A 2000-character Python class isn't rambling; it's just a class. If I left $b = 0.75$, the retriever would only return tiny utility files and ignore my actual core logic blocks. Adjusting this was a crucial fix.

Once the retriever spits out the highest scoring numerical indexes, I just check those numbers against my master corpus to pull the correct `MinimalSource` objects.

## 5. Hooking Up the Generator

Now I have the right `MinimalSource` objects, but I need to feed them to the Qwen LLM so it can actually explain the code to the user in natural language (and hit the requirement to *"Pass retrieved context to the LLM within token limits"*).

Again, the LLM only reads raw strings, not Python objects. I had to write a parser to glue the user's question and the retrieved chunk text together into a new, clean string.

I load up the transformer using Hugging Face pipelines, which handles downloading the Qwen AI model for me. The generated text then gets packed into my `MinimalAnswer` model.

*The catch?* I had to generate these questions in under 2 seconds. The default `self.pipeline(...)` setup wasn't precise enough, so I had to dig in and tweak the settings to optimize the speed.

## 6. Processing the Batches

Finally, to handle the actual JSON files, I built two functions that do exactly that:

* `search_dataset` (Chapter V.6.5): Reads the `UnansweredQuestions.json`, searches my index, and saves the hits to `srcSearchResults.json`.
* `answer_dataset` (Chapter V.6.7): Picks up that newly created `srcSearchResults.json`, feeds those pre-found tickets to the LLM, and writes the final output to `srcSearchResultsAndAnswer.json`.

### Bonuses

## LLM Powered Query Expansion
    this was a good bonus to add because I wanted to improve accuracy specifically for codebase search.
    BM25 is a great lexical search algorithm for standard text but it really struggles with codebases.
    if I search for "flash attention", BM25 will have trouble finding "triton_flash_attention()"
    camelCase and snake_case functions are treated as a single word, making a match super unlikely.

### implementation
    before a search, we route the query to the Generator
    the generator brainstorms related keywords and synonyms linked to the question
    we then append these to the user's original query before searching

## Result Caching (Memory Bank)
    users tend to ask the same questions a lot... so running the model again is expensive and time-consuming for nothing.
    no need to force the GPU to perform the same matrix multiplications it already did.

### implementation
    create a hash map in the generator as a "memory bank"
    before answering a question it checks if the exact same question is already in there
    if found, we just return the cached answer directly

## Semantic Embedding
    2 new concepts:

    sentence transformers:
        running a raw text string through a sentence transformer model translates every chunk of text into a dense array of numbers — a vector.
        we also normalize the embeddings to length 1 so we can measure similarity between vectors using cosine similarity.

    FAISS (Facebook AI Similarity Search):
        the most efficient library for building and searching through massive vector databases.

### implementation
    updated indexer.py to build both a BM25 index and a FAISS semantic index in parallel.
    at search time, both scores are combined (hybrid retrieval) and the top-k results are returned.

## Performance Analysis (private dataset, hybrid BM25 + FAISS)

|               | Docs          | Code         |
|---------------|---------------|--------------|
| Recall@10     | **95.0%** ✅  | **79.0%** ✅ |
| Precision@10  | 13.7%         | 10.3%        |

## the schools computer problems...

if i pip install uv even in goinfre, it will install in my home directory

We also need to control where the computer will save the data and route everything to do to the goifre...

## setp 1:
    make folders for binairies and caches: 
    ```bash
    mkdir -p /goinfre/$USER/bin
    mkdir -p /goinfre/$USER/uv_cache
    mkdir -p /goinfre/$USER/hf_cache
    ```
## step 2:
    reroute enviroment variables so it looks in goinfre for commands and forces the download to goinfre insted of home drive
    ```bash
    export PATH="/goinfre/$USER/bin:$PATH"
    export UV_CACHE_DIR="/goinfre/$USER/uv_cache"
    export HF_HOME="/goinfre/$USER/hf_cache"
    ```
### step 3:
    install uv directy in goinfre 
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="/goinfre/$USER/bin" sh
    ```


## Resources & References

    - AST explorer. (n.d.). Retrieved May 18, 2026, from https://astexplorer.net/

    - Green tree snakes: The missing Python AST docs. (n.d.). Retrieved May 18, 2026, from https://greentreesnakes.readthedocs.io/en/latest/

    - Hugging Face. (n.d.). Pipelines. Retrieved May 18, 2026, from https://huggingface.co/docs/transformers/main_classes/pipelines

    - Kamradt, G. (n.d.). ChunkViz. Retrieved May 18, 2026, from https://www.chunkviz.com/

    - Kamradt, G. (n.d.). The 5 levels of text splitting for retrieval [Video]. YouTube. Retrieved May 18, 2026, from https://www.youtube.com/watch?v=8OJC21T2SL4

    - Tutorialspoint. (n.d.). Python os.walk() method. Retrieved May 18, 2026, from https://www.tutorialspoint.com/python/os_walk.htm

    - xhluca. (n.d.). bm25s [Computer software]. GitHub. Retrieved May 18, 2026, from https://github.com/xhluca/bm25s
