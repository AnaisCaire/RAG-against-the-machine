# RAG-against-the-machine
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

# Explanation of my work:
## 1. models + makefile + pyproject.toml
    copied models from subject

## 2. chunking code and text
    chunking code works by using abstartct syntax tree. 
    we are between where a code is parced and before it gets compiled to byte code. 
use this to understand:
1. visualize AST: https://astexplorer.net/
2. learn about AST: https://greentreesnakes.readthedocs.io/en/latest/
    after making the tree, we need to cut:
    class object if less than 2000 chars.
    else at the functions of the class. 
    if more than 2000 chars, 
    we will use the text chunking

    chunking text will find the 200th caracter then, 
    if between a word, find the nearest paragraph
    else find the nearest next line
    else find the nearest space. 

then we add theses chunks to the MinimalSource class where we know 
the last and first index for the chunking. 

## 3. ingestion model

    the subject says:
    "Read and process all files from the VLLM repository... and create a searchable index"

    this will not be done mannually so we create a class : ingestion for logistics and routing:
    Crawl through every single folder and subfolder in a given directory.

        1. Pick up every file.
        use os.walk() : https://www.tutorialspoint.com/python/os_walk.htm 

        2. Look at the file and ask, "Which tool handles this format?"
        3. Route the file's text to the correct chunker.
        4. Collect all the resulting chunks into one massive, master list.

## 4 Indexing

    first problem, BM25 or TF-IDF are mathemaitcal models and dont undestand what a file path and 
    a first_index/last_index from the MinimalSource means.
    we need to find the raw string again...
    with the make_corpus function
        this function

    build_index function:
    https://github.com/xhluca/bm25s 
    == to understand how to quick start the bm25s
    ok but what is a retreiver???
    bm25 = best matching 25
    its a scoring technique in stats.
    counts every word frequency in every chunk. and then puts a penality to words that apprear 
    in every chunk. 
    then the actual index is where we save the mathematical weights of every token into an
    optimised data base. (inverted index) == because we inverse document frequence, because we 
    penalize the most frequent. 

    now, the subjects states: "Store the index for fast retrieval".
    we will create a directory to save 2 things:
        1. the BM25 Model
        2. my chunks

    ok now the indexes are saves but we need to able to search them...
    with the bm25 lib we can use: self.retriever.retrieve(query_tokens, k=num_results)

    this will return a dict/tuple of the highest scoring chunks
    with thoes numerical indexes, we check back our coprus chunks to get the actual Minimalsource object related to thoses values...

    What i had to do to make my model more precise. 
    OK, so the indexing is very important for making it precise...
    1. clean code text function:
        what is really does: def my_function(x=5): becomes a list: ['def', 'my_function', 'x']
        then, it breaks down the camelCase so: BlockSpaceManager becomes ['Block', 'Space', 'Manager']
        this is better for searching 
        same think for the snake_case : get_num_free_blocks becomes ['get', 'num', 'free', 'blocks']
     2. the length normalisation in BM25 algo:
        b = 1.0 -> pensalize long documents because it assums its rambling
        b = 0.0 -> dosent event look at lenght
        for docs, the reaserch suggests to have b=0.75 
            if a 2000 word essay mentions coffee once
            vs a 100 word essay mentions "cofee" once
            we can assume the 100 word essay is more likely to be relavant
        for code, a 2000 caracter class is not rambling.
        if b = 0.75 it will only return the small utility files and not the real code
        this is a very important distinction to make in the BM25 

## generator

    Now that we have the actual MinimalSource objects, we need to feed them to the Qwen LLM. But the LLM only reads strings, not objects.

    once we found the right code, we need to explain it to the user
    this is to generate natrual language answers and "Pass retrieved context to the LLM within token limits".
    1. LLM cannot real python Minimal source object list... it only reads raw strings
        we need to glue the users question and the retrived text together in a new structured string.

    now we load the transformer... this will sreach the web and download the AI qwen model for us.
        https://huggingface.co/docs/transformers/main_classes/pipelines
    
    2. generate answer:
        we need to use the MinimalAnswer model 

    ok now i have to generate questions in less than 2 seconds... the self.pipeline(..) is not precise enough to change somme settings needed for optimisation.

## batch 

    we need to read the Json files.
    theses 2 functions do that exactly
    - search_dataset (Chapter V.6.5): Reads the UnansweredQuestions JSON, searches the index, and saves a srcSearchResults JSON.

    - answer_dataset (Chapter V.6.7): Reads the newly created srcSearchResults JSON, passes those pre-found tickets to the LLM, and saves a srcSearchResultsAndAnswer JSON.



### Bonuses

## LLM powered Querry Expansion
    this was a good bonus to add becuse i wanted to augment the accuray for the code base search...
    BM25 is a super good lexical search algo for strandar text but it does struggle with codebases.
    if i try to look for "flash attention" BM25 will have trouble finind "triton_flash_attention()"
    Camel snake functions are treated in this model as a single word... making the match super unlikely
# implementation
    before a search, we route the query to the Generator 
    the generator will then brainstorm related words linked to the question
    we then append theses to the user's original question

## Result catching (Memory Bank)
    users tend to ask the same questions alot of times... so Running the model again in expensive and time-consuming for nothing.
    No need to force the GPU to perdorme the same matrix multiplication it aleardy did
# implementation
    create a hash map in the generator as a "memory bank"
    before answering a question it checks if the same question is already there.
    if found we just give out the same answer directly

## Seamntic Embedding
    2 new concepts:
        sentence transformers:
        running a raw text string into a sentence transformer model is actually translating every chunk of text as a dense array of numbers == vectors
        we also normalize the embeddings to the lenght of 1 so we can calculate the cosine similarity between vectors. 
        FAISS: facebook AI Similarity Search = the most optimal library
        this is about building a database and being able to search massive databases of vectors efficiently.

    1 update the indexr.py to have both BM25 and semantic embedding

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


## ressources:

    https://www.youtube.com/watch?v=8OJC21T2SL4

    https://www.chunkviz.com/


## Evaluation preparation:
Need to know the difference between all the steps taken:
Indexing:
Retrieval:
Augmenting:
Generation:

questions:
What happens if I change the max chunk size on the retrieval performance
Explain chunking strategies


1. "What is Retrieval-Augmented Generation (RAG) and why is it useful?"
2. "Walk me through your complete RAG pipeline, from raw documents to a generated answer."
3. "What is the difference between TF-IDF and BM25 as retrieval methods?"
4. "What implementation choices did you make and why? What trade-offs did you consider?"
5. "If you had more time, what would you improve in your system?"