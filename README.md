# RAG-against-the-machine
Build a Retrieval-Augmented Generation system that answers questions about codebases by retrieving relevant information and generating evidence-based responses, implementing intelligent chunking, efficient retrieval (TF-IDF/BM25)


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

    first problem, BM25 or TF-IDF are mathemaitcal models and dont undestant what a file path and 
    a first_index/last_index from the MinimalSource means.
    we need to find the raw string again...
    with the make_corpus function

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
    with the b,25 lib we can use: self.retriever.retrieve(query_tokens, k=num_results)

    this will return a dict/tuple of the highest scoring chunks
    with thoes numerical indeces, we check back our coprus chunks to get the actual Minimalsource object related to thoses values...
    ok but why:
        >>>
        Think of your self.corpus_chunks list as a Coat Check at a fancy restaurant.

        Storing: In Milestone 2, you extracted 71,157 text chunks. You put them in a giant list (self.corpus_chunks).

        The Ticket: The position in that list is the "Ticket Number." The very first chunk is Ticket 0. The 42nd chunk is Ticket 42.

        The Valet (BM25): You gave the text to the BM25 Valet to memorize.

        Retrieval: When you search, the Valet hands you back Ticket 42. To get the actual file path and coordinates required by the subject, you must take Ticket 42, walk over to your self.corpus_chunks list, and pull out the object sitting at index [42].
        <<<

## generator

    Now that you have pulled the actual MinimalSource objects using those tickets, we need to feed them to the Qwen LLM. But the LLM only reads strings, not objects.

    once we found the right code, we need to explain it to the user
    this is to generate natrual language answers and "Pass retrieved context to the LLM within token limits".
    1. LLM cannot real python Minimal source object list... it only reads raw strings
        we need to glue the users question and the retrived text together in a new structured string.

    now we load the transformer... this will sreach the web and download the AI qwen model for us.
        https://huggingface.co/docs/transformers/main_classes/pipelines
    
    2. generate answer:
        we need to use the MinimalAnswer model 