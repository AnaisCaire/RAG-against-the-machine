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
    a first_index/last_index from the MinimalSource means...
    we need to find the raw string again...