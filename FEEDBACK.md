

## SUMMARY TABLE

| Severity   | File              | Issue                                           |
|------------|-------------------|-------------------------------------------------|
| CRITICAL   | generator.py      | Wrong model name (must be Qwen/Qwen3-0.6B)      |
| CRITICAL   | pyproject.toml    | Missing `tqdm` and `fire` dependencies           |
| IMPORTANT  | chunker.py        | Case C TextChunker fallback: wrong char indices  |
| IMPORTANT  | evaluate.py       | Uses IoU instead of simple overlap ratio         |
| IMPORTANT  | batch.py          | search_dataset missing tqdm progress bar         |
| MINOR      | batch.py          | search_engine type hint should be Optional       |
| MINOR      | __main__.py       | max_context_length accepted but ignored          |
| MINOR      | __main__.py       | Index path naming differs from subject example   |


Why you're failing: 3 root causes
Problem 1 (biggest): 98.4% of your index is Python code

Python chunks: 107,698  (98.4%)
Doc chunks:       1,773  (1.6%)
For any docs question, your index is nearly 99% noise. BM25 has 60× more Python chunks to pick from than docs chunks, so it almost always returns code files first. This accounts for 24 of your 35 failing questions.

You can see it in the output — queries like "What hardware platforms does vLLM support?" return Python test files instead of the relevant .md file.

Problem 2: Chunk size 800 is too small for docs

Expected source lengths:
  Median: 1162 chars
  Max:    1997 chars
  Count > 800 chars: 69 out of 100
Most expected answers are longer than your 800-char chunks, so the answer is split across 2–3 chunks. BM25 often returns a chunk from the right file but the wrong section (e.g., the file header instead of the answer). This accounts for 10 of your 35 failing questions.

For example, for "How do you build and run a Docker image for s390x?":

Expected: chars 2452–3126 (674 chars)
You got: chars 0–732 (start of file, 0% overlap)
The correct chunks (2010–2676, 2676–3127) exist in your index but rank outside top-10 because Python files flood the results.
Problem 3 (minor): Duplicate results waste top-10 slots
The same file path appears multiple times in your results (e.g., prefix_caching.py twice), burning retrieval slots that could find the answer.

The fix
You need two separate indices — one for docs (.md/.txt only) and one for code (.py only). Then route the search to the right index based on the dataset.

The simplest path: add a file_types parameter to the index command and a index_dir parameter to search_dataset, so you can build and query them independently.

Alternatively, the quick fix to just pass the docs evaluation: in ingestion.py, only index .md and .txt files (remove the .py branch), increase chunk size to 1500, re-index, and re-run. That removes the Python noise and fixes problem 1. You can then address the code evaluation separately.

Want me to implement the two-index approach, or the quick single-fix?



=========generation changes======

### 1. `AutoTokenizer` (The Translator)

The Neural Network (the model) does not understand English. It only understands math and numbers. The Tokenizer's job is to act as a two-way dictionary between human words and mathematical IDs.

* **`from_pretrained("Model-Name")`**: Downloads the specific "dictionary" for Qwen. Every AI company uses a different dictionary (e.g., OpenAI's token for "Hello" might be 402, but Qwen's might be 8910).
* **`apply_chat_template(messages, ...)`**: This is the most powerful feature of modern tokenizers. If you pass it a list of dictionaries (System/User messages), it automatically formats them into the strict syntax the model was trained on. 
  * *For example, it takes your `messages` array and converts it into:* `<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nWhat is vLLM?<|im_end|>\n<|im_start|>assistant\n`
  * *Because we set `enable_thinking=False`, it explicitly strips out any hidden `<think>` tags during this formatting phase.*
* **`decode(token_ids)`**: The reverse process. When the model spits out an array of numbers like `[8910, 342, 11]`, the tokenizer translates it back into English ("The answer is").

---

### 2. `AutoModelForCausalLM` (The Brain)

This is the actual 500-million parameter Neural Network. "Causal LM" stands for Causal Language Modeling, which is the scientific term for "predicting the next word based on the previous words."

* **`from_pretrained("Model-Name")`**: Loads the massive weights/matrices into memory.
  * *Why `torch_dtype=torch.float16`?* AI models are usually trained in 32-bit float (which is huge). Loading it in 16-bit cuts the RAM usage in half and makes it run twice as fast on your Mac's GPU, with almost zero loss in intelligence.
  * *Why `.to("mps")`?* Moves those weights from your Mac's standard RAM directly into the Apple Silicon GPU's high-speed memory.
* **`eval()`**: Puts the model in "Evaluation Mode." By default, PyTorch models are ready to be trained (which requires tracking gradients and keeping massive memory buffers open). `eval()` locks the weights so the model only reads data, saving massive amounts of memory.
* **`generate(inputs)`**: The core inference loop. You hand it an array of input tokens. It looks at them, does the math, predicts the very next token, adds it to the list, and repeats the process until it hits a stop token or reaches your `max_new_tokens` limit.

---

### The Three-Step Workflow (Summary)

Whenever you use raw `transformers`, you are always executing this exact 1-2-3 dance:

1. **Encode (Tokenizer):** Take the `messages` array -> format it -> convert it to PyTorch tensors (numbers).
2. **Inference (Model):** Feed tensors into `.generate()` -> get a longer array of tensors back.
3. **Decode (Tokenizer):** Take the new tensors -> convert them back to an English string.