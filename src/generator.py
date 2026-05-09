import os
from typing import Any, List
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.models import MinimalSource, MinimalAnswer
from src.config import (
    OMP_NUM_THREADS,
    LLM_MODEL_NAME,
    LLM_DTYPE,
    MAX_NEW_TOKENS_EXPAND,
    MAX_NEW_TOKENS_ANSWER,
    ENABLE_THINKING,
    MAX_CONTEXT_CHARS,
    MAX_CONTEXT_SOURCES,
)
import torch

os.environ["OMP_NUM_THREADS"] = OMP_NUM_THREADS


class Generator:
    """
    Wraps a local Qwen LLM for query expansion and RAG answer generation.
    """

    def __init__(self) -> None:
        """
        loads the Qwen model and tokenizer on best available device.
        """
        self.device: str = (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )

        print(f"\nDevice selected: {self.device}\n")

        self.tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
        self.model: Any = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL_NAME,
            dtype=LLM_DTYPE,
        ).to(self.device)  # type: ignore[arg-type]
        self.model.eval()
        self.answer_cache: dict[str, str] = {}
        self._file_cache: dict[str, str] = {}

    def _build_prompt(self,
                      query: str,
                      retrieved_sources: List[MinimalSource]) -> str:
        """
        Constructs the prompt string for the LLM.
        """

        prompt = ("Please answer the user's question based ONLY"
                  "on the following context:\n\n")

        max_content_chars = MAX_CONTEXT_CHARS
        current_char = 0

        for source in retrieved_sources[:MAX_CONTEXT_SOURCES]:
            try:
                if source.file_path not in self._file_cache:
                    with open(source.file_path, 'r', encoding='utf-8') as f:
                        self._file_cache[source.file_path] = f.read()
                content = self._file_cache[source.file_path]
                chunk_text = content[source.first_character_index:
                                     source.last_character_index]

                chunk_str = (f"--- SOURCE FILE: {source.file_path}"
                             f"---\n{chunk_text}\n\n")

                if current_char + len(chunk_str) > max_content_chars:
                    break

                prompt += chunk_str
                current_char += len(chunk_str)
            except Exception as e:
                print(f"[Error]: could not read {source.file_path}"
                      f"for generation : {e}")

        prompt += f"Question: {query}\nAnswer:"
        return prompt

    def expand_query(self, query: str) -> str:
        """
        make the LLM rewrite the query with more keywords
        """
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a search engine keyword generator. "
                    "Extract the core technical concepts from the user's query and generate likely Python "
                    "variable names, class names, or function names (using snake_case or CamelCase). "
                    "Output ONLY a space-separated list of keywords. Do not explain or write sentences."
                )
            },
            {
                "role": "user",
                "content": query
            }
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=ENABLE_THINKING
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS_EXPAND,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )

        input_length = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0][input_length:]
        decoded = self.tokenizer.decode(generated_tokens,
                                        skip_special_tokens=True)
        expanded_keywords = decoded if isinstance(decoded, str) else decoded[0]
        expanded_keywords = expanded_keywords.strip()

        if "</think>" in expanded_keywords:
            expanded_keywords = expanded_keywords.split("</think>")[-1].strip()

        return f"{query} {expanded_keywords}"

    def generate_answer(self,
                        question_id: str,
                        query: str,
                        retrieved_sources: List[MinimalSource]) -> MinimalAnswer:
        """
        Augmenting phase: take the text from the chunks the Retreiver
        found and glue them together into a string
        Generates an answer using the LLM and
        returns a strictly typed response.
        """

        if query in self.answer_cache:
            return MinimalAnswer(
                question_id=question_id,
                question_str=query,
                retrieved_sources=retrieved_sources,
                answer=self.answer_cache[query]
            )

        prompt = self._build_prompt(query, retrieved_sources)
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a highly precise technical assistant. "
                    "Answer the user's question directly and concisely. "
                    "You MUST base your answer ONLY on the provided context. "
                    "You MUST cite the source of your answer by putting the file path "
                    "in brackets at the end, like: [Source: path/to/file.py]. "
                    "Do not write more than 2 or 3 sentences."
                    "Do NOT show your thinking process. Do NOT use LaTeX or boxed formatting."
                    "You MUST cite the source of your answer using the exact SOURCE FILE path shown "
                    "in the context above, formatted as [Source: path/to/file]. "
                    "Never invent file paths."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=ENABLE_THINKING
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        # Generate (no_grad = no memory tracking)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS_ANSWER,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )
        # Decode only the newly generated tokens
        input_length = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0][input_length:]

        decoded = self.tokenizer.decode(generated_tokens,
                                        skip_special_tokens=True)
        raw_answer_string = decoded if isinstance(decoded, str) else decoded[0]
        raw_answer_string = raw_answer_string.strip()
        # Fallback cleanup just in case the model ignores the instruction
        if "</think>" in raw_answer_string:
            raw_answer_string = raw_answer_string.split("</think>")[-1].strip()
        self.answer_cache[query] = raw_answer_string
        # Package everything into the MinimalAnswer Pydantic model
        return MinimalAnswer(
            question_id=question_id,
            question_str=query,
            retrieved_sources=retrieved_sources,
            answer=raw_answer_string)
