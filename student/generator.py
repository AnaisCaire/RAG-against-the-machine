from typing import List
from transformers import pipeline
from student.models import MinimalSource, MinimalAnswer
import torch


class Generator:
    def __init__(self):
        device = "mps" if torch.backends.mps.is_available() else "cpu"

        if device == "mps":
            print("⚡ Accelerating with Apple Silicon GPU (MPS)")
        else:
            print("🐢 GPU not found. Falling back to CPU (Slow)")

        self.pipeline = pipeline(
            "text-generation",
            model="Qwen/Qwen2.5-0.5B-Instruct",
            device=device,        # <--- THIS IS THE SPEED KEY
            torch_dtype=torch.float16,  # <--- USES HALF-PRECISION (Faster & Less RAM)
            batch_size=8
        )

    def _build_prompt(self,
                      query: str,
                      retrieved_sources: List[MinimalSource]) -> str:
        """
        Constructs the prompt string for the LLM.
        """

        prompt = ("Please answer the user's question based ONLY"
                  "on the following context:\n\n")

        # ---- optimisation ----
        # control the token limit for a smaller prompt < 2s.
        max_content_chars = 4000
        current_char = 0

        # 1. Loop through the retrieved sources
        for source in retrieved_sources:
            try:
                # 2. Extract the actual text for this source
                with open(source.file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    chunk_text = content[source.first_character_index:
                                         source.last_character_index]

                chunk_str = f"--- SOURCE FILE: {source.file_path} ---\n{chunk_text}\n\n"

                if current_char + len(chunk_str) > max_content_chars:
                    break

                prompt += chunk_str
                current_char += len(chunk_str)
            except Exception as e:
                print(f"[Error]: could not read {source.file_path}"
                      f"for generation : {e}")

        prompt += f"Question: {query}\nAnswer:"
        return prompt

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

        # 1. Build the prompt

        prompt = self._build_prompt(query, retrieved_sources)
        # 2. We structure the prompt as a conversation!
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a highly precise technical assistant. "
                    "Answer the user's question directly and concisely. "
                    "You MUST base your answer ONLY on the provided context. "
                    "You MUST cite the source of your answer by putting the file path in brackets at the end, like: [Source: path/to/file.py]. "
                    "Do not write more than 2 or 3 sentences."
                    "Do NOT show your thinking process. Do NOT use LaTeX or boxed formatting."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
        # 3. Ask the AI to generate text
        # We limit the tokens to prevent it from rambling forever
        ai_output = self.pipeline(
            messages,
            max_new_tokens=75,
            return_full_text=False,
            truncation=True,  # for GPU mem
            do_sample=False,  # for faster answer
            repetition_penalty=1.2,
            max_length=1024  # Cap the max input math the GPU has to do
        )
        # 4. Extract the raw string from the pipeline output
        raw_answer_string = ai_output[0]['generated_text'].strip()
        if "</think>" in raw_answer_string:
            raw_answer_string = raw_answer_string.split("</think>")[-1].strip()
        # 4. Package everything into the MinimalAnswer Pydantic model
        return MinimalAnswer(
            question_id=question_id,
            question=query,
            retrieved_sources=retrieved_sources,
            answer=raw_answer_string)
