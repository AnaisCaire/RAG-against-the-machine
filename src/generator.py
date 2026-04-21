from typing import List
from transformers import pipeline
from src.models import MinimalSource, MinimalAnswer


class Generator:
    def __init__(self):
        self.pipeline = pipeline("text-generation",
                                 model="Qwen/Qwen3-0.6B")

    def _build_prompt(self,
                      query: str,
                      retrieved_sources: List[MinimalSource]) -> str:
        """
        Constructs the prompt string for the LLM.
        """

        prompt = ("Please answer the user's question based ONLY"
                  "on the following context:\n\n")

        # 1. Loop through the retrieved sources
        for source in retrieved_sources:
            try:
                # 2. Extract the actual text for this source
                with open(source.file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    chunk_text = content[source.first_character_index:
                                        source.last_character_index]

                prompt += f"File: {source.file_path}\nCode:\n{chunk_text}\n\n"
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
        Generates an answer using the LLM and
        returns a strictly typed response.
        """

        # 1. Build the prompt
        prompt = self._build_prompt(query, retrieved_sources)

        # 2. Ask the AI to generate text
        # We limit the tokens to prevent it from rambling forever
        print("Generating answer...")
        ai_output = self.pipeline(prompt, max_new_tokens=256, return_full_text=False)
        breakpoint()
        # 3. Extract the raw string from the pipeline output
        raw_answer_string = ai_output[0]['generated_text'].strip()

        # 4. Package everything into the MinimalAnswer Pydantic model
        return MinimalAnswer(
            question_id=question_id,
            question=query,
            retrieved_sources=retrieved_sources,
            answer=raw_answer_string)
