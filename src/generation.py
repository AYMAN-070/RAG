from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from .models import MinimalSearchResults, MinimalAnswer


class AnswerGenerator:
    def __init__(self, model_name: str = "Qwen/Qwen3-0.6B"):
        """
        Initialise IA model.
        """
        print(f"Charging LLM model: {model_name}...")
        # Tokenizer make IA understand words by split them into numbers
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # make IA prupose to guess rh next word, device_map to use GPU
        self.model = AutoModelForCausalLM.from_pretrained(model_name,
                                                          device_map="auto")
        # creat a "pipeline" to generat text
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=150,
            clean_up_tokenization_spaces=False,
            return_full_text=False,
            repetition_penalty=1.15
        )

    def _build_prompt(self, question: str, context: str) -> str:
        """
        IA instructions """
        prompt = (
            "You are an expert coding assistant."
            "You MUST strictly follow these 2 rules:\n"
            "1. Answer the question based ONLY on the provided context.\n"
            "2. Your answer MUST be self-contained " 
            "(restate the subject of the question in your first sentence).\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            "Self-contained Answer:"
        )
        return prompt

    def generate(self, search_results: MinimalSearchResults,
                 max_context_length: int = 12000) -> MinimalAnswer:
        """
        Take results and creat a text.
        """
        context_parts = []
        current_length = 0
        for source in search_results.retrieved_sources:
            try:
                with open(source.file_path, "r", encoding="utf-8") as f:
                    full_content = f.read()
                    actual_text = full_content[
                        source.first_character_index:
                        source.last_character_index
                        ]
                    text_snippet = (f"Source ({source.file_path}):\n"
                                    f"{actual_text}\n")
            except Exception:
                text_snippet = f"Source ({source.file_path}): can't read.\n"
            if current_length + len(text_snippet) > max_context_length:
                break
            context_parts.append(text_snippet)
            current_length += len(text_snippet)
        full_context = "\n---\n".join(context_parts)
        prompt = self._build_prompt(question=search_results.question_str,
                                    context=full_context)
        print(f"L'IA reading documents and analyse the question : "
              f"'{search_results.question_str}'...")
        output = self.generator(prompt)
        generated_text = output[0]["generated_text"].strip()
        if '"""' in generated_text:
            generated_text = generated_text.split('"""')[0]
        # if IA add "Answer: " in the beginning, we delete it
        if generated_text.startswith("Answer:"):
            generated_text = generated_text.replace("Answer:", "").strip()
        if "**Answer**" in generated_text:
            generated_text = generated_text.split("**Answer**")[-1]
        if "</think>" in generated_text:
            generated_text = generated_text.split("</think>")[-1]
        if "The answer is:" in generated_text:
            generated_text = generated_text.split("The answer is:")[0].strip()
        generated_text = generated_text.strip()
        return MinimalAnswer(
            question_id=search_results.question_id,
            question_str=search_results.question_str,
            retrieved_sources=search_results.retrieved_sources,
            answer=generated_text
        )
