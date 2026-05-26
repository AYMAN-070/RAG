from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from .models import MinimalSearchResults, MinimalAnswer


class AnswerGenerator:
    def __init__(self, model_name: str = "Qwen/Qwen3-0.6B"):
        """
        Initialise IA model.
        """
        print(f"Chargement du modèle LLM : {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name,
                                                          device_map="auto")
        # creat a "pipeline" to generat text
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=200,
            return_full_text=False
        )

    def _build_prompt(self, question: str, context: str) -> str:
        """
        IA instructions """
        prompt = (
            "You are an expert coding assistant. Answer the user's question"
            "based ONLY on the provided context. If the answer is not in the "
            "context, say 'I don't know'.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            "Answer:"
        )
        return prompt

    def generate(self, search_results: MinimalSearchResults,
                 max_context_length: int = 2000) -> MinimalAnswer:
        """
        Take results and creat a text.
        """
        context_parts = []
        current_length = 0
        for source in search_results.retrieved_sources:
            text_snippet = f"File: {source.file_path} - Content goes here..."
            if current_length + len(text_snippet) > max_context_length:
                break
            context_parts.append(text_snippet)
            current_length += len(text_snippet)
        full_context = "\n---\n".join(context_parts)
        prompt = self._build_prompt(question=search_results.question,
                                    context=full_context)
        print(f"L'IA réfléchit à la question : '{search_results.question}'...")
        output = self.generator(prompt)
        generated_text = output[0]["generated_text"].strip()
        return MinimalAnswer(
            question_id=search_results.question_id,
            question=search_results.question,
            retrieved_sources=search_results.retrieved_sources,
            answer=generated_text
        )
