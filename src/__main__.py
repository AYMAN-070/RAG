import fire
from typing import Optional

class RAGCLI:
    """Interface en Ligne de Commande (CLI) pour le système RAG."""

    def index(self, max_chunk_size: int = 2000) -> None:
        """
        Index the repository.
        """
        print(f"Starting indexation (max_chunk_size={max_chunk_size})...")
        from .ingestion import DocumentProcessor
        processor = DocumentProcessor(max_chunk_size=max_chunk_size)
        # Ensure your raw data is extracted, for example in 'data/raw'
        docs = processor.load_directory("data/raw") 
        print(f"Found {len(docs)} files to process!")

    def search(self, query: str, k: int = 5) -> None:
        """
        Search for a single query.
        """
        print(f"Recherche en cours pour : '{query}' (top {k})...")

    def search_dataset(self, dataset_path: str, save_directory: str, k: int = 10) -> None:
        """
        Process multiple questions and output search results.
        """
        import os
        import json
        from tqdm import tqdm
        from .retrieval import Retriever
        from .models import RagDataset, StudentSearchResults
        print(f"Charging dataset from {dataset_path}...")
        # Open JSON file for questions
        with open(dataset_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
        # Validat all dataset !
        dataset = RagDataset(**raw_data)
        # Préparation du moteur de recherche
        retriever = Retriever()
        retriever.load_index()
        all_results = []
        print(f"Launch of the search for {len(dataset.rag_questions)} questions...")
        # 4. CREAT chargingn bar
        for item in tqdm(dataset.rag_questions, desc="Recherche en cours"):
            result = retriever.search(item.question, k=k)
            result.question_id = item.question_id
            all_results.append(result)
        final_output = StudentSearchResults(
            search_results=all_results,
            k=k
        )
        # Save results
        os.makedirs(save_directory, exist_ok=True)
        filename = os.path.basename(dataset_path)
        save_path = os.path.join(save_directory, filename)
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(final_output.model_dump_json(indent=2))
        print(f"\nRecherche terminée ! Résultats sauvegardés dans {save_path}")

    def answer(self, query: str, k: int = 5) -> None:
        """
        Generat an answer using AI and Documents.
        """
        print(f"\nQuestion posée : '{query}'")
        print("1. Recherche des documents en cours (BM25)...")
        from .retrieval import Retriever
        from .generation import AnswerGenerator
        retriever = Retriever()
        search_results = retriever.search(query, k=k)
        # Verify if we found something
        if not search_results.retrieved_sources:
            print("No relevant documents were found to answer this question.")
            return
        print(f"-> {len(search_results.retrieved_sources)} documents found !")
        print("2. Generating response (Qwen)...")
        # L'Expert (l'IA) prend le relais
        generator = AnswerGenerator()
        final_answer = generator.generate(search_results)
        print("\n" + "="*50)
        print("AI ANSWER :")
        print("="*50)
        print(final_answer.answer)
        print("\n" + "-"*50)
        print("SOURCES USED :")
        for i, source in enumerate(final_answer.retrieved_sources, 1):
            print(f"{i}. File : {source.file_path} (Lines : "
                  f"{source.first_character_index} "
                  f"à {source.last_character_index})")
        print("-"*50 + "\n")

    def answer_dataset(self, student_search_results_path: str, save_directory: str) -> None:
        """
        Generate answers from search results.
        """
        print(f"Génération des réponses pour le dataset {student_search_results_path}...")

    def evaluate(self, student_answer_path: str, dataset_path: str, k: int = 10, max_context_length: int = 2000) -> None:
        """
        Evaluate search results against ground truth.
        """
        print("Évaluation des résultats en cours...")

if __name__ == '__main__':
    fire.Fire(RAGCLI)