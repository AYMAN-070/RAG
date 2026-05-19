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
        print(f"Recherche sur le dataset {dataset_path}...")

    def answer(self, query: str, k: int = 5) -> None:
        """
        Answer a single question with context.
        """
        print(f"Génération de la réponse pour : '{query}'...")

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