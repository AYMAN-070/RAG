import fire


class RAGCLI:
    """Interface en Ligne de Commande (CLI) pour le système RAG."""

    def index(self, max_chunk_size: int = 2000) -> None:
        """
        Index the repository.
        """
        print(f"Starting indexation (max_chunk_size={max_chunk_size})...")
        from .ingestion import DocumentProcessor
        processor = DocumentProcessor(max_chunk_size=max_chunk_size)
        # Ensure raw data is extracted, for example in 'data/raw'
        docs = processor.load_directory("data/raw")
        print(f"Found {len(docs)} files to process!")
        chunks = processor.process_documents(docs)
        print(f"Created {len(chunks)} chunks! Building index...")
        from .retrieval import Retriever
        retriever = Retriever()
        retriever.build_and_save_index(chunks)

    def search(self, query: str, k: int = 5) -> None:
        """
        Search for a single query using the BM25 index.
        """
        print(f"\nSearching for: '{query}' (top {k})...")
        from .retrieval import Retriever
        try:
            # Wake up the Retriever
            retriever = Retriever()
            # Load the saved BM25 index
            retriever.load_index()
            # Search for the best text chunks
            results = retriever.search(query, k=k)
            # Check if anything was found
            if not results.retrieved_sources:
                print("No relevant documents were found.")
                return
            # 5. Display a clean report in the terminal
            print("\n" + "="*50)
            print(f"SEARCH RESULTS (Top {k})")
            print("="*50)
            for i, source in enumerate(results.retrieved_sources, 1):
                print(f"Result {i}:")
                print(f"File: {source.file_path}")
                print(f"Position: Characters {source.first_character_index} "
                      f"to {source.last_character_index}")
                print("-" * 40)
            print("\n")
        except FileNotFoundError:
            # Safety catch: If the user forgot to run 'index' first
            print("Error: BM25 index not found.")
            print("Tip: Run the 'index' command first to analyze the files")

    def search_dataset(self, dataset_path: str, save_directory: str,
                       k: int = 10) -> None:
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
        print(f"Launch of the search for {len(dataset.rag_questions)} "
              f"questions...")
        # 4. CREAT chargingn bar
        for item in tqdm(dataset.rag_questions, desc="Seaching..."):
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
        print(f"\nEnd searching! Results saved in {save_path}")

    def answer(self, query: str, k: int = 5) -> None:
        """
        Generat an answer using AI and Documents.
        """
        print(f"\nQuestion  : '{query}'")
        print("1. Looking for documents (BM25)...")
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

    def answer_dataset(self, student_search_results_path: str,
                       save_directory: str) -> None:
        """
        Generate answers from search results.
        """
        import os
        import json
        from tqdm import tqdm
        from .generation import AnswerGenerator
        from .models import StudentSearchResults
        print(f"Loading results from {student_search_results_path}...")
        with open(student_search_results_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
        search_dataset = StudentSearchResults(**raw_data)
        generator = AnswerGenerator()
        final_answers = []
        total_q = len(search_dataset.search_results)
        print(f"Launching generation for {total_q} questions...")
        # The magic loop with the tqdm progress bar
        for search_result in tqdm(search_dataset.search_results,
                                  desc="Generating answers"):
            answer = generator.generate(search_result)
            # We append the generated answer to our final list
            final_answers.append(answer.model_dump())
        os.makedirs(save_directory, exist_ok=True)
        base_name = os.path.basename(student_search_results_path)
        filename = base_name.replace("search_results", "answers")
        save_path = os.path.join(save_directory, f"final_answers_{filename}")
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(final_answers, f, indent=2, ensure_ascii=False)
        print(f"\nDONE! All answers are saved in {save_path}")

    def calculate_overlap_percentage(self, retrieved_start: int,
                                     retrieved_end: int,
                                     correct_start: int,
                                     correct_end: int) -> float:
        """Calculates overlap percentage between retrieved chunk and truth."""
        # find intersection of two segments
        overlap_start = max(retrieved_start, correct_start)
        overlap_end = min(retrieved_end, correct_end)
        # Calculat length of overlap
        overlap_length = max(0, overlap_end - overlap_start)
        # Calculat length of main source
        correct_length = correct_end - correct_start
        if correct_length == 0:
            return 0.0
        # Return percentage
        return overlap_length / correct_length

    def evaluate(self, student_answer_path: str, dataset_path: str,
                 k: int = 10, max_context_length: int = 2000) -> None:
        """
        Evaluate search results against ground truth.
        """
        import os
        import json
        print("\nEvaluating results...")
        print(f"Analyzing file: {student_answer_path}")
        # We use the variables here to satisfy Flake8
        print(f"Reference: {dataset_path} | k={k} | "
              f" max_len={max_context_length}")
        if not os.path.exists(student_answer_path):
            print("Error: Answer file not found. "
                  "Launch 'answer_dataset' first.")
            return
        with open(student_answer_path, "r", encoding="utf-8") as f:
            answers = json.load(f)
        with open(dataset_path, "r", encoding="utf-8") as f:
            correct_dataset = json.load(f)
            correct_questions = correct_dataset.get("rag_questions",
                                                    correct_dataset)
        total_recall_score = 0.0
        total_questions = len(answers)
        valid_answers = 0
        total_length = 0
        hallucination_suspicions = 0
        for item in answers:
            answer_text = item.get("answer", "")
            if answer_text and len(answer_text.strip()) > 5:
                valid_answers += 1
                total_length += len(answer_text)
                is_hallucinating = (
                    "I don't know" in answer_text or
                    "not in the context" in answer_text
                )
                if is_hallucinating:
                    hallucination_suspicions += 1
            q_id = item.get("question_id")
        # We break down math calculations to avoid long lines
        correct_q = next((q for q in correct_questions
                          if q.get("question_id") == q_id), None)
        if correct_q:
            correct_sources = correct_q.get("sources", [])
            retrieved_sources = item.get("retrieved_sources", [])
            sources_found = 0

            for correct_src in correct_sources:
                for ret_src in retrieved_sources:
                    # if same file we verify overlap
                    ret_path = ret_src.get("file_path")
                    correct_path = correct_src.get("file_path")
                    if ret_path and ret_path == correct_path:
                        overlap = self.calculate_overlap_percentage(
                            ret_src.get("first_character_index", 0),
                            ret_src.get("last_character_index", 0),
                            correct_src.get("first_character_index", 0),
                            correct_src.get("last_character_index", 0)
                        )
                        if overlap >= 0.05:
                            sources_found += 1
                            break
            if len(correct_sources) > 0:
                total_recall_score += (sources_found / len(correct_sources))
        # We break down math calculations to avoid long lines
        if total_questions > 0:
            success_rate = (valid_answers / total_questions) * 100
            final_recall = (total_recall_score / total_questions) * 100
        else:
            success_rate = 0
            final_recall = 0
        if valid_answers > 0:
            avg_length = total_length // valid_answers
        else:
            avg_length = 0
        print("\n" + "="*50)
        print("RAG SYSTEM EVALUATION REPORT")
        print("="*50)
        print(f"Processed questions     : {total_questions}")
        print(f"Valid answers generated : {valid_answers} "
              f"({success_rate:.1f}%)")
        print(f"Average length/answer   : {avg_length} characters")
        print(f"Unanswered questions    : {hallucination_suspicions} "
              "(AI admitted not knowing)")
        print("-" * 50)
        print(f"SYSTEM RECALL SCORE  : {final_recall:.1f}%")
        print("="*50)
        if success_rate == 100:
            print("Excellent! Your system successfully answered "
                  "all questions.")
        else:
            print("Warning: Some questions did not "
                  "get a valid answer.")
        print("\n")


if __name__ == '__main__':
    fire.Fire(RAGCLI)
