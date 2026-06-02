import os
import json
import bm25s
from typing import List, Dict, Any
from .models import MinimalSource, MinimalSearchResults


class Retriever:
    def __init__(self, save_dir: str = "data/processed"):
        """
        save_dir is where we store the index and the chunks on the hard drive.
        """
        self.save_dir = save_dir
        self.index: Any = None
        self.chunks_data: List[Dict[str, Any]] = []

    def build_and_save_index(self, chunks: List[Dict[str, Any]]) -> None:
        """
        Builds the BM25 index from our chunks and saves it to disk.
        """
        print("Tokenizing the text (breaking it into words)...")
        # Extract just the text from our chunks
        corpus = [chunk["content"] for chunk in chunks]
        self.chunks_data = chunks
        # Tokenize (cut into words) using the BM25 library
        corpus_tokens = bm25s.tokenize(corpus)
        # Create scores for each word
        print("Building the BM25 index...")
        self.index = bm25s.BM25()
        self.index.index(corpus_tokens)
        # Save the dict of scores in data/processed
        os.makedirs(self.save_dir, exist_ok=True)
        self.index.save(f"{self.save_dir}/bm25_index")
        # save chunks (text + metadata) to retrieve them later
        chunks_path = os.path.join(self.save_dir, "chunks_data.json")
        # Convert MinimalSource objects to dict so we can save them in JSON
        serialized_chunks = []
        for c in chunks:
            serialized_chunks.append({
                "content": c["content"],
                "metadata": c["metadata"].model_dump()
            })
        with open(chunks_path, "w", encoding="utf-8") as f:
            json.dump(serialized_chunks, f)
        print(f"Index successfully saved in {self.save_dir}!")

    def load_index(self) -> None:
        """
        Loads the previously saved index and chunks from the disk.
        """
        index_path = os.path.join(self.save_dir, "bm25_index")
        chunks_path = os.path.join(self.save_dir, "chunks_data.json")

        if not os.path.exists(index_path) or not os.path.exists(chunks_path):
            raise FileNotFoundError("Index not found. Please run 'index' "
                                    "command first.")
        # Load the BM25 index
        self.index = bm25s.BM25.load(index_path, load_corpus=False)
        # Load the chunks data
        with open(chunks_path, "r", encoding="utf-8") as f:
            loaded_chunks = json.load(f)
        # Reconstruct the MinimalSource Pydantic objects
        self.chunks_data = []
        for c in loaded_chunks:
            self.chunks_data.append({
                "content": c["content"],
                "metadata": MinimalSource(**c["metadata"])
            })

    def search(self, query: str, k: int = 5) -> MinimalSearchResults:
        """
        Searches the index for the query and returns the top k results.
        """
        if self.index is None:
            self.load_index()
        query_tokens = bm25s.tokenize(query)
        # Retrieve the top 'k' most relevant chunks(best proba score)
        results_index, _ = self.index.retrieve(query_tokens, corpus=None, k=k)
        top_indices = results_index[0]
        # Fetch the actual metadata for these top results
        retrieved_sources = []
        for idx in top_indices:
            chunk = self.chunks_data[idx]
            retrieved_sources.append(chunk["metadata"])

        # Format the output using our Pydantic model
        return MinimalSearchResults(
            question_id="temp_id",
            question_str=query,
            retrieved_sources=retrieved_sources
        )
