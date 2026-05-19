import os
from typing import List, Dict, Any
from .models import MinimalSource


class DocumentProcessor:
    def __init__(self, max_chunk_size: int = 2000):
        self.max_chunk_size = max_chunk_size

    def load_directory(self, directory_path: str) -> List[Dict[str, Any]]:
        """
        Walks through the directory and loads the content of .py and .md files.
        """
        documents = []
        for root, _, files in os.walk(directory_path):
            for file in files:
                if file.endswith(('.py', '.md')):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        documents.append({
                            "file_path": file_path,
                            "content": content,
                            "is_python": file.endswith('.py')
                        })
                    except Exception as e:
                        print(f"Error reading file {file_path}: {e}")
        return documents

    def _smart_chunk(self, text: str, file_path: str,
                     separators: List[str]) -> List[Dict[str, Any]]:
        """
        Helper method to chunk text based on a list of prioritized separators.
        It tries to find the best place to cut without exceeding max_chunksize.
        """
        chunks = []
        start_idx = 0
        text_length = len(text)

        while start_idx < text_length:
            # Define the maximum reach for this chunk
            end_idx = start_idx + self.max_chunk_size
            # If we reach the end of the text, just take the rest
            if end_idx >= text_length:
                end_idx = text_length
            else:
                best_split_idx = -1
                for sep in separators:
                    window = text[start_idx:end_idx]
                    sep_idx = window.rfind(sep)
                    if sep_idx != -1:
                        # Found a separator! We cut just after it.
                        best_split_idx = start_idx + sep_idx + len(sep)
                        break
                if best_split_idx != -1 and best_split_idx > start_idx:
                    end_idx = best_split_idx
            chunk_content = text[start_idx:end_idx]
            # Avoid appending empty chunks
            if chunk_content.strip():
                chunks.append({
                    "content": chunk_content,
                    "metadata": MinimalSource(
                        file_path=file_path,
                        first_character_index=start_idx,
                        last_character_index=end_idx
                    )
                })
            start_idx = end_idx
        return chunks

    def chunk_text(self, text: str, file_path: str) -> List[Dict[str, Any]]:
        """
        Chunking strategy for Markdown/Text.
        Prioritizes splitting by double newline, single newline, period, then space.
        """
        separators = ["\n\n", "\n", ". ", " "]
        return self._smart_chunk(text, file_path, separators)

    def chunk_python(self, text: str, file_path: str) -> List[Dict[str, Any]]:
        """
        Chunking strategy for Python code.
        Prioritizes splitting before classes, functions, or newlines.
        """
        separators = ["\nclass ", "\ndef ", "\n\n", "\n"]
        return self._smart_chunk(text, file_path, separators)

    def process_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        all_chunks = []
        for doc in documents:
            if doc["is_python"]:
                chunks = self.chunk_python(doc["content"], doc["file_path"])
            else:
                chunks = self.chunk_text(doc["content"], doc["file_path"])
            all_chunks.extend(chunks)
        return all_chunks
