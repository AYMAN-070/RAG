*This project has been created as part of the 42 curriculum by ayhammou.*

# RAG Against The Machine

## Description
This project implements a complete Retrieval-Augmented Generation (RAG) system designed to answer questions about a specific codebase (the vLLM repository). It ingests raw Python and Markdown files, indexes them, retrieves the most relevant context using BM25, and leverages a Large Language Model (Qwen/Qwen3-0.6B) to generate accurate, source-grounded answers formatted strictly in JSON using Pydantic.

## System Architecture
The RAG pipeline is divided into four main decoupled components:
1. **Ingestion (`ingestion.py`):** Reads the raw dataset, cleans it, and applies a chunking strategy.
2. **Retrieval (`retrieval.py`):** Encodes the chunks and builds a mathematical index to perform semantic search.
3. **Generation (`generation.py`):** Loads the Qwen LLM, injects the top-K retrieved sources into a strict prompt, and generates the final answer.
4. **Evaluation (`evaluate` in CLI):** Compares the AI's generated answers against the Ground Truth to compute Recall@k metrics and hallucination rates.

## Chunking Strategy
The system processes Python (`.py`) and Markdown (`.md`) files distinctly. We use an intelligent chunking strategy that splits documents into blocks with a maximum size of 2000 characters (configurable via CLI). To maintain semantic integrity, the chunker avoids cutting words in half by searching for natural separators like spaces or newline characters before breaking the text.

## Retrieval Method
We implemented the BM25 (Best Matching 25) algorithm as our core retrieval method.
Code-specific optimization (snake_case hack): BM25 struggles with Python code because it treats snake_case or CamelCase variables as single tokens. To solve this, our pre-processor extracts snake_case words, splits them into separate words (e.g., calculate_overlap -> calculate overlap), and injects them as hidden keywords at the end of the chunk. We also inject the file path as keywords. This technique allowed us to reach maximum performance (5 stars) on code retrieval.

## Performance Analysis
The system was evaluated using a strict Recall@k metric. 
- **Indexing time:** Extremely fast (under 1 minute), well below the 5-minute limit.
- **Recall@5 Performance:** The BM25 algorithm successfully achieves >80% recall on documentation questions and >50% on code-specific questions, ensuring a highly relevant context window for the LLM.

## Design Decisions
- **Strict Typing:** All data flowing between the Ingestion, Retrieval, and Generation modules is strictly validated using `Pydantic` models (`MinimalSource`, `MinimalAnswer`, etc.) to prevent data corruption.
- **CLI Framework:** We used Google's `fire` library to expose the Python classes seamlessly to the command line.
- **Makefile:** A Makefile is included to automate the `uv` virtual environment workflows, making testing and execution highly streamlined.

## ⚠️ Important: Handling "No space left on device" Errors

This project downloads large dependencies (PyTorch, CUDA) and a ~1.5GB language model (Qwen). In constrained environments with strict `/home/` quotas (like 42 school iMacs), the installation will crash with a `No space left on device` error.

To fix this, **redirect the cache to your external storage (`sgoinfre`)** before running any commands. 

Run these lines in your terminal:

# 1. Create cache directories in sgoinfre
mkdir -p /sgoinfre/$USER/.uv_cache
mkdir -p /sgoinfre/$USER/.huggingface_cache

# 2. Export variables to redirect the downloads
export UV_CACHE_DIR="/sgoinfre/$USER/.uv_cache"
export HF_HOME="/sgoinfre/$USER/.huggingface_cache"

## Bonus Features Implemented
To push the system's performance further, we implemented the following advanced RAG features:

1. Index Caching: The BM25 index is automatically saved to disk in the data/processed directory after creation. Subsequent queries load the pre-built index, resulting in near-instantaneous search speeds (0s throughput).

2. Query Expansion: We implemented a query expansion mechanism that enriches the user's prompt with synonyms (e.g., start -> launch run init). Note: This feature is behind a use_expansion=False flag by default, as empirical testing with the Moulinette showed that BM25 prefers exact keyword matching to avoid Query Drift.

## Challenges Faced
- **Disk Space / Environment Setup:** Installing the heavy Machine Learning dependencies (`torch`, `triton`) initially caused `os error 28 (No space left on device)` in the standard `/home` directory. This was solved by configuring the `uv` cache to target the `sgoinfre` partition and utilizing CPU-only PyTorch wheels for local development and testing.
- **Context Length Limits:** Balancing the `k` parameter (number of
