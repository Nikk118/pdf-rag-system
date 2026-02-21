# Local PDF RAG System (CLI)

A Command-Line Interface (CLI) based PDF Question Answering system built using Retrieval-Augmented Generation (RAG). This tool allows you to ingest PDF documents and ask questions directly from the terminal. The system retrieves relevant context using semantic search and generates accurate answers using a local language model.

This project runs completely locally and does not require external APIs.

---

## Features

* CLI-based interface (no web UI required)
* Ask questions directly from PDF documents
* Semantic search using FAISS vector database
* Context-aware answer generation using local LLM
* Fast and efficient embedding using MiniLM
* Conversation memory support
* Fully local execution (no internet required after setup)

---

## Tech Stack

* Python
* Sentence Transformers (MiniLM)
* FAISS (Vector Database)
* Transformers (TinyLlama)
* NumPy

---

## Project Structure

```
local-pdf-rag/
│
├── config.py
├── ingest.py
├── query.py
├── vector_store/
├── requirements.txt
└── README.md
```

---

## Installation

### 1. Clone the repository

```
git clone https://github.com/Nikk118/local-pdf-rag.git
cd local-pdf-rag
```

### 2. Create virtual environment

```
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```
pip install -r requirements.txt
```

---

## Configuration

Default configuration in `config.py`:

```python
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

LLM_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

VECTOR_DB_PATH = "vector_store/faiss_index"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

TOP_K = 3
```

### Parameter Explanation

* EMBED_MODEL — Converts text into vector embeddings
* LLM_MODEL — Generates answers using retrieved context
* VECTOR_DB_PATH — Location of FAISS vector database
* CHUNK_SIZE — Size of text chunks during ingestion
* CHUNK_OVERLAP — Overlap between chunks to preserve context
* TOP_K — Number of relevant chunks retrieved per query

---

## Usage

### Step 1: Ingest PDF

```
python ingest.py
```

This will:

* Read the PDF
* Split into chunks
* Generate embeddings
* Store in FAISS vector database

---

### Step 2: Run CLI Question Answering

```
python query.py
```

---

### Step 3: Ask Questions

Example:

```
Ask question (type 'exit'): What is attention?
```

Output:

```
Answer:
Attention is a mechanism used in transformer models that allows the system to focus on relevant parts of the input...
```

Exit:

```
exit
```

---

## How It Works

1. PDF is loaded and split into text chunks
2. Each chunk is converted into embeddings using MiniLM
3. Embeddings are stored in FAISS vector database
4. User asks a question via CLI
5. System converts question into embedding
6. FAISS retrieves most relevant chunks
7. LLM generates answer using retrieved context

---

## Example Workflow

```
python ingest.py
python query.py

Ask question: What is transformer architecture?
Answer: The transformer architecture is a neural network model based on attention mechanisms...
```

---

## Requirements

* Python 3.9+
* 8GB RAM recommended
* Windows, Linux, or MacOS

---

## Limitations

* Runs on CPU (slower than GPU)
* Requires ingestion before querying
* Does not display source file name by default

---

## Future Improvements

* Source citation display
* Support multiple PDFs with source tracking
* Faster inference using optimized runtimes
* Optional web interface

---

## License

MIT License

---

## Author

Built as a local CLI-based PDF Question Answering system using RAG architecture.
