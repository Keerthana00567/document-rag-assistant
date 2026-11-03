# PDF RAG Assistant

A Retrieval-Augmented Generation (RAG) application that allows users to upload PDF documents and ask questions about their contents using a locally running Large Language Model.

## Project Overview

Traditional language models may not have access to information contained in a user's private documents.

This project addresses that problem by combining semantic search with a locally running language model to generate context-aware answers from uploaded PDF documents.

## Features

- Upload PDF documents through a web interface.
- Extract text using PyMuPDF.
- Split documents into overlapping text chunks.
- Generate document embeddings using Sentence Transformers.
- Perform semantic search using cosine similarity.
- Retrieve the top 5 relevant document chunks.
- Generate answers using Ollama.
- Display retrieved sources and similarity scores.
- Maintain document-specific chat history.
- Cache document processing and embeddings.

## Architecture

1. **PDF Upload:** User uploads a PDF document.
2. **Text Extraction:** PyMuPDF extracts text from the document.
3. **Chunking:** RecursiveCharacterTextSplitter divides text into chunks of 1000 characters with 200 characters of overlap.
4. **Embedding Generation:** `all-MiniLM-L6-v2` converts chunks into 384-dimensional vectors.
5. **Semantic Retrieval:** Cosine similarity identifies the most relevant chunks.
6. **Context Construction:** Retrieved chunks are combined into context.
7. **Answer Generation:** Ollama runs `llama3.2:3b` to generate an answer using the retrieved context.
8. **Source Display:** The application displays retrieved chunks and their similarity scores.

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Streamlit | Web interface |
| PyMuPDF | PDF text extraction |
| LangChain Text Splitters | Text chunking |
| Sentence Transformers | Text embeddings |
| PyTorch | Tensor operations |
| Ollama | Local LLM inference |
| all-MiniLM-L6-v2 | Embedding model |
| llama3.2:3b | Generative language model |

## Installation

Clone the repository:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd document-rag
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Install Ollama and download the language model:

```bash
ollama pull llama3.2:3b
```

Run the application:

```bash
streamlit run app.py
```

## Limitations

- Currently supports text-based PDFs.
- Scanned PDFs require OCR support.
- Answer quality depends on retrieval relevance and language model capabilities.
- The application currently uses a fixed chunk size and top-k retrieval value.
- Local inference performance depends on available hardware.

## Future Improvements

- OCR support for scanned PDFs.
- Page-number tracking and source citations.
- Improved retrieval using reranking.
- Conversation persistence.
- Support for multiple document collections.
- More advanced RAG evaluation.
- Improved mathematical formula rendering.

## Project Status

Core PDF ingestion, embedding generation, semantic retrieval, local LLM integration and Streamlit interface have been implemented.