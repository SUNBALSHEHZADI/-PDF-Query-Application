# RAG-based PDF Query Application

This is a Streamlit-based application that allows users to upload a PDF document and ask questions about its content. The application uses a Retrieval-Augmented Generation (RAG) approach to retrieve relevant information from the PDF and generate answers using the Groq API.

## Features

- **PDF Upload**: Users can upload a PDF document.
- **Text Extraction**: The application extracts text from the uploaded PDF.
- **Text Chunking**: The extracted text is split into smaller chunks for efficient processing.
- **FAISS Indexing**: The text chunks are embedded using a sentence transformer model and indexed using FAISS for fast similarity search.
- **Question Answering**: Users can ask questions about the PDF content, and the application retrieves the most relevant chunks and generates an answer using the Groq API.

## How It Works

1. **PDF Upload**: The user uploads a PDF file.
2. **Text Extraction**: The application extracts text from the PDF.
3. **Text Chunking**: The text is split into smaller chunks with a specified size and overlap.
4. **FAISS Indexing**: The text chunks are embedded and indexed using FAISS.
5. **Question Answering**: The user asks a question, and the application retrieves the most relevant chunks using FAISS.
6. **Answer Generation**: The retrieved chunks are passed to the Groq API, which generates an answer based on the context.

## Requirements

- Python 3.8+
- Streamlit
- PyMuPDF (fitz)
- FAISS
- Sentence Transformers
- Groq API

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/SUNBALSHEHZADI/rag-pdf-query.git
   cd rag-pdf-query
