import os
import fitz  # PyMuPDF for PDF processing
import faiss
import numpy as np
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv




# Load API key
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize Groq client
client = Groq(api_key= GROQ_API_KEY)

# Load sentence transformer model for embedding
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file using PyMuPDF."""
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text("text") + "\n"
    return text.strip()
def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file using PyMuPDF."""
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text("text") + "\n"
    return text.strip()
def create_text_chunks(text, chunk_size=500, chunk_overlap=100):
    """Split text into chunks of specified size with overlap."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_text(text)
    return chunks
def create_faiss_index(chunks):
    """Generate embeddings for text chunks and store them in FAISS."""
    embeddings = embedding_model.encode(chunks, convert_to_numpy=True)
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)  # L2 (Euclidean) distance
    index.add(embeddings)  # Add embeddings to FAISS index

    return index, embeddings, chunks
def retrieve_similar_chunks(query, index, embeddings, chunks, top_k=3):
    """Retrieve the most relevant text chunks using FAISS."""
    query_embedding = embedding_model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, top_k)

    results = [chunks[idx] for idx in indices[0]]
    return results
def query_groq_api(query, context):
    """Send the query along with retrieved context to Groq API."""
    prompt = f"Use the following context to answer the question:\n\n{context}\n\nQuestion: {query}\nAnswer:"

    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.3-70b-versatile",
    )

    return chat_completion.choices[0].message.content
import streamlit as st

st.title("📚 RAG-based PDF Query Application")
st.write("Upload a PDF and ask questions!")

# File Upload
uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file is not None:
    pdf_path = "uploaded_document.pdf"

    # Save file temporarily
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Process the PDF
    st.write("Processing PDF...")
    text = extract_text_from_pdf(pdf_path)
    chunks = create_text_chunks(text)
    index, embeddings, chunk_texts = create_faiss_index(chunks)

    st.success("PDF processed! Now you can ask questions.")

    # User Query
    query = st.text_input("Ask a question about the PDF:")

    if st.button("Get Answer"):
        if query:
            # Retrieve top chunks
            relevant_chunks = retrieve_similar_chunks(query, index, embeddings, chunk_texts)
            context = "\n\n".join(relevant_chunks)

            # Query Groq API
            response = query_groq_api(query, context)

            st.subheader("Answer:")
            st.write(response)
        else:
            st.warning("Please enter a question.")

