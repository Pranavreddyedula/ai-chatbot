import os
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
from langchain.embeddings import OllamaEmbeddings

# ===============================
# Ensure necessary folders exist
# ===============================
if not os.path.exists("data"):
    os.makedirs("data")
if not os.path.exists("vectorstore"):
    os.makedirs("vectorstore")

# ===============================
# Streamlit App
# ===============================
st.set_page_config(page_title="Local Multi-PDF AI Chatbot", layout="wide")
st.title("📄 Local Multi-PDF AI Chatbot")

# Upload PDFs
uploaded_files = st.file_uploader(
    "Upload PDF files (multiple allowed)",
    type="pdf",
    accept_multiple_files=True
)

if uploaded_files:
    documents = []

    for pdf in uploaded_files:
        pdf_path = os.path.join("data", pdf.name)

        # Save uploaded file
        with open(pdf_path, "wb") as f:
            f.write(pdf.getbuffer())

        # Load PDF
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        documents.extend(docs)

    st.success(f"✅ {len(uploaded_files)} PDF(s) uploaded and loaded!")

    # ===============================
    # Split documents into chunks
    # ===============================
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    docs_split = text_splitter.split_documents(documents)
    st.info(f"Documents split into {len(docs_split)} chunks for embedding.")

    # ===============================
    # Create or load vector store
    # ===============================
    embeddings = OllamaEmbeddings()
    vectorstore = Chroma.from_documents(
        docs_split,
        embeddings,
        persist_directory="vectorstore"
    )
    vectorstore.persist()
    st.success("✅ Embeddings created and stored in Chroma vectorstore!")

    # ===============================
    # Query interface
    # ===============================
    query = st.text_input("Ask a question about your PDFs:")

    if query:
        results = vectorstore.similarity_search(query, k=3)  # top 3 results
        st.subheader("Answers:")
        for i, res in enumerate(results, 1):
            st.write(f"**Result {i}:** {res.page_content}")
