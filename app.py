# app.py

import streamlit as st
from pathlib import Path

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OllamaEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import Ollama

# PDF processing
from PyPDF2 import PdfReader

# ------------------------
# Streamlit UI
# ------------------------
st.set_page_config(page_title="Local Multi-PDF AI Chatbot", layout="wide")

st.title("📄 Local PDF AI Chatbot")

# Upload PDFs
uploaded_files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    raw_texts = []
    for pdf_file in uploaded_files:
        pdf_reader = PdfReader(pdf_file)
        for page in pdf_reader.pages:
            raw_texts.append(page.extract_text())

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # adjust chunk size as needed
        chunk_overlap=100
    )
    docs = text_splitter.split_text(" ".join(raw_texts))

    # Create vectorstore with Ollama embeddings
    embeddings = OllamaEmbeddings()
    vectorstore = Chroma.from_texts(docs, embedding=embeddings, persist_directory="vectorstore")

    st.success("✅ PDF processed and embeddings created!")

    # Query interface
    query = st.text_input("Ask a question about your PDFs:")
    if query:
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        qa_chain = RetrievalQA.from_chain_type(
            llm=Ollama(model="llama2", model_kwargs={"temperature": 0}),
            chain_type="stuff",
            retriever=retriever
        )
        answer = qa_chain.run(query)
        st.write("💡 Answer:", answer)
