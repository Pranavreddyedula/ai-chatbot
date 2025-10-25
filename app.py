import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OllamaEmbeddings
from langchain.llms import Ollama
from langchain.chains import ConversationalRetrievalChain
import os

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="Local Multi-PDF AI Chatbot", layout="wide")
st.title("🤖 Local Multi-PDF AI Chatbot (Llama 3 + LangChain + Ollama)")

# -------------------------------
# Session State
# -------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# -------------------------------
# Directories
# -------------------------------
data_dir = "data"
vector_dir = "vectorstore"
os.makedirs(data_dir, exist_ok=True)
os.makedirs(vector_dir, exist_ok=True)

# -------------------------------
# PDF Upload
# -------------------------------
uploaded_files = st.file_uploader(
    "📂 Upload PDF files", type=["pdf"], accept_multiple_files=True
)

if st.button("Process Documents"):
    if uploaded_files:
        all_texts = []
        for file in uploaded_files:
            file_path = os.path.join(data_dir, file.name)
            with open(file_path, "wb") as f:
                f.write(file.read())
            
            # Load PDF and split into chunks
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
            all_texts.extend(splitter.split_documents(docs))

        # Create embeddings and vectorstore
        embeddings = OllamaEmbeddings(model="llama3")
        vectorstore = Chroma.from_documents(all_texts, embeddings, persist_directory=vector_dir)
        vectorstore.persist()
        st.session_state.vectorstore = vectorstore
        st.success("✅ Documents processed successfully!")
    else:
        st.warning("Please upload PDFs first.")

# -------------------------------
# Query Section
# -------------------------------
query = st.text_input("💬 Ask a question about your PDFs:")

if query:
    if st.session_state.vectorstore is None:
        # Load vectorstore from disk if not in session
        embeddings = OllamaEmbeddings(model="llama3")
        st.session_state.vectorstore = Chroma(persist_directory=vector_dir, embedding_function=embeddings)

    retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 4})
    llm = Ollama(model="llama3")
    chain = ConversationalRetrievalChain.from_llm(llm, retriever)
    
    result = chain.invoke({"question": query, "chat_history": st.session_state.chat_history})
    st.session_state.chat_history.append((query, result["answer"]))
    
    st.markdown(f"**AI:** {result['answer']}")
