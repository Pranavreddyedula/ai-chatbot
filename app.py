import os
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OllamaEmbeddings
from langchain.llms import Ollama
from langchain.chains import ConversationalRetrievalChain

# Optional: for PDF loading
try:
    from langchain.document_loaders import PyPDFLoader
except ImportError:
    st.error("PyPDFLoader not found. Make sure you have langchain>=0.1.159 installed.")

# Streamlit page config
st.set_page_config(page_title="Local Multi-PDF AI Chatbot", layout="wide")
st.title("🤖 Local Multi-PDF AI Chatbot (Llama 3 + LangChain + Ollama)")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Folders
DATA_DIR = "data"
VECTOR_DIR = "vectorstore"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(VECTOR_DIR, exist_ok=True)

# PDF Upload
uploaded_files = st.file_uploader("📂 Upload PDF files", type=["pdf"], accept_multiple_files=True)

if st.button("Process Documents"):
    if uploaded_files:
        all_texts = []
        for file in uploaded_files:
            file_path = os.path.join(DATA_DIR, file.name)
            with open(file_path, "wb") as f:
                f.write(file.read())
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
            all_texts.extend(splitter.split_documents(docs))

        embeddings = OllamaEmbeddings(model="llama3")
        vectorstore = Chroma.from_documents(all_texts, embeddings, persist_directory=VECTOR_DIR)
        vectorstore.persist()
        st.success("✅ Documents processed successfully!")
    else:
        st.warning("Please upload PDFs first.")

# Chat Interface
query = st.text_input("💬 Ask a question about your PDFs:")
if query:
    embeddings = OllamaEmbeddings(model="llama3")
    db = Chroma(persist_directory=VECTOR_DIR, embedding_function=embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 4})
    llm = Ollama(model="llama3")
    chain = ConversationalRetrievalChain.from_llm(llm, retriever)
    result = chain.invoke({"question": query, "chat_history": st.session_state.chat_history})
    st.session_state.chat_history.append((query, result["answer"]))
    st.markdown(f"**AI:** {result['answer']}")
