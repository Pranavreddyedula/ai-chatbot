import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OllamaEmbeddings
from langchain.llms import Ollama
from langchain.chains import ConversationalRetrievalChain
import os

st.set_page_config(page_title="Local Multi-PDF AI Chatbot", layout="wide")
st.title("🤖 Local Multi-PDF AI Chatbot (Llama 3 + LangChain + Ollama)")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

data_dir = "data"
vector_dir = "vectorstore"
os.makedirs(data_dir, exist_ok=True)
os.makedirs(vector_dir, exist_ok=True)

uploaded_files = st.file_uploader("📂 Upload PDF files", type=["pdf"], accept_multiple_files=True)

if st.button("Process Documents"):
    if uploaded_files:
        all_texts = []
        for file in uploaded_files:
            file_path = os.path.join(data_dir, file.name)
            with open(file_path, "wb") as f:
                f.write(file.read())
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
            all_texts.extend(splitter.split_documents(docs))

        embeddings = OllamaEmbeddings(model="llama3")
        vectorstore = Chroma.from_documents(all_texts, embeddings, persist_directory=vector_dir)
        vectorstore.persist()
        st.success("✅ Documents processed successfully!")
    else:
        st.warning("Please upload PDFs first.")

query = st.text_input("💬 Ask a question about your PDFs:")
if query:
    embeddings = OllamaEmbeddings(model="llama3")
    db = Chroma(persist_directory=vector_dir, embedding_function=embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 4})
    llm = Ollama(model="llama3")
    chain = ConversationalRetrievalChain.from_llm(llm, retriever)
    result = chain.invoke({"question": query, "chat_history": st.session_state.chat_history})
    st.session_state.chat_history.append((query, result["answer"]))
    st.markdown(f"**AI:** {result['answer']}")
