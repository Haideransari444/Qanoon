import os
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

# 🔒 Hardcoded path to your Knowledge Base
KB_PATH = r"E:\legeslative bot\dataa"  # Raw string to support Windows backslashes
VECTOR_STORE_PATH = "vectordb"

def load_kb_documents(folder_path):
    documents = []

    for filename in os.listdir(folder_path):
        full_path = os.path.join(folder_path, filename)

        if filename.lower().endswith(".pdf"):
            print(f"📄 Loading PDF: {filename}")
            loader = PyPDFLoader(full_path)
            documents.extend(loader.load())

        elif filename.lower().endswith(".txt"):
            print(f"📜 Loading TXT: {filename}")
            loader = TextLoader(full_path, encoding="utf-8")
            documents.extend(loader.load())

    return documents

def split_documents(documents, chunk_size=1000, chunk_overlap=150):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(documents)

def embed_documents(chunks, save_path):
    print("🔗 Creating embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(save_path)

    print(f"✅ Vector store saved to: {save_path}")

if __name__ == "__main__":
    if not os.path.exists(KB_PATH):
        print(f"❌ The path '{KB_PATH}' does not exist.")
        exit(1)

    print(f"📁 Loading documents from: {KB_PATH}")

    raw_docs = load_kb_documents(KB_PATH)
    print(f"📄 Loaded {len(raw_docs)} documents")

    chunks = split_documents(raw_docs)
    print(f"✂️ Split into {len(chunks)} chunks")

    embed_documents(chunks, VECTOR_STORE_PATH)
