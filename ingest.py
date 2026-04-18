"""
ingest.py — One-time script to build the ChromaDB vector store from local PDFs.
Run this ONCE before starting the Flask server:
    python ingest.py
"""

import os
import glob
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# ── Configuration ─────────────────────────────────────────────────────────────
PDF_DIR       = "./"          # Folder containing your PDF files
CHROMA_DIR    = "./chroma_db" # Where the vector store will be saved
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE    = 500
CHUNK_OVERLAP = 50
# ─────────────────────────────────────────────────────────────────────────────

def load_pdfs(directory: str):
    """Load every PDF in *directory* and return a flat list of LangChain Documents."""
    pdf_paths = glob.glob(os.path.join(directory, "*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(f"No PDF files found in '{directory}'")

    all_docs = []
    for path in pdf_paths:
        print(f"  Loading: {os.path.basename(path)}")
        loader = PyPDFLoader(path)
        docs = loader.load()
        # Attach the source filename to every chunk's metadata
        for doc in docs:
            doc.metadata["source"] = os.path.basename(path)
        all_docs.extend(docs)

    print(f"\n✔ Loaded {len(all_docs)} pages from {len(pdf_paths)} PDFs.")
    return all_docs


def split_documents(documents):
    """Split raw pages into smaller overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"✔ Split into {len(chunks)} chunks.")
    return chunks


def build_vector_store(chunks):
    """Embed chunks with all-MiniLM-L6-v2 and persist to ChromaDB."""
    print(f"\nLoading embedding model '{EMBEDDING_MODEL}' …")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    print(f"Building vector store → '{CHROMA_DIR}' …")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
    )
    vector_store.persist()
    print(f"✔ Vector store saved to '{CHROMA_DIR}'.")
    return vector_store


if __name__ == "__main__":
    print("=" * 55)
    print("  RAG Ingestion Pipeline")
    print("=" * 55)

    print("\n[1/3] Loading PDFs …")
    documents = load_pdfs(PDF_DIR)

    print("\n[2/3] Splitting documents …")
    chunks = split_documents(documents)

    print("\n[3/3] Embedding & persisting …")
    build_vector_store(chunks)

    print("\n🎉 Ingestion complete! You can now run: python app.py")
