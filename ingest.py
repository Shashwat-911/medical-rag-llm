"""
ingest.py — Multi-format RAG Ingestion Script.
Processes PDF, TXT, CSV, JSON, and JSONL files and builds a ChromaDB vector store.
"""

import os
import glob
import json
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# ── Configuration ─────────────────────────────────────────────────────────────
# Folder containing your 862 raw files
DATA_DIR      = r"C:\Users\Shashwat\Desktop\internship\RAG\db"        

# New separate folder where the vector store will be saved
CHROMA_DIR    = r"C:\Users\Shashwat\Desktop\internship\RAG\my_chroma_db"   

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
# ─────────────────────────────────────────────────────────────────────────────

def process_json(file_path: str) -> list[Document]:
    """Helper function to extract text from generic JSON files."""
    docs = []
    with open(file_path, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            print(f"    [!] Error reading {file_path}. Skipping.")
            return docs

        # If it's a list of dictionaries (common for structured data)
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    # Flatten the dictionary into a readable text block
                    content = "\n".join([f"{str(k).capitalize()}: {str(v)}" for k, v in item.items()])
                    docs.append(Document(page_content=content, metadata={"source": os.path.basename(file_path)}))
                elif isinstance(item, str):
                    docs.append(Document(page_content=item, metadata={"source": os.path.basename(file_path)}))
                    
        # If it's a single dictionary
        elif isinstance(data, dict):
            content = "\n".join([f"{str(k).capitalize()}: {str(v)}" for k, v in data.items()])
            docs.append(Document(page_content=content, metadata={"source": os.path.basename(file_path)}))
            
    return docs


def load_all_documents(directory: str):
    """Load PDF, TXT, CSV, JSON, and JSONL files from the directory."""
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory '{directory}' does not exist. Please check your paths.")

    all_docs = []
    
    # Get all file paths in the directory
    all_files = glob.glob(os.path.join(directory, "*.*"))
    
    if not all_files:
        raise FileNotFoundError(f"No files found in '{directory}'")

    for path in all_files:
        ext = os.path.splitext(path)[1].lower()
        filename = os.path.basename(path)
        
        # 1. Process PDFs
        if ext == '.pdf':
            print(f"  Loading PDF: {filename}")
            try:
                loader = PyPDFLoader(path)
                docs = loader.load()
                for doc in docs: doc.metadata["source"] = filename
                all_docs.extend(docs)
            except Exception as e:
                print(f"    [!] Failed to load {filename}: {e}")
            
        # 2. Process Text Files
        elif ext == '.txt':
            print(f"  Loading TXT: {filename}")
            try:
                loader = TextLoader(path, encoding="utf-8")
                docs = loader.load()
                for doc in docs: doc.metadata["source"] = filename
                all_docs.extend(docs)
            except Exception as e:
                print(f"    [!] Failed to load {filename}: {e}")
                
        # 3. Process CSV Files
        elif ext == '.csv':
            print(f"  Loading CSV: {filename}")
            try:
                # CSVLoader creates one document per row
                loader = CSVLoader(file_path=path, encoding="utf-8")
                docs = loader.load()
                for doc in docs: doc.metadata["source"] = filename
                all_docs.extend(docs)
            except Exception as e:
                print(f"    [!] Failed to load {filename}: {e}")
                
        # 4. Process JSON Files
        elif ext == '.json':
            print(f"  Loading JSON: {filename}")
            docs = process_json(path)
            all_docs.extend(docs)
            
        # 5. Process JSONL Files
        elif ext == '.jsonl':
            print(f"  Loading JSONL: {filename}")
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip(): # Ignore empty lines
                            try:
                                data = json.loads(line)
                                if isinstance(data, dict):
                                    content = "\n".join([f"{str(k).capitalize()}: {str(v)}" for k, v in data.items()])
                                    doc = Document(page_content=content, metadata={"source": filename})
                                    all_docs.append(doc)
                            except json.JSONDecodeError:
                                pass # Skip bad lines silently
            except Exception as e:
                 print(f"    [!] Failed to load {filename}: {e}")
            
        else:
            # Silently skip unsupported formats to avoid terminal clutter
            pass

    print(f"\n✔ Total loaded: {len(all_docs)} raw document parts.")
    return all_docs


def split_documents(documents):
    """Split raw pages/rows into smaller overlapping chunks for the database."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"✔ Split into {len(chunks)} searchable chunks.")
    return chunks


def build_vector_store(chunks):
    """Embed chunks and persist to ChromaDB."""
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
    print(f"✔ Vector store successfully saved to '{CHROMA_DIR}'.")
    return vector_store


if __name__ == "__main__":
    print("=" * 55)
    print("  Multi-Format RAG Ingestion Pipeline")
    print("=" * 55)

    print("\n[1/3] Scanning and Loading Files …")
    documents = load_all_documents(DATA_DIR)

    print("\n[2/3] Splitting documents …")
    chunks = split_documents(documents)

    print("\n[3/3] Embedding & persisting …")
    build_vector_store(chunks)

    print(f"\n🎉 Ingestion complete! Your database is ready at: {CHROMA_DIR}")
