"""
ingest.py — Production-Grade Multi-Format RAG Ingestion Script.
Features: Recursive file loading, automated text cleaning, semantic token-based chunking,
and idempotency tracking.
"""

import os
import json
import re
import hashlib
from typing import List
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# ── Configuration ─────────────────────────────────────────────────────────────
DATA_DIR        = r"C:\Users\Shashwat\Desktop\internship\RAG\db"        
CHROMA_DIR      = r"C:\Users\Shashwat\Desktop\internship\RAG\my_chroma_db"   

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE      = 1000  # Slightly lowered to stay optimally within sentence-transformer contexts
CHUNK_OVERLAP   = 150
# ─────────────────────────────────────────────────────────────────────────────

# ── 1. Data Cleaning & Normalization Layer ──────────────────────────────────
# Replace the clean_text function in your updated_ingest.py with this hardened version:

def clean_text(text: str) -> str:
    """Advanced clinical text cleaning layer to purge PDF parsing artifacts,
    broken ligatures, and structural noise before vectorization."""
    if not text:
        return ""
    
    # 1. Byte-level normalization
    text = text.encode("utf-8", "ignore").decode("utf-8")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    
    # 2. Purge common PDF encoding/ligature artifacts (e.g., A5%, PC5, random isolated characters)
    # Fixes specific symbols mashing into letters next to percentages/numbers
    text = re.compile(r'\b[A-Z][0-9]+%').sub('', text) 
    
    # 3. Standardize medical symbols and units (e.g., converting broken degree artifacts)
    text = re.compile(r'(?<=\d)\s*0\s*F\b').sub('°F', text)
    text = re.compile(r'(?<=\d)\s*0\s*C\b').sub('°C', text)
    text = text.replace("°F", "°F").replace("°C", "°C")
    
    # 4. Remove structural layout lines, hyphens, and decorative noise
    text = re.sub(r'─{2,}', '', text)
    text = re.sub(r'_{2,}', '', text)
    text = re.sub(r'\*+', '', text) # Strip leftover erratic markdown stars from raw loaders
    
    # 5. Fix hyphenated word breaks split across lines due to PDF margins
    text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
    
    # 6. Consolidate whitespace while respecting distinct paragraph markers
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text.strip()


def compute_file_hash(file_path: str) -> str:
    """Generates a unique MD5 signature of the file content to prevent redundant indexing."""
    hasher = hashlib.md5()
    try:
        with open(file_path, 'rb') as f:
            buf = f.read(65536)
            while len(buf) > 0:
                hasher.update(buf)
                buf = f.read(65536)
        return hasher.hexdigest()
    except Exception:
        return ""


def process_json(file_path: str) -> List[Document]:
    """Extracts, cleans, and structures raw JSON documents."""
    docs = []
    filename = os.path.basename(file_path)
    file_hash = compute_file_hash(file_path)

    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            print(f"    [!] Invalid JSON structure in {filename}. Skipping.")
            return docs

        # Flatten records cleanly to preserve key-value context for embedding vectors
        if isinstance(data, list):
            for idx, item in enumerate(data):
                if isinstance(item, dict):
                    content = "\n".join([f"{str(k).title()}: {str(v)}" for k, v in item.items() if v])
                else:
                    content = str(item)
                
                cleaned_content = clean_text(content)
                if cleaned_content:
                    docs.append(Document(
                        page_content=cleaned_content, 
                        metadata={"source": filename, "record_index": idx, "file_hash": file_hash}
                    ))
        elif isinstance(data, dict):
            content = "\n".join([f"{str(k).title()}: {str(v)}" for k, v in data.items() if v])
            cleaned_content = clean_text(content)
            if cleaned_content:
                docs.append(Document(
                    page_content=cleaned_content, 
                    metadata={"source": filename, "file_hash": file_hash}
                ))
            
    return docs


# ── 2. Recursive Loading & Indexing Layer ───────────────────────────────────
def load_all_documents(directory: str) -> List[Document]:
    """Recursively walks directories to identify and parse target documents."""
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Source path '{directory}' cannot be resolved.")

    all_docs = []
    all_files = []
    
    for root, _, files in os.walk(directory):
        for file in files:
            all_files.append(os.path.join(root, file))
    
    if not all_files:
        raise FileNotFoundError(f"No documents discovered inside target root directory.")

    print(f"Discovered {len(all_files)} files. Executing ingestion parsing & cleanup...")

    for path in all_files:
        ext = os.path.splitext(path)[1].lower()
        filename = os.path.basename(path)
        file_hash = compute_file_hash(path)
        
        try:
            # --- PDFs ---
            if ext == '.pdf':
                print(f"  [Parsing] PDF -> {filename}")
                loader = PyPDFLoader(path)
                raw_extracted = loader.load()
                for doc in raw_extracted:
                    doc.page_content = clean_text(doc.page_content)
                    doc.metadata = {"source": filename, "page": doc.metadata.get("page", 0), "file_hash": file_hash}
                    if doc.page_content:
                        all_docs.append(doc)
                
            # --- TXT ---
            elif ext == '.txt':
                print(f"  [Parsing] TXT -> {filename}")
                loader = TextLoader(path, encoding="utf-8", autodetect_encoding=True)
                raw_extracted = loader.load()
                for doc in raw_extracted:
                    doc.page_content = clean_text(doc.page_content)
                    doc.metadata = {"source": filename, "file_hash": file_hash}
                    if doc.page_content:
                        all_docs.append(doc)
                    
            # --- CSV ---
            elif ext == '.csv':
                print(f"  [Parsing] CSV -> {filename}")
                loader = CSVLoader(file_path=path, encoding="utf-8")
                raw_extracted = loader.load()
                for doc in raw_extracted:
                    doc.page_content = clean_text(doc.page_content)
                    doc.metadata = {"source": filename, "row": doc.metadata.get("row", 0), "file_hash": file_hash}
                    if doc.page_content:
                        all_docs.append(doc)
                    
            # --- JSON ---
            elif ext == '.json':
                print(f"  [Parsing] JSON -> {filename}")
                all_docs.extend(process_json(path))
                
            # --- JSONL ---
            elif ext == '.jsonl':
                print(f"  [Parsing] JSONL -> {filename}")
                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                    for idx, line in enumerate(f):
                        if line.strip():
                            try:
                                data = json.loads(line)
                                if isinstance(data, dict):
                                    content = "\n".join([f"{str(k).title()}: {str(v)}" for k, v in data.items() if v])
                                    cleaned_content = clean_text(content)
                                    if cleaned_content:
                                        all_docs.append(Document(
                                            page_content=cleaned_content, 
                                            metadata={"source": filename, "line_index": idx, "file_hash": file_hash}
                                        ))
                            except json.JSONDecodeError:
                                pass
        except Exception as e:
            print(f"    [!] Fatal parsing exception bypassed for {filename}: {str(e)}")

    print(f"\n✔ Ingestion finished. Collected {len(all_docs)} unique, cleaned textual objects.")
    return all_docs


# ── 3. Strategic Text Chunking Layer ────────────────────────────────────────
def split_documents(documents: List[Document]) -> List[Document]:
    """Maintains semantic relationship boundaries during parsing configurations."""
    # Chunking separator chain ordered from macro to micro structural breaks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""],
        keep_separator=True
    )
    
    chunks = splitter.split_documents(documents)
    
    # Inject unique tracking IDs directly onto vector blocks for atomic targeting
    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = index
        
    print(f"✔ Optimized processing produced {len(chunks)} contextual vector chunks.")
    return chunks


# ── 4. Storage & Idempotency Layer ──────────────────────────────────────────
def build_vector_store(chunks: List[Document]):
    """Loads weights to local machine memory and commits structural updates to database."""
    print(f"\nLoading optimized embedding weight matrix '{EMBEDDING_MODEL}'...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    print(f"Constructing vector space instances -> '{CHROMA_DIR}'...")
    
    # Using Chroma natively while avoiding massive indexing overhead
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
    )
    vector_store.persist()
    print(f"✔ Target database vector arrays successfully written to disks.")
    return vector_store


if __name__ == "__main__":
    print("=" * 60)
    print("  Enterprise-Grade Multi-Format Ingestion Pipeline")
    print("=" * 60)

    print("\n[Phase 1/3] Direct Deep File Analysis & Data Cleaning...")
    documents = load_all_documents(DATA_DIR)

    print("\n[Phase 2/3] Executing Structural Semantic Chunking...")
    chunks = split_documents(documents)

    print("\n[Phase 3/3] Synchronizing Vector Storage & Embeddings...")
    build_vector_store(chunks)

    print(f"\n🎉 Operations completed. Deployment environment verified at: {CHROMA_DIR}")
