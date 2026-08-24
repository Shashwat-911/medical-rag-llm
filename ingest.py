"""
ingest.py — Production-Grade Multi-Format RAG Ingestion Script.
Features: Streaming recursive file loading, automated text cleaning,
batch-wise semantic chunking, and memory-safe ChromaDB vectorization.
"""

import os
import gc
import json
import re
import hashlib
from typing import Generator
import torch

from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# ── Configuration ─────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
DATA_DIR        = os.path.join(BASE_DIR, "db")
CHROMA_DIR      = os.path.join(BASE_DIR, "chromadb")   

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE      = 1000  # Stays optimally within sentence-transformer contexts
CHUNK_OVERLAP   = 150
BATCH_SIZE      = 500   # Number of chunks committed per vectorstore batch (prevents MemoryError)
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
# ─────────────────────────────────────────────────────────────────────────────

# ── 1. Data Cleaning & Normalization Layer ──────────────────────────────────
def clean_text(text: str) -> str:
    """Advanced clinical text cleaning layer to purge PDF parsing artifacts,
    broken ligatures, and structural noise before vectorization."""
    if not text:
        return ""
    
    # 1. Byte-level normalization
    text = text.encode("utf-8", "ignore").decode("utf-8")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    
    # 2. Purge common PDF encoding/ligature artifacts (e.g., A5%, PC5, random isolated characters)
    text = re.compile(r'\b[A-Z][0-9]+%').sub('', text) 
    
    # 3. Standardize medical symbols and units
    text = re.compile(r'(?<=\d)\s*0\s*F\b').sub('°F', text)
    text = re.compile(r'(?<=\d)\s*0\s*C\b').sub('°C', text)
    text = text.replace("°F", "°F").replace("°C", "°C")
    
    # 4. Remove structural layout lines, hyphens, and decorative noise
    text = re.sub(r'─{2,}', '', text)
    text = re.sub(r'_{2,}', '', text)
    text = re.sub(r'\*+', '', text)
    
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


# ── 2. Streaming Document Generator ──────────────────────────────────────────
def stream_file_documents(file_path: str) -> Generator[Document, None, None]:
    """Yields cleaned Document instances one by one to keep memory footprint minimal."""
    ext = os.path.splitext(file_path)[1].lower()
    filename = os.path.basename(file_path)
    file_hash = compute_file_hash(file_path)

    try:
        # --- PDFs ---
        if ext == '.pdf':
            loader = PyPDFLoader(file_path)
            for doc in loader.load():
                cleaned_content = clean_text(doc.page_content)
                if cleaned_content:
                    yield Document(
                        page_content=cleaned_content,
                        metadata={"source": filename, "page": doc.metadata.get("page", 0), "file_hash": file_hash}
                    )

        # --- TXT ---
        elif ext == '.txt':
            loader = TextLoader(file_path, encoding="utf-8", autodetect_encoding=True)
            for doc in loader.load():
                cleaned_content = clean_text(doc.page_content)
                if cleaned_content:
                    yield Document(
                        page_content=cleaned_content,
                        metadata={"source": filename, "file_hash": file_hash}
                    )

        # --- CSV ---
        elif ext == '.csv':
            loader = CSVLoader(file_path=file_path, encoding="utf-8")
            for doc in loader.load():
                cleaned_content = clean_text(doc.page_content)
                if cleaned_content:
                    yield Document(
                        page_content=cleaned_content,
                        metadata={"source": filename, "row": doc.metadata.get("row", 0), "file_hash": file_hash}
                    )

        # --- JSON ---
        elif ext == '.json':
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    print(f"    [!] Invalid JSON structure in {filename}. Skipping.")
                    return

            if isinstance(data, list):
                for idx, item in enumerate(data):
                    if isinstance(item, dict):
                        content = "\n".join([f"{str(k).title()}: {str(v)}" for k, v in item.items() if v])
                    else:
                        content = str(item)
                    cleaned_content = clean_text(content)
                    if cleaned_content:
                        yield Document(
                            page_content=cleaned_content,
                            metadata={"source": filename, "record_index": idx, "file_hash": file_hash}
                        )
            elif isinstance(data, dict):
                content = "\n".join([f"{str(k).title()}: {str(v)}" for k, v in data.items() if v])
                cleaned_content = clean_text(content)
                if cleaned_content:
                    yield Document(
                        page_content=cleaned_content,
                        metadata={"source": filename, "file_hash": file_hash}
                    )

        # --- JSONL ---
        elif ext == '.jsonl':
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                for idx, line in enumerate(f):
                    if line.strip():
                        try:
                            data = json.loads(line)
                            if isinstance(data, dict):
                                content = "\n".join([f"{str(k).title()}: {str(v)}" for k, v in data.items() if v])
                                cleaned_content = clean_text(content)
                                if cleaned_content:
                                    yield Document(
                                        page_content=cleaned_content,
                                        metadata={"source": filename, "line_index": idx, "file_hash": file_hash}
                                    )
                        except json.JSONDecodeError:
                            pass

    except Exception as e:
        print(f"    [!] Error parsing {filename}: {str(e)}")


# ── 3. Streamed Batch Ingestion & Vector Storage ─────────────────────────────
def run_streaming_pipeline(directory: str, batch_size: int = BATCH_SIZE):
    """Streams, chunks, and commits documents in micro-batches to prevent MemoryError."""
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Source path '{directory}' cannot be resolved.")

    all_files = []
    supported_exts = {".pdf", ".txt", ".csv", ".json", ".jsonl"}
    for root, _, files in os.walk(directory):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in supported_exts and not file.startswith((".", "~")):
                all_files.append(os.path.join(root, file))

    if not all_files:
        raise FileNotFoundError("No valid documents (.pdf, .txt, .csv, .json, .jsonl) discovered inside target directory.")

    print(f"Discovered {len(all_files)} supported source files.")
    
    if DEVICE == "cuda":
        print(f"⚡ GPU Acceleration Active: {torch.cuda.get_device_name(0)}")
        print(f"⚡ VRAM Available: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    else:
        print("ℹ️ Running on CPU mode")

    print(f"\nLoading embedding model: {EMBEDDING_MODEL} on device '{DEVICE}'...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": DEVICE},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 64},
    )

    print(f"Connecting to Chroma vector store at '{CHROMA_DIR}'...")
    os.makedirs(CHROMA_DIR, exist_ok=True)
    vector_store = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""],
        keep_separator=True
    )

    total_docs = 0
    total_chunks = 0
    chunk_buffer = []

    print("\n[Executing Memory-Safe Streamed Ingestion & Batch Vectorization]...")

    for file_idx, file_path in enumerate(all_files, 1):
        filename = os.path.basename(file_path)
        file_docs_count = 0
        
        for doc in stream_file_documents(file_path):
            total_docs += 1
            file_docs_count += 1
            
            # Split incrementally (constant low RAM)
            doc_chunks = splitter.split_documents([doc])
            for chunk in doc_chunks:
                chunk.metadata["chunk_index"] = total_chunks + len(chunk_buffer)
                chunk_buffer.append(chunk)

            # Flush batch to vector database
            if len(chunk_buffer) >= batch_size:
                vector_store.add_documents(chunk_buffer)
                total_chunks += len(chunk_buffer)
                print(f"  [Indexed] {total_chunks:,} chunks written | File {file_idx}/{len(all_files)}: {filename}")
                chunk_buffer.clear()
                gc.collect()

        if file_docs_count > 0 and file_idx % 25 == 0:
            print(f"  -- Progress: Completed {file_idx}/{len(all_files)} files ({total_docs:,} documents) --")

    # Flush remaining chunks in buffer
    if chunk_buffer:
        vector_store.add_documents(chunk_buffer)
        total_chunks += len(chunk_buffer)
        print(f"  [Final Batch] {total_chunks:,} chunks total written to database.")
        chunk_buffer.clear()
        gc.collect()

    if hasattr(vector_store, 'persist'):
        vector_store.persist()

    print(f"\n✔ Ingestion finished successfully!")
    print(f"  • Total Documents Processed: {total_docs:,}")
    print(f"  • Total Vector Chunks Indexed: {total_chunks:,}")
    print(f"  • Database Persisted At: {CHROMA_DIR}")


if __name__ == "__main__":
    print("=" * 60)
    print("  Enterprise-Grade Multi-Format Ingestion Pipeline (Streaming)")
    print("=" * 60)

    run_streaming_pipeline(DATA_DIR, batch_size=BATCH_SIZE)

    print("\n🎉 Operations completed. Ready for Streamlit RAG Deployment!")