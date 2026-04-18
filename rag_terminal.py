import os
# Suppress unnecessary logging
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

# --- UPDATED IMPORTS FOR LANGCHAIN V1 ---
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain

# --- 1. SET UP EMBEDDINGS ---
print("🔄 Loading embedding model: sentence-transformers/all-MiniLM-L6-v2...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

# --- 2. LOAD CHROMA DATABASE ---
CHROMA_DIR = "./chroma_db" 

if not os.path.exists(CHROMA_DIR):
    print(f"❌ Error: Folder '{CHROMA_DIR}' not found. Did you run ingest.py?")
    exit()

print(f"📂 Accessing vector database at '{CHROMA_DIR}'...")
vectordb = Chroma(
    persist_directory=CHROMA_DIR, 
    embedding_function=embeddings
)

retriever = vectordb.as_retriever(search_kwargs={"k": 3})

# --- 3. SET UP THE LLM (GROQ) ---
os.environ["GROQ_API_KEY"] = "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" 

llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.1)

# --- 4. CREATE THE RAG CHAIN ---
system_prompt = (
    "You are a helpful assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer the question. "
    "If you don't know the answer, just say that you don't know. "
    "Keep the answer concise.\n\n"
    "Context:\n{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

# Use the updated classic chain creators
combine_docs_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, combine_docs_chain)

# --- 5. TERMINAL INTERFACE ---
print("\n" + "="*55)
print("🚀 RAG SYSTEM READY (Python 3.14 / Groq / Classic Mode)")
print("Type 'exit' or 'quit' to close the program.")
print("="*55 + "\n")

while True:
    user_query = input("\n🧑 User: ")
    
    if user_query.lower() in ['exit', 'quit']:
        print("Goodbye!")
        break
        
    if not user_query.strip():
        continue

    print("🔎 Searching local PDFs and generating answer...")
    
    try:
        response = rag_chain.invoke({"input": user_query})
        print(f"\n🤖 Assistant: {response['answer']}")
        
        print("\n📚 Sources:")
        sources = set([doc.metadata.get("source", "Unknown") for doc in response["context"]])
        for source in sources:
            print(f"- {source}")
            
    except Exception as e:
        print(f"⚠️ An error occurred: {e}")