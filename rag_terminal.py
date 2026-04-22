import os
# Suppress unnecessary logging
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="langchain_core._api")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOllama
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
# Updated to match exactly where your ingest.py saved the database
CHROMA_DIR = "./my_chroma_db" 

if not os.path.exists(CHROMA_DIR):
    print(f"❌ Error: Folder '{CHROMA_DIR}' not found. Did you run your ingest script?")
    exit()

print(f"📂 Accessing vector database at '{CHROMA_DIR}'...")
vectordb = Chroma(
    persist_directory=CHROMA_DIR, 
    embedding_function=embeddings
)

retriever = vectordb.as_retriever(
    search_type="mmr", 
    search_kwargs={"k": 8, "fetch_k": 20}
) # Retrieves top 10 chunks

# --- 3. SET UP LOCAL LLM (OLLAMA) ---
print("🧠 Connecting to local Ollama model (Llama 3)...")
llm = ChatOllama(model="llama3", temperature=0)

# --- 4. CREATE THE RAG CHAIN ---
system_prompt = (
    "You are a helpful medical assistant. "
    "Use the following pieces of retrieved context to answer the question. "
    "If you don't know the answer, just say that you don't know. "
    "Do not make up information. Keep the answer concise.\n\n"
    "Context:\n{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

combine_docs_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, combine_docs_chain)

# --- 5. TERMINAL INTERFACE ---
print("\n" + "="*55)
print("🚀 100% LOCAL RAG SYSTEM READY (Ollama / Llama 3)")
print("Type 'exit' or 'quit' to close the program.")
print("="*55 + "\n")

while True:
    user_query = input("\n🧑 User: ")
    
    if user_query.lower() in ['exit', 'quit']:
        print("Goodbye!")
        break
        
    if not user_query.strip():
        continue

    print("🔎 Searching local database and generating answer...")
    
    try:
        response = rag_chain.invoke({"input": user_query})
        print(f"\n🤖 Assistant: {response['answer']}")
        
        # Print the filenames it used to answer
        print("\n📚 Sources:")
        sources = set([doc.metadata.get('source', 'Unknown') for doc in response.get("context", [])])
        for source in sources:
            print(f"- {source}")
            
    except Exception as e:
        print(f"❌ Error: {e}\n(Make sure the Ollama app is currently running on your computer!)")
