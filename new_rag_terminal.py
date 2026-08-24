import os
import torch
# Suppress unnecessary logging
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="langchain_core._api")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
# --- FIXED: Use modern langchain-ollama package to clear deprecation alerts ---
from langchain_ollama import ChatOllama
# --- UPDATED IMPORTS FOR LANGCHAIN V1 ---
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- 1. SET UP EMBEDDINGS ---
if DEVICE == "cuda":
    print(f"⚡ GPU Active: {torch.cuda.get_device_name(0)}")
else:
    print("ℹ️ Running embeddings on CPU")

print("🔄 Loading embedding model: sentence-transformers/all-MiniLM-L6-v2...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": DEVICE},
    encode_kwargs={"normalize_embeddings": True},
)

# --- 2. LOAD CHROMA DATABASE ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_DIR = os.path.join(BASE_DIR, "chromadb")
if not os.path.exists(CHROMA_DIR) and os.path.exists(os.path.join(BASE_DIR, "my_chroma_db")):
    CHROMA_DIR = os.path.join(BASE_DIR, "my_chroma_db")

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
) 

# --- 3. SET UP LOCAL LLM (OLLAMA) ---
print("🧠 Connecting to local Ollama model (Llama 3)...")
llm = ChatOllama(model="qwen2.5-coder:latest", temperature=0)


# --- 4. CREATE THE CONSOLIDATED ADVANCED RAG CHAIN ---
# Completely cleaned and streamlined to force strict document extraction limits
system_prompt = (
    "You are a precise clinical assistant that answers multi-part questions using ONLY the provided text snippets.\n"
    "You have no outside medical knowledge. Do not use pre-trained background guidelines under any circumstances.\n\n"
    "STRICT EXECUTION PROTOCOLS:\n"
    "1. BULLETED ISOLATION: Break the user's input down. Address each sub-question independently under its own clear, bold Markdown line matching that specific question.\n"
    "2. CLINICAL CONCISION: Provide a maximum of 2 short bullet points per sub-question. Prioritize hard numbers, dosages, thresholds, and data over explanations.\n"
    "3. SCOPE-AWARE REFUSAL: If the provided text blocks do not contain the answer to a sub-question, explicitly state that the specific information requested is not present in the local database. Keep the question scope intact (e.g., 'The exact fluid resuscitation protocol for severe sepsis is not detailed in the local sources').\n\n"
    "Retrieved Context Snippets:\n{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

combine_docs_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, combine_docs_chain)

# --- 5. TERMINAL INTERFACE ---
print("\n" + "="*55)
print("🚀 ADVANCED CONSOLIDATED LOCAL RAG SYSTEM READY")
print("Type 'exit' or 'quit' to close the program.")
print("="*55 + "\n")

while True:
    user_query = input("\n🧑 User: ")
    
    if user_query.lower() in ['exit', 'quit']:
        print("Goodbye!")
        break
        
    if not user_query.strip():
        continue

    print("🔎 Analyzing query, searching local database, and processing answers...")
    
    try:
        response = rag_chain.invoke({"input": user_query})
        print(f"\n🤖 Assistant:\n{response['answer']}")
        
        # Print the filenames it used to answer
        print("\n📚 Sources:")
        sources = set([doc.metadata.get('source', 'Unknown') for doc in response.get("context", [])])
        for source in sources:
            print(f"- {source}")
            
    except Exception as e:
        print(f"❌ Error: {e}\n(Make sure the Ollama app is currently running on your computer!)")