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
# Updated path to point explicitly to the absolute path used in your ingest script
CHROMA_DIR = r"C:\Users\Shashwat\Desktop\internship\RAG\my_chroma_db" 

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
# Set a slightly higher context control or standard parameters for analytical processing
llm = ChatOllama(model="llama3", temperature=0)

# --- 4. CREATE THE ADVANCED STRUCTURAL RAG CHAIN ---
# This prompt forces Llama 3 to split multi-part user inputs and answer them individually
system_prompt = (
    "You are an expert, highly reliable medical assistant tasked with processing complex queries.\n"
    "The user may ask multiple distinct questions within a single query. You must address ALL of them.\n\n"
    "INSTRUCTIONS:\n"
    "1. Deconstruct the user's input into its individual sub-questions.\n"
    "2. Evaluate the provided context to answer each sub-question independently, accurately, and objectively.\n"
    "3. Structure your final response using clear bullet points or numbered sub-headings corresponding to each detected question.\n"
    "4. Synthesize your findings into a clear, reliable, and consolidated medical summary.\n"
    "5. Rely STRICTLY on the retrieved context below. If a specific sub-question cannot be answered using the context, state: "
    "'[Information not available in local data source]' for that specific part instead of making things up.\n\n"
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
print("🚀 ADVANCED MULTI-QUESTION LOCAL RAG SYSTEM READY")
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
