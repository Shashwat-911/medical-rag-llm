import os
import re
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="langchain_core._api")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.documents import Document

# --- 1. SET UP EMBEDDINGS ---
print("🔄 Loading embedding model: sentence-transformers/all-MiniLM-L6-v2...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

# --- 2. LOAD CHROMA DATABASE ---
CHROMA_DIR = r"C:\Users\Shashwat\Desktop\internship\RAG\my_chroma_db" 

if not os.path.exists(CHROMA_DIR):
    print(f"❌ Error: Folder '{CHROMA_DIR}' not found. Did you run your ingest script?")
    exit()

print(f"📂 Accessing vector database at '{CHROMA_DIR}'...")
vectordb = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 4, "fetch_k": 10}) 

# --- 3. SET UP LOCAL LLM ---
print("🧠 Connecting to local Ollama model (Llama 3)...")
llm = ChatOllama(model="llama3", temperature=0)

# --- 4. CONSTRUCT THE SYNTHESIS PROMPT ---
system_prompt = (
    "You are a precise clinical assistant answering a specific medical question using ONLY the provided text snippets.\n"
    "You have no outside medical knowledge. Do not use pre-trained background guidelines under any circumstances.\n\n"
    "STRICT PROTOCOLS:\n"
    "1. CLINICAL CONCISION: Provide a maximum of 2 short bullet points. Prioritize hard numbers, dosages, thresholds, and data over explanations.\n"
    "2. SCOPE-AWARE REFUSAL: If the provided text blocks do not contain the answer, explicitly state that the specific information requested is not present in the local database.\n\n"
    "Retrieved Context Snippets:\n{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

# Using native LangChain Expression Language (LCEL) for maximum control
synthesis_chain = prompt | llm

def deconstruct_query(query: str) -> list[str]:
    """Splits a complex multi-part query into individual questions using punctuation markers."""
    # Split on question marks or distinct sentence boundaries that look like questions
    sub_queries = [q.strip() + "?" for q in query.split("?") if q.strip()]
    if len(sub_queries) == 1 and "," in query:
        # Fallback split if they used commas instead of question marks
        sub_queries = [q.strip() for q in query.split(",") if q.strip()]
    return sub_queries

# --- 5. TERMINAL INTERFACE ---
print("\n" + "="*55)
print("🚀 MULTI-ROUTE SUB-QUERY RAG SYSTEM READY")
print("Type 'exit' or 'quit' to close the program.")
print("="*55 + "\n")

while True:
    user_query = input("\n🧑 User: ")
    
    if user_query.lower() in ['exit', 'quit']:
        print("Goodbye!")
        break
        
    if not user_query.strip():
        continue

    print("🔎 Planning query execution routes and generating sequential answers...")
    
    # Step A: Deconstruct the multi-part prompt
    sub_questions = deconstruct_query(user_query)
    all_sources = set()
    final_output = []

    # Step B: Run independent Retrieval and Generation for each sub-question
    for idx, sub_q in enumerate(sub_questions, start=1):
        print(f"  [Route {idx}/{len(sub_questions)}] Retrieving for: '{sub_q[:40]}...'")
        
        # Pull diverse chunks specific to THIS sub-question only
        retrieved_docs = retriever.get_relevant_documents(sub_q)
        context_str = "\n\n".join([doc.page_content for doc in retrieved_docs])
        
        # Track sources used across sub-queries
        for doc in retrieved_docs:
            all_sources.add(doc.metadata.get('source', 'Unknown'))
            
        # Run local generation for this specific part
        response = synthesis_chain.invoke({"context": context_str, "input": sub_q})
        
        # Format the combined output beautifully
        final_output.append(f"**Question: {sub_q}**\n{response.content.strip()}")

    # Step C: Print consolidated scannable response
    print("\n🤖 Assistant:\n" + "\n\n".join(final_output))
    
    print("\n📚 Sources:")
    for source in all_sources:
        print(f"- {source}")
