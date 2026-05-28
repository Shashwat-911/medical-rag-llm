"""
val.py — Multi-Route Local Ragas Evaluation Suite
Fixes: Formally transitions retriever.get_relevant_documents() to retriever.invoke()
to completely bypass Python 3.14/LangChain v0.3 attribute crashes.
"""

import os
import asyncio
from datasets import Dataset
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

# --- Modern Ragas & Run Config Frameworks ---
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig

# ── 1. INITIALIZE LOCAL EMBEDDINGS & COMPONENT LINKS ────────────────────────
CHROMA_DIR = r"C:\Users\Shashwat\Desktop\internship\RAG\my_chroma_db"

print("🔄 Loading Local Embedding Weights...")
local_embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)
ragas_embeddings = LangchainEmbeddingsWrapper(local_embeddings)

vectordb = Chroma(persist_directory=CHROMA_DIR, embedding_function=local_embeddings)
# Explicitly configured retriever instance using Maximal Marginal Relevance (MMR)
retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 4, "fetch_k": 10})

print("🧠 Initializing Local Ragas Judging Models...")
eval_llm = ChatOllama(model="llama3", temperature=0, timeout=300)
ragas_llm = LangchainLLMWrapper(eval_llm)

system_prompt = (
    "You are a precise clinical assistant answering a specific medical question using ONLY the provided text snippets.\n"
    "You have no outside medical knowledge. Do not use pre-trained background guidelines under any circumstances.\n\n"
    "STRICT PROTOCOLS:\n"
    "1. CLINICAL CONCISION: Provide a maximum of 2 short bullet points. Prioritize data points over text descriptions.\n"
    "2. SCOPE-AWARE REFUSAL: If the text blocks do not contain the answer, state that the information is not present.\n\n"
    "Retrieved Context Snippets:\n{context}"
)
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])
synthesis_chain = prompt | eval_llm

# ── 2. FLATTENED ATOMIC TEST DATASET CONFIGURATION ──────────────────────────
eval_dataset = [
    {
        "sub_question": "What are the baseline clinical criteria used to identify severe sepsis in an adult patient?",
        "ground_truth": "Criteria include Systolic BP <90 mmHg, HR >90/min, RR >20/min, and GCS <15."
    },
    {
        "sub_question": "What is the fluid resuscitation protocol for an adult sepsis patient?",
        "ground_truth": "Fluid resuscitation requires aggressive volume targeting to restore systemic perfusion and blood pressure."
    },
    {
        "sub_question": "How do you clinically differentiate between pediatric croup and acute epiglottitis based on presentation?",
        "ground_truth": "Croup presents with a characteristic barking cough and hoarseness. Epiglottitis presents with sudden severe respiratory distress, high fever, and active drooling."
    }
]

# ── 3. DATA ACQUISITION DECONSTRUCTION RUN ──────────────────────────────────
print("\n🚀 Phase 1: Generating Route-Isolated RAG outputs across test array...")
questions = []
answers = []
contexts = []
ground_truths = []

for item in eval_dataset:
    sub_q = item["sub_question"]
    print(f"  [Executing Query Planner Route] -> '{sub_q[:50]}...'")
    
    # ── FIXED HERE: Replaced get_relevant_documents with native invoke ──
    retrieved_docs = retriever.invoke(sub_q)
    context_str = "\n\n".join([doc.page_content for doc in retrieved_docs])
    
    # Run the isolated local generation pass
    response = synthesis_chain.invoke({"context": context_str, "input": sub_q})
    
    questions.append(sub_q)
    answers.append(response.content.strip())
    contexts.append([doc.page_content for doc in retrieved_docs])
    ground_truths.append(item["ground_truth"])

# Build clean dataset dictionary using legacy Ragas data slots
data_dict = {
    "question": questions,
    "answer": answers,
    "contexts": contexts,
    "ground_truth": ground_truths
}
dataset = Dataset.from_dict(data_dict)

# ── 4. RUN VALIDATION CONFIGURATION ──────────────────────────────────────────
print("\n📊 Phase 2: Launching Local Ragas Evaluator (Throttled CPU Compute Mode)...")

cpu_run_config = RunConfig(timeout=180, max_workers=1)

faithfulness.llm = ragas_llm
faithfulness.run_config = cpu_run_config

answer_relevancy.llm = ragas_llm
answer_relevancy.run_config = cpu_run_config

context_precision.llm = ragas_llm
context_precision.run_config = cpu_run_config

metrics = [faithfulness, answer_relevancy, context_precision]

results = evaluate(
    dataset=dataset,
    metrics=metrics,
    llm=ragas_llm,
    embeddings=ragas_embeddings,
    run_config=cpu_run_config
)

print("\n" + "="*60)
print("🎯 FINAL SUB-QUERY LOCAL RAGAS SCORE REPORT")
print("="*60)
print(results)
print("="*60)
