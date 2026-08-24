"""
val.py — 100% Local Ragas Evaluation Suite
Optimized for: Stable CPU execution using thread throttling and custom timeout extensions.
"""

import os
import asyncio
from datasets import Dataset
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain

# --- Ragas Namespace Imports ---
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from ragas.llms import LangchainLLMWrapper
import torch
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── 1. INITIALIZE LOCAL EMBEDDINGS & RAG COMPONENTS ──────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_DIR = os.path.join(BASE_DIR, "chromadb")
if not os.path.exists(CHROMA_DIR) and os.path.exists(os.path.join(BASE_DIR, "my_chroma_db")):
    CHROMA_DIR = os.path.join(BASE_DIR, "my_chroma_db")

print(f"🔄 Loading Local Embedding Weights (device: {DEVICE})...")
local_embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": DEVICE},
    encode_kwargs={"normalize_embeddings": True},
)

ragas_embeddings = LangchainEmbeddingsWrapper(local_embeddings)

vectordb = Chroma(persist_directory=CHROMA_DIR, embedding_function=local_embeddings)
retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 5, "fetch_k": 15})

print("🧠 Initializing Local Ragas Judging Models...")
# Explicitly set a higher timeout threshold on the client connection
eval_llm = ChatOllama(model="llama3", temperature=0, timeout=300)
ragas_llm = LangchainLLMWrapper(eval_llm)

# Recreate your system prompt structure
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
combine_docs_chain = create_stuff_documents_chain(eval_llm, prompt)
rag_chain = create_retrieval_chain(retriever, combine_docs_chain)

# ── 2. TEST DATASET CONFIGURATION ───────────────────────────────────────────
eval_dataset = [
    {
        "question": "What are the baseline clinical criteria used to identify severe sepsis in an adult patient, and what is the fluid resuscitation protocol?",
        "ground_truth": "Criteria include Systolic BP <90 mmHg, HR >90/min, RR >20/min, and GCS <15. Fluid resuscitation requires aggressive volume targeting to restore systemic perfusion."
    },
    {
        "question": "How do you clinically differentiate between pediatric croup and acute epiglottitis based on presentation?",
        "ground_truth": "Croup presents with a characteristic barking cough and hoarseness. Epiglottitis presents with sudden severe respiratory distress, high fever, and active drooling."
    }
]

# ── 3. DATA ACQUISITION & EXECUTION TRACE ───────────────────────────────────
print("\n🚀 Phase 1: Generating RAG outputs across evaluation dataset...")
questions = []
answers = []
contexts = []
ground_truths = []

for item in eval_dataset:
    print(f"  [Invoking Chain] -> {item['question'][:50]}...")
    response = rag_chain.invoke({"input": item["question"]})
    
    questions.append(item["question"])
    answers.append(response["answer"])
    retrieved_chunks = [doc.page_content for doc in response.get("context", [])]
    contexts.append(retrieved_chunks)
    ground_truths.append(item["ground_truth"])

data_dict = {
    "question": questions,
    "answer": answers,
    "contexts": contexts,
    "ground_truth": ground_truths
}
dataset = Dataset.from_dict(data_dict)

# ── 4. CONFIGURE RUN CONFIG & EXECUTE VALIDATION ────────────────────────────
print("\n📊 Phase 2: Launching Local Ragas Evaluator (Throttled CPU Compute Mode)...")

# CRITICAL CPU FIX: Define a strict execution constraint layout
# max_workers=1 stops parallel jobs from slamming your CPU at the same time
# timeout=180 gives your CPU a generous 3-minute window per evaluation sentence
cpu_run_config = RunConfig(
    timeout=180,
    max_workers=1
)

# Apply configuration weights to each chosen metric handler
faithfulness.llm = ragas_llm
faithfulness.run_config = cpu_run_config

answer_relevancy.llm = ragas_llm
answer_relevancy.run_config = cpu_run_config

context_precision.llm = ragas_llm
context_precision.run_config = cpu_run_config

metrics = [faithfulness, answer_relevancy, context_precision]

# Run evaluation with the run_config passed into the core engine
results = evaluate(
    dataset=dataset,
    metrics=metrics,
    llm=ragas_llm,
    embeddings=ragas_embeddings,
    run_config=cpu_run_config
)

print("\n" + "="*60)
print("🎯 FINAL LOCAL RAGAS SCORE REPORT")
print("="*60)
print(results)
print("="*60)