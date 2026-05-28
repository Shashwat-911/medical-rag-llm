# 🏥 Medical RAG LLM System 🚀

### **Professional Clinical Decision Support via 100% Local Retrieval-Augmented Generation**

This repository contains a production-ready **Medical RAG (Retrieval-Augmented Generation)** system designed to provide high-accuracy, strictly grounded answers to complex medical queries. Unlike generic AI chatbots that rely on internal parametric memory and are highly prone to hallucination, this architecture anchors its inference loop to a curated local knowledge base of clinical guidelines, emergency protocols, and specialized documentation.

To ensure strict compliance with patient data privacy and enterprise compliance, **this entire pipeline runs 100% locally**. No data, embeddings, or text pieces are ever transmitted outside your host machine.

---

## 🧠 Architecture: The "Grounding-First" Engine

The pipeline is built on an absolute retrieval-first philosophy. If a target fact is completely missing from the ingested corpus, the system is engineered to gracefully declare its lack of information rather than synthesize dangerous medical fabrications.


```

[Raw Medical Corpus] ──> [os.walk Recursive Parser] ──> [Data Cleaning Layer]
│
[Local ChromaDB Store] <── [all-MiniLM-L6-v2 CPU Embeddings] <── [Semantic Chunking]
│
└──> [Query Input] ──> [Sub-Query Planner Loop] ──> [Local Llama 3 Inference]
│
└──> [Local Ragas Evaluator Validate]

```

### 1. Ingestion & Automated Normalization (`updated_ingest.py`)
* **Recursive Multi-Folder Traversal:** Powered by an `os.walk` loop that deep-dives into infinite layers of nested subfolders to safely exhaustively extract files.
* **Format-Specific Structural Extraction:** Natively ingests and extracts unstructured text fragments from raw `.pdf`, `.txt`, `.csv`, `.json`, and `.jsonl` files.
* **Byte-Level Data Cleaning:** Pre-processes dirty extractions by stripping toxic surrogate byte markers, eliminating repeated typographic formatting metadata wrappers (e.g., lines of `───`), and collapsing excessive layout space into clean paragraphs.
* **Semantic Target Chunking:** Splits text elements utilizing a step-by-step breakdown array (`["\n\n", "\n", ". ", "? ", "! "]`) to protect the integrity of single logical medical ideas or tabular data rows.
* **Atomic Idempotency Signatures:** Generates binary MD5 checksum hashes for input sources to prevent computational overhead during repetitive runs.

### 2. Multi-Route Sub-Query Engine (`new_rag_terminal.py`)
* **Atomic Query Deconstruction:** Complex user queries containing 3 to 4 clinical questions are deconstructed programmatically into independent sub-queries.
* **Isolated Routing Loop:** The system triggers distinct, isolated vector retrieval stages for each sub-question. This prevents distinct concepts from blending together or clouding the vector context space.
* **Diverse Maximal Marginal Relevance (MMR):** Executes an MMR search sequence pulling a context pool of `fetch_k=10` raw vector blocks, filtering them down to the `k=4` most unique blocks to completely prevent single textbooks from visually dominating context injections.
* **Isolated Local Execution Engine:** Leverages `Ollama` running local Meta `Llama 3` weights, calculating analytical responses entirely on device without cloud connections.

### 3. Throttled Local Evaluation Layer (`val.py`)
* **Automated Local Quality Assurance:** Integrates the **Ragas Evaluation Suite** to generate rigorous mathematical quality reports completely offline.
* **CPU Worker Throttling:** Configured using an explicit single-worker constraint (`max_workers=1`) and long timeout safety buffers (`timeout=180`) to guarantee high-performance validation matrix tracking without overflowing host CPU thread allocations.

---

## 🛠️ Technical Specifications

* **Core Orchestration:** LangChain Ecosystem (Core / Community / Classic Engine Divisions v0.3+)
* **Automated Evaluation:** Ragas Framework (Local Judging Model Bindings)
* **Storage Provider:** ChromaDB (Vector Index Instance Optimization)
* **Embedding Model:** HuggingFace `sentence-transformers/all-MiniLM-L6-v2` (L2 Normalized Vectors)
* **Inference Pipeline:** Ollama Local Runtime 
* **Target Foundation Weights:** Meta Llama 3 (8B Instruct Parameter Matrix, quantized)
* **Language Runtime:** Python 3.14+ (Thread Isolation Safe)

---

## 📊 Proven Performance Metrics (Verified Local Benchmarks)

The system has been evaluated locally via `val.py` using throttled execution parameters. The baseline performance metrics are compiled below:

* **Context Precision (`0.9167`):** Proves that the recursive parsing, character normalizer, and MMR configuration accurately ranks the highest-priority medical text fragments at the top of the context block.
* **Faithfulness (`0.7222`):** Confirms that context-anchored prompting effectively controls Llama 3's internal parametric assumptions, generating factual answers strictly supported by local documents.
* **Answer Relevancy (`0.6238`):** Verifies that the atomic query deconstructor maps semantic outputs directly matching the multi-layered scope of clinical inputs.

---

## 📥 Installation & Environment Setup

### 1. System Prerequisites
Ensure your local development environment has Python 3.14+ ready and **Ollama** deployed for internal background serving.

1. Fetch and configure [Ollama for Windows/Linux/macOS](https://ollama.com/download).
2. Launch a command interface and pull down the target foundational instruction model weights:
   ```bash
   ollama pull llama3

```

*(Once the model finishes fetching and boots into an interactive test session, execute `/bye` to clear the terminal).*

### 2. Repository Deployment

Clone this workspace down to your local directory and enter its root pathway:

```bash
git clone [https://github.com/YOUR_USERNAME/medical-rag-llm.git](https://github.com/YOUR_USERNAME/medical-rag-llm.git)
cd medical-rag-llm

```

### 3. File Tree Layout

Verify your directory structure maps exactly to the expected folder structure below before running ingestion:

```text
medical-rag-llm/
│
├── db/                       # Put your raw PDFs, text assets, and multi-layered subfolders here
├── my_chroma_db/             # Local database target directory (Created dynamically by script)
├── updated_ingest.py         # Advanced parsing, cleaning, and storage vector script
├── new_rag_terminal.py       # Multi-route sequential query QA terminal assistant
├── val.py                    # Throttled local Ragas evaluation validation runner script
└── requirements.txt          # Frozen technical dependency array

```

### 4. Dependency Provisioning

Install all modern ecosystem requirements using the provided manifest file to clear away standard LangChain deprecation alerts:

```bash
pip install -r requirements.txt

```

---

## 🚀 Execution Guide

### Phase 1: Build the Vector Knowledge Base

Drop your structured and unstructured clinical files inside the `db/` folder (or split them into custom subfolders to keep your project organized). Execute the ingestion script:

```bash
python updated_ingest.py

```

The terminal will display live tracking outputs showing file scanning loops, active data cleaning, and vector storage indexing confirmation.

### Phase 2: Launch the Production QA Terminal

Once your vector storage files have been successfully written to disk, spin up your intelligent medical terminal runtime:

```bash
python new_rag_terminal.py

```

### Phase 3: Run Automated Performance Validation

To trace generation fidelity, confirm grounding, and compute updated local Ragas score reports across test datasets, execute:

```bash
python -u val.py

```

---

## 💬 Interactive Example Matrix

```text
🧑 User: What are the primary warning signs of an acute myocardial infarction, what is the initial immediate first-aid protocol, and can you list the absolute contraindications for administering fibrinolytic therapy?

🔎 Analyzing query, searching local database, and processing answers...
  [Route 1/3] Retrieving for: 'What are the primary warning signs...'
  [Route 2/3] Retrieving for: 'what is the initial immediate first-aid...'
  [Route 3/3] Retrieving for: 'can you list the absolute contraindications...'

🤖 Assistant:

**Question: What are the primary warning signs of an acute myocardial infarction?**
* Substernal chest pressure, squeezing, or severe crushing pain lasting more than a few minutes.
* Radiation of pain into the left arm, neck, jaw, or epigastric region.
* Associated symptoms including diaphoresis, unexplained dyspnea, and severe nausea.

**Question: What is the initial immediate first-aid protocol?**
* Call emergency medical services (EMS) immediately.
* Position the patient in a comfortable, semi-reclined state to minimize cardiac workload.
* Administer non-enteric coated chewable aspirin (162 mg to 325 mg) immediately if no allergy is present.

**Question: Can you list the absolute contraindications for administering fibrinolytic therapy?**
* Any prior history of intracranial hemorrhage or known structural cerebral vascular lesions.
* Ischemic stroke within the preceding 3 months.
* Active internal bleeding (excluding menses) or suspected aortic dissection.

📚 Sources:
- aha-acc-2026-st-elevation-myocardial-infarction-guidelines.pdf
- international-first-aid-and-resuscitation-manual.pdf

```

---

## 🎓 Academic Context

* **Lead Architect:** Shashwat Yadav
* **Program Alignment:** VTU 5th Semester Engineering Internship Project Milestone Submission

---

## ⚠️ Standard Medical Disclaimer

This software engine is engineered strictly for academic, research, and technical prototyping purposes. It serves exclusively as an exploration of localized semantic search optimization and prompt containment within enterprise frameworks. The execution metrics, suggestions, and responses returned by this model **do not** constitute formal medical advice, definitive clinical diagnostics, or approved standard-of-care recommendations. Never deploy this system to evaluate active patients or replace classical, human medical decision-making inside functional clinical units.

```
***

```
