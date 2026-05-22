
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
└──> [Query Input] ──> [Advanced MMR Deconstruction] ──> [Local Llama 3 Inference]

```

### 1. Ingestion & Automated Normalization (`updated_ingest.py`)
* **Recursive Multi-Folder Traversal:** Powered by an `os.walk` loop that deep-dives into infinite layers of nested subfolders to safely exhaustively extract files.
* **Format-Specific Structural Extraction:** Natively ingests and extracts unstructured text fragments from raw `.pdf`, `.txt`, `.csv`, `.json`, and `.jsonl` files.
* **Byte-Level Data Cleaning:** Pre-processes dirty extractions by stripping toxic surrogate byte markers, eliminating repeated typographic formatting metadata wrappers (e.g., lines of `───`), and collapsing excessive layout space into clean paragraphs.
* **Semantic Target Chunking:** Splits text elements utilizing a step-by-step breakdown array (`["\n\n", "\n", ". ", "? ", "! "]`) to protect the integrity of single logical medical ideas or tabular data rows.
* **Atomic Idempotency Signatures:** Generates binary MD5 checksum hashes for input sources to prevent computational overhead during repetitive runs.

### 2. Deconstruction Retrieval & Inference Loop (`new_rag_terminal.py`)
* **Multi-Question Splitter Layer:** The prompt framework targets complex, multi-layered queries. If a clinician packs 3 to 4 distinct medical questions into one text prompt, the execution chain automatically isolates each constraint to build distinct logical sub-answers.
* **Diverse Maximal Marginal Relevance (MMR):** Executes an MMR search sequence pulling a context pool of `fetch_k=20` raw vector blocks, filtering them down to the `k=8` most unique blocks to completely prevent single textbooks from visually dominating context injections.
* **Isolated Local Execution Engine:** Leverages `Ollama` running local Meta `Llama 3` weights, calculating analytical responses entirely on device without cloud connections.

---

## 🛠️ Technical Specifications

* **Core Orchestration:** LangChain Ecosystem (Core / Community / Classic Engine Divisions)
* **Storage Provider:** ChromaDB (Vector Index Instance Optimization)
* **Embedding Model:** HuggingFace `sentence-transformers/all-MiniLM-L6-v2` (L2 Normalized Vectors)
* **Inference Pipeline:** Ollama Local Runtime 
* **Target Foundation Weights:** Meta Llama 3 (8B Instruct Parameter Matrix, quantized)
* **Language Runtime:** Python 3.8+ (Thread Isolation Safe)

---

## 📥 Installation & Environment Setup

### 1. System Prerequisites
Ensure your local development environment has Python 3.8+ ready and **Ollama** deployed for internal background serving.

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
├── new_rag_terminal.py       # High-capacity structural QA terminal assistant
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

---

## 💬 Interactive Example Matrix

```text
🧑 User: What are the primary warning signs of an acute myocardial infarction, what is the initial immediate first-aid protocol, and can you list the absolute contraindications for administering fibrinolytic therapy?

🔎 Analyzing query, searching local database, and processing answers...

🤖 Assistant:
Based on the ingested clinical materials, here is the point-by-point breakdown of your multi-part query:

1. Primary Warning Signs of Acute Myocardial Infarction:
   - Substernal chest pressure, squeezing, or severe crushing pain lasting more than a few minutes.
   - Radiation of pain into the left arm, neck, jaw, or epigastric region.
   - Associated symptoms including diaphoresis (profuse sweating), unexplained dyspnea (shortness of breath), and severe nausea/lightheadedness.

2. Initial Immediate First-Aid Protocol:
   - Call emergency medical services (EMS) immediately.
   - Position the patient in a comfortable, semi-reclined state to minimize cardiac workload.
   - Administer non-enteric coated chewable aspirin (162 mg to 325 mg) immediately if no allergy or active severe gastrointestinal bleeding is present.

3. Absolute Contraindications for Fibrinolytic Therapy:
   - Any prior history of intracranial hemorrhage.
   - Known structural cerebral vascular lesion (e.g., an arteriovenous malformation).
   - Ischemic stroke within the preceding 3 months (except acute ischemic stroke within 4.5 hours).
   - Active internal bleeding (excluding menses).
   - Suspected aortic dissection or closed head/facial trauma within 3 months.

Summary Analysis:
Immediate diagnostic and stabilization steps are crucial within the initial golden hour of chest pain onset. Fibrinolytics must strictly be gated behind rigorous contraindication verification to safeguard against lethal hemorrhagic stroke occurrences.

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
