# **🏥 Medical RAG LLM System 🚀**

### **Professional Clinical Decision Support via 100% Local Retrieval-Augmented Generation**

This repository contains a **Medical RAG (Retrieval-Augmented Generation)** system designed to provide high-accuracy, grounded answers to complex medical queries. Unlike standard AI chatbots that rely on internal memory and are prone to hallucinations, this system is strictly anchored to a curated local library of high-stakes clinical guidelines, first-aid manuals, and medical textbooks.

To ensure maximum patient privacy and data security, **this entire pipeline runs 100% locally**. No data or queries are ever sent to the cloud.

## **🧠 How It Works: The "Grounding First" Architecture**

The system operates on a strict retrieval-first principle. If the answer is not in the ingested clinical text, the AI is instructed to honestly reply, "I don't know."

### **1\. Ingestion & Vectorization (updated\_ingest.py)**

* **Multi-Format Parsing:** The system seamlessly reads local .pdf, .txt, .csv, .json, and .jsonl files.  
* **Optimized Chunking:** Medical texts are dense. Documents are intelligently broken into **1200-character segments** with a **200-character overlap** to preserve critical clinical context.  
* **Embedding:** Using the sentence-transformers/all-MiniLM-L6-v2 model via HuggingFace, text is converted into high-dimensional mathematical vectors directly on the CPU.  
* **Storage:** These vectors are stored permanently in a local **ChromaDB** instance.

### **2\. Retrieval & Inference (new\_rag\_terminal.py)**

* **Query Transformation:** User questions are converted into vectors using the same embedding model.  
* **Advanced MMR Search:** The system uses **Maximal Marginal Relevance (MMR)** to fetch the top 20 most relevant chunks, then mathematically filters them down to the 8 *most diverse* chunks. This prevents a single textbook from dominating the search results.  
* **Augmented Prompting:** These diverse medical segments are injected into a strict system prompt.  
* **Local Generation:** The **Llama 3** model (running locally via Ollama) reads the provided context and generates a concise, evidence-based answer complete with source citations.

## **🛠️ Technical Stack**

* **Orchestration:** LangChain (Classic & Core)  
* **Vector Database:** ChromaDB  
* **Embeddings:** HuggingFace (all-MiniLM-L6-v2)  
* **Inference Engine:** Ollama (Local Execution)  
* **LLM:** Meta Llama 3  
* **Language:** Python 3.8+

## **📥 Installation & Setup**

### **1\. Prerequisites**

You must have Python installed, as well as **Ollama** to run the Llama 3 model locally.

1. Download and install [Ollama](https://ollama.com/download).  
2. Open your terminal and pull the Llama 3 model:  
   ollama run llama3

   *(Once it downloads and opens a chat prompt, type /bye to exit).*

### **2\. Clone and Install Dependencies**

git clone \[https://github.com/YOUR\_USERNAME/medical-rag-llm.git\](https://github.com/YOUR\_USERNAME/medical-rag-llm.git)  
cd medical-rag-llm

\# Install required packages  
pip install langchain langchain-community langchain-core langchain-text-splitters langchain-chroma langchain-huggingface langchain-classic chromadb sentence-transformers pypdf

### **3\. Folder Structure**

Ensure your project directory is set up as follows:

├── db/                     \# Place all your raw clinical PDFs, TXTs, and JSONL files here  
├── my\_chroma\_db/           \# Auto-generated vector database (Do not create manually)  
├── updated\_ingest.py       \# Script to build the vector database  
├── new\_rag\_terminal.py     \# Terminal interface to chat with the AI  
└── README.md

## **🚀 Usage**

### **Phase 1: Build the Knowledge Base**

Place your medical files into the db folder, then run the ingestion script to build the vector database:

python updated\_ingest.py

### **Phase 2: Launch the System**

Once ingestion is complete and your database is ready, start the interactive terminal session:

python new\_rag\_terminal.py

## **💬 Example Output**

🧑 User: According to the latest guidelines, what is the recommended time window for administering intravenous thrombolysis in a patient with acute ischemic stroke?

🔎 Searching local database and generating answer...

🤖 Assistant: According to the latest guidelines (Prabhakaran et al 2026 Acute Ischemic Stroke Guideline), the recommended time window for administering intravenous thrombolysis (IVT) in a patient with acute ischemic stroke is within 4.5 hours of symptom onset or last known well.

📚 Sources:  
\- prabhakaran-et-al-2026-2026-guideline-for-the-early-management-of-patients-with-acute-ischemic-stroke-a-guideline-from.pdf

## **🎓 Academic Context**

**Developed by:** Shashwat Yadav

**Context:** VTU 5th Semester Engineering Internship Project

## **⚠️ Medical Disclaimer**

This tool is for educational and research purposes only. It is intended to demonstrate the capabilities of local RAG systems in specialized domains. The AI system provided in this repository is **not** a substitute for professional medical advice, diagnosis, or treatment, and should not be used in actual clinical settings.