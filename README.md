# Medical RAG LLM System 🚀

### Professional Clinical Decision Support via Retrieval-Augmented Generation

This repository contains a **Medical RAG (Retrieval-Augmented Generation)** system designed to provide high-accuracy, grounded answers to medical queries. Unlike standard AI, which relies on "internal memory," this system is strictly anchored in a curated library of high-stakes clinical guidelines and medical handbooks from 2020–2026.

---

## 🧠 How It Works: The RAG Architecture
The system operates on a **"Grounding First"** principle to prevent medical hallucinations. Here is the technical workflow:

### 1. Ingestion & Vectorization (`ingest.py`)
* **Parsing:** The system reads local PDF files (clinical manuals, research papers, guidelines).
* **Chunking:** Documents are broken into 500-character segments with a 50-character overlap to preserve local context.
* **Embedding:** Using the `sentence-transformers/all-MiniLM-L6-v2` model, text is converted into high-dimensional mathematical vectors.
* **Storage:** These vectors are stored in a local **ChromaDB** instance.

### 2. Retrieval & Inference (`rag_terminal.py`)
* **Query Transformation:** When a user asks a question, the system converts that question into a vector using the same embedding model.
* **Similarity Search:** The system performs a "nearest neighbor" search in ChromaDB to find the 3 most relevant segments of text from your medical PDFs.
* **Augmented Prompting:** These 3 segments are injected into a system prompt along with the user's question.
* **Generation:** The **Llama-3.3-70b-Versatile** model (via **Groq**) reads the provided context and generates a concise, evidence-based answer. If the answer is not in the text, it is instructed to say "I don't know."

---

## 🛠️ Technical Stack
* **Orchestration:** LangChain (Classic & Core)
* **Vector Database:** ChromaDB
* **Embeddings:** HuggingFace (`all-MiniLM-L6-v2`)
* **Inference Engine:** Groq Cloud API
* **LLM:** Llama-3.3-70b-Versatile
* **Language:** Python 3.14

---

## 📥 Installation & Setup

### 1. Prerequisites
* Python 3.10+
* A Groq API Key (Obtain at [console.groq.com](https://console.groq.com/))

### 2. Clone and Install

bash
git clone [https://github.com/YOUR_USERNAME/medical-rag-llm.git](https://github.com/YOUR_USERNAME/medical-rag-llm.git)
cd medical-rag-llm

# Install required packages
pip install -r requirements.txt 

### 
3. Build the Knowledge Base
Place your medical PDFs in the root directory and run the ingestion script:
python ingest.py

4. Launch the System
   python rag_terminal.py

⚠️ Disclaimer
This tool is for educational and research purposes only. It is intended to demonstrate the capabilities of RAG systems in specialized domains and should not be used for actual clinical diagnosis or treatment.

Developed by: Shashwat Yadav

Academic Context: VTU 5th Semester Engineering Internship Project


