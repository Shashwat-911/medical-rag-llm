Medical RAG LLM System
🧬 Professional Clinical Decision Support via Retrieval-Augmented Generation
This repository contains a Medical RAG (Retrieval-Augmented Generation) system designed to provide high-accuracy, grounded answers to medical queries. Unlike standard AI, which relies on "internal memory," this system is strictly anchored in a curated library of high-stakes clinical guidelines and medical handbooks from 2020–2026.

🧠 How It Works: The RAG Architecture
The system operates on a "Grounding First" principle to prevent medical hallucinations. Here is the technical workflow:

Ingestion & Vectorization (ingest.py):

Parsing: The system reads local PDF files (clinical manuals, research papers, guidelines).

Chunking: Documents are broken into 500-character segments with a 50-character overlap to preserve local context.

Embedding: Using the sentence-transformers/all-MiniLM-L6-v2 model, text is converted into high-dimensional mathematical vectors.

Storage: These vectors are stored in a local ChromaDB instance.

Retrieval & Inference (rag_terminal.py):

Query Transformation: When a user asks a question, the system converts that question into a vector using the same embedding model.

Similarity Search: The system performs a "nearest neighbor" search in ChromaDB to find the 3 most relevant segments of text from your medical PDFs.

Augmented Prompting: These 3 segments are injected into a system prompt along with the user's question.

Generation: The Llama-3.3-70b-Versatile model (via Groq) reads the provided context and generates a concise, evidence-based answer. If the answer is not in the text, it is instructed to say "I don't know."

🛠️ Technical Stack
Orchestration: LangChain (Classic & Core)

Vector Database: ChromaDB

Embeddings: HuggingFace (all-MiniLM-L6-v2)

Inference Engine: Groq Cloud API

LLM: Llama-3.3-70b-Versatile

Language: Python 3.14

📥 Installation & Setup
1. Prerequisites
Python 3.10+

A Groq API Key (Obtain at console.groq.com)

2. Clone and Install
Bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/medical-rag-llm.git
cd medical-rag-llm

# Install required packages
pip install -r requirements.txt
3. Data Preparation
Place your medical PDFs in the root directory. Your current library includes:

Prabhakaran et al. 2026 Stroke Guidelines

GINA 2025 Asthma Update

IFRC International First Aid Guidelines 2025

Medical Oncology Handbook (2020)

4. Build the Knowledge Base
Run the ingestion script once to process your files:

Bash
python ingest.py
5. Launch the System
Run the terminal interface to start chatting with your medical database:

Bash
python rag_terminal.py
📊 Evaluation & Testing
To demonstrate the system's effectiveness, try the following types of queries:

Specific Guideline Retrieval: "What are the 2026 recommendations for pediatric stroke imaging?"

Comparative Analysis: "Compare the GINA 2025 asthma step-treatment to previous years."

Safety Check: Ask a question about a topic not in your PDFs. The system should refuse to answer, proving it is not hallucinating.

⚠️ Disclaimer
This tool is for educational and research purposes only. It is intended to demonstrate the capabilities of RAG systems in specialized domains and should not be used for actual clinical diagnosis or treatment.

📜 License
Distributed under the MIT License. See LICENSE for more information.

Developed by: Shashwat Yadav

Academic Context: VTU 6th Semester Engineering Project
