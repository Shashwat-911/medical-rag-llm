import os
import json
import uuid
import datetime
import streamlit as st
import warnings

# Suppress LangChain warnings
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain

import torch

# ── Paths Configuration ───────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_DIR = os.path.join(BASE_DIR, "chromadb")
if not os.path.exists(CHROMA_DIR) and os.path.exists(os.path.join(BASE_DIR, "my_chroma_db")):
    CHROMA_DIR = os.path.join(BASE_DIR, "my_chroma_db")
HISTORY_FILE = os.path.join(BASE_DIR, "chat_history.json")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Page Setup ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Emergency Medical Assistance",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom Minimalist CSS (ChatGPT-Inspired Aesthetic) ────────────────────────
st.markdown("""
<style>
    /* Global Styling */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    
    /* Main Layout Tweaks */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 5rem;
        max-width: 860px;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #f9f9f9;
        border-right: 1px solid #ececec;
    }
    
    .sidebar-header {
        display: flex;
        align-items: center;
        gap: 10px;
        font-size: 1.15rem;
        font-weight: 700;
        color: #1a1a1a;
        margin-bottom: 1.2rem;
        padding: 0.2rem 0;
    }
    
    .sidebar-section-title {
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #8e8ea0;
        margin: 1.2rem 0 0.5rem 0.2rem;
    }
    
    /* New Chat Button */
    div.stButton > button[key="new_chat_btn"] {
        width: 100%;
        background-color: #ffffff;
        color: #1a1a1a;
        border: 1px solid #e5e5e5;
        border-radius: 8px;
        padding: 0.6rem 1rem;
        font-weight: 500;
        font-size: 0.95rem;
        display: flex;
        align-items: center;
        justify-content: flex-start;
        transition: all 0.2s ease;
        box-shadow: 0 1px 2px rgba(0,0,0,0.03);
    }
    
    div.stButton > button[key="new_chat_btn"]:hover {
        background-color: #ececec;
        border-color: #d1d1d1;
    }
    
    /* Chat Item Buttons */
    div.stButton > button[key^="session_"] {
        width: 100%;
        text-align: left;
        background: transparent;
        border: none;
        border-radius: 6px;
        padding: 0.5rem 0.7rem;
        color: #333333;
        font-size: 0.88rem;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
        display: block;
        margin-bottom: 2px;
        transition: background-color 0.15s ease;
    }
    
    div.stButton > button[key^="session_"]:hover {
        background-color: #ebebeb;
    }
    
    /* Active Chat Item */
    div.stButton > button[key^="active_session_"] {
        width: 100%;
        text-align: left;
        background-color: #e5e5e5 !important;
        border: none;
        border-radius: 6px;
        padding: 0.5rem 0.7rem;
        color: #111111;
        font-weight: 600;
        font-size: 0.88rem;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
        display: block;
        margin-bottom: 2px;
    }

    /* Hero / Empty State Screen */
    .hero-container {
        text-align: center;
        padding-top: 14vh;
        padding-bottom: 3vh;
    }
    
    .hero-title {
        font-size: 2.2rem;
        font-weight: 600;
        color: #1a1a1a;
        margin-bottom: 0.6rem;
    }
    
    .hero-subtitle {
        font-size: 1.05rem;
        color: #666666;
        margin-bottom: 2.5rem;
    }
    
    /* Starter Prompt Cards */
    .prompt-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
        gap: 12px;
        margin-top: 1.5rem;
    }
    
    /* Chat Messages */
    .stChatMessage {
        border-radius: 12px;
        padding: 0.9rem 1.1rem;
        margin-bottom: 1rem;
    }
    
    /* Medical Alert Badge */
    .medical-badge {
        display: inline-flex;
        align-items: center;
        background-color: #fee2e2;
        color: #991b1b;
        font-size: 0.75rem;
        font-weight: 600;
        padding: 2px 8px;
        border-radius: 9999px;
        margin-bottom: 0.5rem;
    }
    
    /* Sources Box */
    .sources-box {
        background-color: #f8fafc;
        border-left: 3px solid #0284c7;
        padding: 0.6rem 0.9rem;
        border-radius: 4px;
        font-size: 0.82rem;
        color: #475569;
        margin-top: 0.8rem;
    }
    
    /* Status Badge */
    .db-status {
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.78rem;
        font-weight: 500;
    }
    .db-online {
        background-color: #dcfce7;
        color: #166534;
    }
    .db-building {
        background-color: #fef9c3;
        color: #854d0e;
    }
</style>
""", unsafe_allow_html=True)

# ── History Persistence ───────────────────────────────────────────────────────
def load_saved_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_history(history_data):
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history_data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Error saving history: {e}")

# ── State Initialization ──────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = load_saved_history()

if "current_session_id" not in st.session_state:
    # Set default session or create a new one
    if st.session_state.history:
        # Most recent session
        sorted_sessions = sorted(
            st.session_state.history.items(),
            key=lambda x: x[1].get("updated_at", ""),
            reverse=True
        )
        st.session_state.current_session_id = sorted_sessions[0][0]
    else:
        new_id = str(uuid.uuid4())
        st.session_state.history[new_id] = {
            "title": "New Chat",
            "created_at": datetime.datetime.now().isoformat(),
            "updated_at": datetime.datetime.now().isoformat(),
            "messages": []
        }
        st.session_state.current_session_id = new_id

# Helper to get current session messages
def get_current_messages():
    sid = st.session_state.current_session_id
    if sid in st.session_state.history:
        return st.session_state.history[sid].get("messages", [])
    return []

# Helper to create a new session
def create_new_session():
    new_id = str(uuid.uuid4())
    st.session_state.history[new_id] = {
        "title": "New Chat",
        "created_at": datetime.datetime.now().isoformat(),
        "updated_at": datetime.datetime.now().isoformat(),
        "messages": []
    }
    st.session_state.current_session_id = new_id
    save_history(st.session_state.history)

# Helper to delete session
def delete_session(sid):
    if sid in st.session_state.history:
        del st.session_state.history[sid]
        save_history(st.session_state.history)
        if st.session_state.history:
            st.session_state.current_session_id = list(st.session_state.history.keys())[0]
        else:
            create_new_session()

# ── Cached RAG Backend Components ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": DEVICE},
        encode_kwargs={"normalize_embeddings": True},
    )

def build_rag_chain(model_name="llama3", temperature=0.0, top_k=8):
    if not os.path.exists(CHROMA_DIR):
        return None, "Database directory not found. Please ensure ingest pipeline has completed."

    try:
        embeddings = load_embeddings()
        vectordb = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=embeddings
        )
        retriever = vectordb.as_retriever(
            search_type="mmr",
            search_kwargs={"k": top_k, "fetch_k": max(20, top_k * 2)}
        )
        llm = ChatOllama(model=model_name, temperature=temperature)
        
        system_prompt = (
            "You are a precise clinical assistant specialized in emergency medical assistance.\n"
            "Answer questions using ONLY the provided text snippets from the verified medical database.\n"
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
        return rag_chain, None
    except Exception as e:
        return None, str(e)

# ── Sidebar UI ────────────────────────────────────────────────────────────────
with st.sidebar:
    # App Brand Header
    st.markdown("""
        <div class="sidebar-header">
            <span>🩺 Emergency Medical Assistance</span>
        </div>
    """, unsafe_allow_html=True)
    
    # New Chat Button
    if st.button("➕ New chat", key="new_chat_btn", use_container_width=True):
        create_new_session()
        st.rerun()

    # Recents / History List
    st.markdown('<div class="sidebar-section-title">Recents</div>', unsafe_allow_html=True)
    
    sorted_sessions = sorted(
        st.session_state.history.items(),
        key=lambda x: x[1].get("updated_at", ""),
        reverse=True
    )
    
    for sid, sdata in sorted_sessions:
        title = sdata.get("title", "Untitled Chat")
        is_active = (sid == st.session_state.current_session_id)
        btn_key = f"active_session_{sid}" if is_active else f"session_{sid}"
        
        col_chat, col_del = st.columns([5, 1])
        with col_chat:
            # Display conversation button
            display_title = title if len(title) <= 24 else title[:22] + "..."
            if st.button(f"💬 {display_title}", key=btn_key, use_container_width=True):
                st.session_state.current_session_id = sid
                st.rerun()
        with col_del:
            if st.button("🗑️", key=f"del_{sid}", help="Delete chat"):
                delete_session(sid)
                st.rerun()

    st.markdown("---")
    
    # System Status & Config Drawer
    with st.expander("⚙️ System & Engine Settings"):
        model_name = st.selectbox("LLM Model", ["qwen2.5-coder:latest", "qwen2.5-coder:1.5b", "llama3", "llama3:latest", "mistral", "medllama2"], index=0)
        top_k = st.slider("Retrieved Context Chunks (k)", min_value=2, max_value=15, value=8)
        temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.0, step=0.05)
        
        st.markdown("---")
        db_exists = os.path.exists(CHROMA_DIR)
        if db_exists:
            st.markdown('<span class="db-status db-online">● Vector DB: Active</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="db-status db-building">⏳ Ingestion in progress...</span>', unsafe_allow_html=True)

# ── Main Content Area ─────────────────────────────────────────────────────────
current_messages = get_current_messages()

# If no messages in current session, render Hero Welcome Screen
if len(current_messages) == 0:
    st.markdown("""
        <div class="hero-container">
            <div class="hero-title">Ready when you are.</div>
            <div class="hero-subtitle">Ask anything about emergency clinical protocols, symptoms, or medical databases.</div>
        </div>
    """, unsafe_allow_html=True)
    
    # Quick suggestion cards
    col1, col2, col3 = st.columns(3)
    preset_query = None
    with col1:
        if st.button("🚨 Sepsis Protocol\n\nBaseline criteria & fluid resuscitation guidelines", use_container_width=True):
            preset_query = "What are the baseline clinical criteria used to identify severe sepsis in an adult patient, and what is the fluid resuscitation protocol?"
    with col2:
        if st.button("👶 Pediatric Emergency\n\nClinical differentiation: Croup vs Acute Epiglottitis", use_container_width=True):
            preset_query = "How do you clinically differentiate between pediatric croup and acute epiglottitis based on presentation?"
    with col3:
        if st.button("⚠️ Anaphylaxis Response\n\nFirst-line epinephrine dosing & management", use_container_width=True):
            preset_query = "What is the emergency management protocol and epinephrine dosage for severe acute anaphylaxis?"

    if preset_query:
        user_input = preset_query
    else:
        user_input = None
else:
    # Render existing conversation messages
    for msg in current_messages:
        role = msg.get("role")
        content = msg.get("content", "")
        sources = msg.get("sources", [])
        
        with st.chat_message(role, avatar="🧑" if role == "user" else "🩺"):
            st.markdown(content)
            if sources:
                with st.expander("📚 Verified Clinical Sources & Context", expanded=False):
                    for src in sources:
                        st.markdown(f"- `{src}`")
    user_input = None

# ── Chat Input ────────────────────────────────────────────────────────────────
chat_query = st.chat_input("Ask anything about emergency medical assistance...")
if chat_query:
    user_input = chat_query

# ── Handle Submission ─────────────────────────────────────────────────────────
if user_input:
    sid = st.session_state.current_session_id
    session = st.session_state.history[sid]
    
    # Update title if it's the first query
    if len(session["messages"]) == 0:
        # Create a short friendly title from prompt
        short_title = user_input.strip()
        if len(short_title) > 30:
            short_title = short_title[:28] + "..."
        session["title"] = short_title
    
    # Append User Message
    session["messages"].append({
        "role": "user",
        "content": user_input,
        "timestamp": datetime.datetime.now().isoformat()
    })
    session["updated_at"] = datetime.datetime.now().isoformat()
    save_history(st.session_state.history)
    
    # Display User Message in UI
    with st.chat_message("user", avatar="🧑"):
        st.markdown(user_input)
    
    # Generate Assistant Response
    with st.chat_message("assistant", avatar="🩺"):
        with st.spinner("Analyzing verified medical documents & clinical database..."):
            rag_chain, error_msg = build_rag_chain(
                model_name=model_name,
                temperature=temperature,
                top_k=top_k
            )
            
            if error_msg:
                response_text = f"⚠️ **Notice:** {error_msg}"
                sources_list = []
                st.warning(response_text)
            else:
                try:
                    res = rag_chain.invoke({"input": user_input})
                    response_text = res.get("answer", "No response generated.")
                    raw_sources = res.get("context", [])
                    
                    sources_list = list(set([
                        doc.metadata.get("source", "Unknown Document") for doc in raw_sources
                    ]))
                    
                    st.markdown(response_text)
                    if sources_list:
                        with st.expander("📚 Verified Clinical Sources & Context", expanded=False):
                            for src in sources_list:
                                st.markdown(f"- `{src}`")
                except Exception as e:
                    response_text = f"❌ An error occurred during retrieval/generation: {str(e)}"
                    sources_list = []
                    st.error(response_text)
    
    # Append Assistant Message to History
    session["messages"].append({
        "role": "assistant",
        "content": response_text,
        "sources": sources_list,
        "timestamp": datetime.datetime.now().isoformat()
    })
    session["updated_at"] = datetime.datetime.now().isoformat()
    save_history(st.session_state.history)
    
    st.rerun()
