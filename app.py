import os
import json
import uuid
import datetime
import zipfile
import urllib.request
import streamlit as st
import warnings
import torch

# Suppress LangChain warnings
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain

# Optional / Dynamic Provider Imports
try:
    from langchain_ollama import ChatOllama
except ImportError:
    ChatOllama = None

try:
    from langchain_groq import ChatGroq
except ImportError:
    ChatGroq = None

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except ImportError:
    ChatGoogleGenerativeAI = None

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
        padding-top: 10vh;
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
    
    /* Chat Messages */
    .stChatMessage {
        border-radius: 12px;
        padding: 0.9rem 1.1rem;
        margin-bottom: 1rem;
    }
    
    /* Status Badges */
    .db-status {
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.78rem;
        font-weight: 500;
        display: inline-block;
    }
    .db-online {
        background-color: #dcfce7;
        color: #166534;
    }
    .db-building {
        background-color: #fef9c3;
        color: #854d0e;
    }
    .db-cloud {
        background-color: #e0f2fe;
        color: #0369a1;
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
    if st.session_state.history:
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

def get_current_messages():
    sid = st.session_state.current_session_id
    if sid in st.session_state.history:
        return st.session_state.history[sid].get("messages", [])
    return []

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

def delete_session(sid):
    if sid in st.session_state.history:
        del st.session_state.history[sid]
        save_history(st.session_state.history)
        if st.session_state.history:
            st.session_state.current_session_id = list(st.session_state.history.keys())[0]
        else:
            create_new_session()

# ── Vector DB Cloud Helper ───────────────────────────────────────────────────
def ensure_vector_database():
    """Checks if ChromaDB exists, or downloads prebuilt database if URL provided in secrets."""
    global CHROMA_DIR
    if os.path.exists(CHROMA_DIR):
        return True, CHROMA_DIR
    
    # Check secrets for remote ChromaDB zip
    db_zip_url = ""
    try:
        db_zip_url = st.secrets.get("CHROMA_DB_ZIP_URL", "")
    except Exception:
        pass

    if db_zip_url:
        target_zip = os.path.join(BASE_DIR, "chromadb.zip")
        try:
            with st.spinner("📦 Downloading prebuilt Chroma Vector Store from cloud archive..."):
                urllib.request.urlretrieve(db_zip_url, target_zip)
                with zipfile.ZipFile(target_zip, 'r') as zip_ref:
                    zip_ref.extractall(BASE_DIR)
                if os.path.exists(target_zip):
                    os.remove(target_zip)
            if os.path.exists(os.path.join(BASE_DIR, "my_chroma_db")):
                CHROMA_DIR = os.path.join(BASE_DIR, "my_chroma_db")
                return True, CHROMA_DIR
            elif os.path.exists(os.path.join(BASE_DIR, "chromadb")):
                CHROMA_DIR = os.path.join(BASE_DIR, "chromadb")
                return True, CHROMA_DIR
        except Exception as e:
            return False, f"Failed to download prebuilt vector DB: {str(e)}"
    
    return False, "Database directory not found. Run `python ingest.py` locally or configure `CHROMA_DB_ZIP_URL` in secrets."

# ── Cached RAG Backend Components ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": DEVICE},
        encode_kwargs={"normalize_embeddings": True},
    )

def instantiate_llm(provider: str, model_name: str, temperature: float, api_key: str = "", ollama_base_url: str = ""):
    """Instantiates the appropriate LLM based on chosen provider (Ollama, Groq, or Gemini)."""
    if provider == "Ollama (Local Engine)":
        if ChatOllama is None:
            raise ImportError("langchain-ollama is not installed.")
        kwargs = {"model": model_name, "temperature": temperature}
        if ollama_base_url:
            kwargs["base_url"] = ollama_base_url
        return ChatOllama(**kwargs)
    
    elif provider == "Groq (Cloud - Fast)":
        if ChatGroq is None:
            raise ImportError("langchain-groq is not installed.")
        if not api_key:
            raise ValueError("Groq API key is missing. Please provide it in settings or Streamlit secrets.")
        return ChatGroq(model_name=model_name, groq_api_key=api_key, temperature=temperature)
    
    elif provider == "Google Gemini (Cloud)":
        if ChatGoogleGenerativeAI is None:
            raise ImportError("langchain-google-genai is not installed.")
        if not api_key:
            raise ValueError("Google Gemini API key is missing. Please provide it in settings or Streamlit secrets.")
        return ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key, temperature=temperature)
    
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")

def build_rag_chain(provider="Ollama (Local Engine)", model_name="qwen2.5-coder:latest", temperature=0.0, top_k=8, api_key="", ollama_url=""):
    db_ok, db_msg = ensure_vector_database()
    if not db_ok:
        return None, db_msg

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
        
        llm = instantiate_llm(
            provider=provider,
            model_name=model_name,
            temperature=temperature,
            api_key=api_key,
            ollama_base_url=ollama_url
        )
        
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

# ── Secrets Auto-Detection ────────────────────────────────────────────────────
groq_secret = ""
gemini_secret = ""
ollama_url_secret = ""
try:
    groq_secret = st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY", ""))
    gemini_secret = st.secrets.get("GOOGLE_API_KEY", st.secrets.get("GEMINI_API_KEY", os.environ.get("GEMINI_API_KEY", "")))
    ollama_url_secret = st.secrets.get("OLLAMA_BASE_URL", os.environ.get("OLLAMA_BASE_URL", ""))
except Exception:
    pass

# Determine sensible default provider
default_provider_idx = 0
if not os.path.exists(CHROMA_DIR) or groq_secret:
    if groq_secret:
        default_provider_idx = 1
    elif gemini_secret:
        default_provider_idx = 2

# ── Sidebar UI ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
        <div class="sidebar-header">
            <span>🩺 Emergency Medical Assistance</span>
        </div>
    """, unsafe_allow_html=True)
    
    if st.button("➕ New chat", key="new_chat_btn", use_container_width=True):
        create_new_session()
        st.rerun()

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
    with st.expander("⚙️ Inference & Engine Settings", expanded=True):
        provider = st.selectbox(
            "Inference Provider",
            ["Ollama (Local Engine)", "Groq (Cloud - Fast)", "Google Gemini (Cloud)"],
            index=default_provider_idx
        )
        
        active_api_key = ""
        ollama_url_input = ""

        if provider == "Ollama (Local Engine)":
            model_options = ["qwen2.5-coder:latest", "qwen2.5-coder:1.5b", "llama3", "llama3:latest", "mistral", "medllama2"]
            model_name = st.selectbox("Ollama Model", model_options, index=0)
            ollama_url_input = st.text_input("Ollama Host URL (Optional)", value=ollama_url_secret, placeholder="http://localhost:11434")
            
        elif provider == "Groq (Cloud - Fast)":
            model_options = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768", "gemma2-9b-it"]
            model_name = st.selectbox("Groq Model", model_options, index=0)
            active_api_key = st.text_input("Groq API Key", value=groq_secret, type="password", help="Get free key at console.groq.com")
            
        elif provider == "Google Gemini (Cloud)":
            model_options = ["gemini-1.5-flash", "gemini-2.0-flash", "gemini-1.5-pro"]
            model_name = st.selectbox("Gemini Model", model_options, index=0)
            active_api_key = st.text_input("Gemini API Key", value=gemini_secret, type="password", help="Get key from aistudio.google.com")

        top_k = st.slider("Retrieved Context Chunks (k)", min_value=2, max_value=15, value=8)
        temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.0, step=0.05)
        
        st.markdown("---")
        db_exists = os.path.exists(CHROMA_DIR)
        if db_exists:
            st.markdown('<span class="db-status db-online">● Vector DB: Active (80k+ chunks)</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="db-status db-building">⏳ Local DB Missing</span>', unsafe_allow_html=True)

# ── Main Content Area ─────────────────────────────────────────────────────────
current_messages = get_current_messages()

if len(current_messages) == 0:
    st.markdown("""
        <div class="hero-container">
            <div class="hero-title">Emergency Medical Assistance</div>
            <div class="hero-subtitle">Ask anything about emergency clinical protocols, first-aid, or medication guidelines.</div>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    preset_query = None
    with col1:
        if st.button("🚨 Sepsis Protocol\n\nBaseline criteria & resuscitation", use_container_width=True):
            preset_query = "What are the baseline clinical criteria used to identify severe sepsis in an adult patient, and what is the fluid resuscitation protocol?"
    with col2:
        if st.button("👶 Pediatric Emergency\n\nCroup vs Acute Epiglottitis", use_container_width=True):
            preset_query = "How do you clinically differentiate between pediatric croup and acute epiglottitis based on presentation?"
    with col3:
        if st.button("⚠️ Anaphylaxis Response\n\nFirst-line epinephrine dosing", use_container_width=True):
            preset_query = "What is the emergency management protocol and epinephrine dosage for severe acute anaphylaxis?"

    if preset_query:
        user_input = preset_query
    else:
        user_input = None
else:
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
    
    if len(session["messages"]) == 0:
        short_title = user_input.strip()
        if len(short_title) > 30:
            short_title = short_title[:28] + "..."
        session["title"] = short_title
    
    session["messages"].append({
        "role": "user",
        "content": user_input,
        "timestamp": datetime.datetime.now().isoformat()
    })
    session["updated_at"] = datetime.datetime.now().isoformat()
    save_history(st.session_state.history)
    
    with st.chat_message("user", avatar="🧑"):
        st.markdown(user_input)
    
    with st.chat_message("assistant", avatar="🩺"):
        with st.spinner(f"Searching verified medical database & generating answer with {model_name}..."):
            rag_chain, error_msg = build_rag_chain(
                provider=provider,
                model_name=model_name,
                temperature=temperature,
                top_k=top_k,
                api_key=active_api_key,
                ollama_url=ollama_url_input
            )
            
            if error_msg:
                response_text = f"⚠️ **Configuration Notice:** {error_msg}"
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
    
    session["messages"].append({
        "role": "assistant",
        "content": response_text,
        "sources": sources_list,
        "timestamp": datetime.datetime.now().isoformat()
    })
    session["updated_at"] = datetime.datetime.now().isoformat()
    save_history(st.session_state.history)
    
    st.rerun()
