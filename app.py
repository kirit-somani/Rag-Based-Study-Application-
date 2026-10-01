import os
from dotenv import load_dotenv
load_dotenv()

os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "critical"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import logging
logging.getLogger("transformers").setLevel(logging.CRITICAL)
import streamlit as st
from streamlit.runtime.scriptrunner import add_script_run_ctx
import sys
from pathlib import Path
import yaml
import time
import pandas as pd
import threading

# Add src to Python path so modules can be imported
sys.path.append(str(Path(__file__).resolve().parent))

from src.utils.logger import logger
from src.ingestion.ingestion_pipeline import process_pdf
from src.utils.file_utils import calculate_file_hash
from src.embeddings.embedding_manager import EmbeddingManager
from src.vectorstore.vector_store_manager import VectorStoreManager
from src.retrieval.retrieval_pipeline import retrieve_context
from src.generation.llm_manager import LLMManager
from src.rag.rag_pipeline import generate_rag_response
from src.generation.study_generator import generate_quiz, generate_flashcards, generate_summary
from src.utils.history_manager import load_history, save_history, add_history_entry, delete_history_entry, clear_all_history

GLOBAL_BACKGROUND_JOBS = {}

# Load basic config
def load_config():
    config_path = Path("config/settings.yaml")
    if config_path.exists():
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    return {}

config = load_config()
default_chunk_size = config.get("chunking", {}).get("chunk_size", 500)
default_chunk_overlap = config.get("chunking", {}).get("chunk_overlap", 50)
default_top_k = config.get("retrieval", {}).get("top_k", 5)
default_threshold = config.get("retrieval", {}).get("similarity_threshold", 0.20)

st.set_page_config(
    page_title="StudyRAG - Smart AI Study Partner",
    page_icon="📚",
    layout="wide"
)

# Custom Styling for Student-Centric Interface
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #4F46E5, #06B6D4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #6B7280;
        margin-bottom: 1.5rem;
    }
    .doc-card {
        background: #F9FAFB;
        border-radius: 10px;
        padding: 0.75rem 1rem;
        margin-bottom: 0.5rem;
        border: 1px solid #E5E7EB;
    }
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state 
if "processed_documents" not in st.session_state:
    st.session_state.processed_documents = {}
if "vector_store_manager" not in st.session_state:
    st.session_state.vector_store_manager = None
if "document_sources" not in st.session_state:
    st.session_state.document_sources = ["All Materials"]
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []
if "app_theme" not in st.session_state:
    st.session_state.app_theme = "🌙 Dark Mode"
if "background_jobs" not in st.session_state:
    st.session_state.background_jobs = {}

def apply_theme_css(theme_mode: str):
    is_dark = (theme_mode == "🌙 Dark Mode")
    
    bg_primary = "#0F172A" if is_dark else "#F8FAFC"
    bg_secondary = "#1E293B" if is_dark else "#FFFFFF"
    card_bg = "#1E293B" if is_dark else "#FFFFFF"
    card_border = "#334155" if is_dark else "#E2E8F0"
    text_primary = "#F8FAFC" if is_dark else "#0F172A"
    text_secondary = "#94A3B8" if is_dark else "#475569"
    input_bg = "#1E293B" if is_dark else "#FFFFFF"
    hover_bg = "#334155" if is_dark else "#F1F5F9"
    active_item_bg = "#334155" if is_dark else "#E2E8F0"
    grad_text = "linear-gradient(90deg, #818CF8, #38BDF8)" if is_dark else "linear-gradient(90deg, #4F46E5, #0891B2)"
    btn_bg = "linear-gradient(135deg, #6366F1 0%, #0284C7 100%)" if is_dark else "linear-gradient(135deg, #4F46E5 0%, #06B6D4 100%)"
    
    st.markdown(f"""
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Outfit:wght@500;600;700&display=swap" rel="stylesheet">
    
    <style>
        /* Base typography & colors */
        html, body, [class*="css"] {{
            font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
        }}
        
        .stApp {{
            background-color: {bg_primary} !important;
            color: {text_primary} !important;
        }}
        
        /* Fixed Header Bar styling */
        header[data-testid="stHeader"] {{
            background-color: {bg_primary} !important;
            color: {text_primary} !important;
        }}
        header[data-testid="stHeader"] * {{
            color: {text_primary} !important;
        }}
        
        /* Sidebar container styling */
        section[data-testid="stSidebar"] {{
            background-color: {bg_secondary} !important;
            border-right: 1px solid {card_border} !important;
        }}
        section[data-testid="stSidebar"] * {{
            color: {text_primary} !important;
        }}
        section[data-testid="stSidebar"] .stCaption,
        section[data-testid="stSidebar"] .stCaption * {{
            color: {text_secondary} !important;
        }}
        
        /* Header typography */
        .main-header {{
            font-family: 'Outfit', sans-serif !important;
            font-size: 2.2rem !important;
            font-weight: 700 !important;
            background: {grad_text} !important;
            -webkit-background-clip: text !important;
            -webkit-text-fill-color: transparent !important;
            margin-bottom: 0.3rem !important;
        }}
        
        .sub-header {{
            font-size: 1rem !important;
            color: {text_secondary} !important;
            margin-bottom: 1.5rem !important;
        }}
        
        .web-card {{
            background-color: {card_bg} !important;
            border: 1px solid {card_border} !important;
            border-radius: 12px !important;
            padding: 1.25rem 1.5rem !important;
            margin-bottom: 1rem !important;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05) !important;
        }}
        
        /* Labels, Paragraphs, Headings */
        .stMarkdown, p, span, h1, h2, h3, h4, h5, h6, label {{
            color: {text_primary} !important;
        }}
        
        .stCaption {{
            color: {text_secondary} !important;
        }}

        /* Sidebar Radio Navigation items */
        section[data-testid="stSidebar"] div[data-testid="stRadio"] label {{
            padding: 8px 12px !important;
            border-radius: 8px !important;
            margin-bottom: 4px !important;
            cursor: pointer !important;
            transition: all 0.15s ease !important;
            border: 1px solid transparent !important;
        }}

        section[data-testid="stSidebar"] div[data-testid="stRadio"] label:hover {{
            background-color: {hover_bg} !important;
        }}

        section[data-testid="stSidebar"] div[data-testid="stRadio"] label:has(input:checked) {{
            background-color: {active_item_bg} !important;
            border-color: {card_border} !important;
            font-weight: 600 !important;
        }}

        section[data-testid="stSidebar"] div[data-testid="stRadio"] label p,
        section[data-testid="stSidebar"] div[data-testid="stRadio"] label span,
        section[data-testid="stSidebar"] div[data-testid="stRadio"] label div {{
            color: {text_primary} !important;
            font-size: 0.95rem !important;
        }}

        /* Buttons */
        .stButton > button {{
            border-radius: 8px !important;
            font-weight: 600 !important;
            background: {btn_bg} !important;
            color: #FFFFFF !important;
            border: none !important;
            padding: 0.55rem 1.2rem !important;
            box-shadow: 0 4px 12px rgba(79, 70, 229, 0.25) !important;
            transition: all 0.2s ease !important;
        }}

        .stButton > button:hover {{
            transform: translateY(-1px) !important;
            box-shadow: 0 6px 16px rgba(79, 70, 229, 0.35) !important;
            color: #FFFFFF !important;
        }}
        .stButton > button * {{
            color: #FFFFFF !important;
        }}

        /* Selectboxes, Inputs & Text Areas */
        div[data-baseweb="select"] > div, input, textarea {{
            background-color: {input_bg} !important;
            color: {text_primary} !important;
            border: 1px solid {card_border} !important;
            border-radius: 8px !important;
        }}

        /* Chat messages */
        div[data-testid="stChatMessage"] {{
            background-color: {card_bg} !important;
            border: 1px solid {card_border} !important;
            border-radius: 12px !important;
            padding: 1rem !important;
            margin-bottom: 0.75rem !important;
        }}

        /* Expanders */
        .streamlit-expanderHeader {{
            background-color: {card_bg} !important;
            color: {text_primary} !important;
            border: 1px solid {card_border} !important;
            border-radius: 8px !important;
        }}

        .streamlit-expanderContent {{
            background-color: {card_bg} !important;
            border: 1px solid {card_border} !important;
            border-top: none !important;
            border-bottom-left-radius: 8px !important;
            border-bottom-right-radius: 8px !important;
        }}
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def get_embedding_manager(model_key: str):
    return EmbeddingManager(model_key)

@st.cache_resource
def get_llm_manager(provider: str, model_name: str, temperature: float):
    return LLMManager(provider=provider, model_name=model_name, temperature=temperature)

def update_document_sources():
    sources = set()
    for chunks in st.session_state.processed_documents.values():
        for chunk in chunks:
            if "source" in chunk:
                sources.add(chunk["source"])
    st.session_state.document_sources = ["All Materials"] + sorted(list(sources))

def ensure_vector_store_loaded(embedding_model_choice="minilm", vector_db_choice="faiss"):
    """Guarantees vector_store_manager is populated instantly without redundant disk scans."""
    if st.session_state.vector_store_manager is not None:
        return st.session_state.vector_store_manager

    upload_dir = Path("data/uploaded")
    if upload_dir.exists():
        for pdf_path in upload_dir.glob("*.pdf"):
            try:
                with open(pdf_path, "rb") as f:
                    file_bytes = f.read()
                file_hash = calculate_file_hash(file_bytes)
                if file_hash not in st.session_state.processed_documents:
                    chunks = process_pdf(pdf_path, chunk_size=500, chunk_overlap=50)
                    if chunks:
                        st.session_state.processed_documents[file_hash] = chunks
            except Exception as ex:
                logger.error(f"Error indexing {pdf_path.name}: {ex}")
        update_document_sources()

    try:
        emb_manager = get_embedding_manager(embedding_model_choice)
        vs_manager = VectorStoreManager(vector_db_choice, emb_manager.get_embeddings())
        if vs_manager.load():
            st.session_state.vector_store_manager = vs_manager
        elif st.session_state.processed_documents:
            all_chunks = []
            for chunks in st.session_state.processed_documents.values():
                all_chunks.extend(chunks)
            if all_chunks:
                docs = vs_manager.raw_chunks_to_documents(all_chunks)
                vs_manager.add_documents(docs)
                vs_manager.save()
                st.session_state.vector_store_manager = vs_manager
    except Exception as e:
        logger.error(f"Error loading vector store: {e}")

    return st.session_state.vector_store_manager

def delete_study_material(doc_name: str, embedding_model_choice="minilm", vector_db_choice="faiss"):
    """Deletes document file, clears its chunks, updates sources, and rebuilds vector store."""
    try:
        # 1. Delete physical file
        file_path = Path("data/uploaded") / doc_name
        if file_path.exists():
            file_path.unlink()
            
        # 2. Remove chunks matching doc_name
        hashes_to_remove = []
        for f_hash, chunks in st.session_state.processed_documents.items():
            if chunks and chunks[0].get("source") == doc_name:
                hashes_to_remove.append(f_hash)
        for f_hash in hashes_to_remove:
            del st.session_state.processed_documents[f_hash]
            
        update_document_sources()
        
        # 3. Rebuild or reset vector store
        emb_manager = get_embedding_manager(embedding_model_choice)
        vs_manager = VectorStoreManager(vector_db_choice, emb_manager.get_embeddings())
        vs_manager.reset()
        
        all_chunks = []
        for chunks in st.session_state.processed_documents.values():
            all_chunks.extend(chunks)
            
        if all_chunks:
            docs = vs_manager.raw_chunks_to_documents(all_chunks)
            vs_manager.add_documents(docs)
            vs_manager.save()
            st.session_state.vector_store_manager = vs_manager
        else:
            st.session_state.vector_store_manager = None
            
        add_history_entry("material", f"Deleted Study Material: {doc_name}", {"doc_name": doc_name, "action": "delete"})
        st.success(f"🗑️ Removed **{doc_name}** from your library.")
        time.sleep(0.5)
        st.rerun()
    except Exception as e:
        logger.error(f"Failed to delete {doc_name}: {e}")
        st.error(f"Could not delete {doc_name}: {e}")

# Background worker functions for non-blocking generation
def worker_generate_quiz(job_id, topic, vs_manager, llm_manager, num_questions, source_filter):
    try:
        quiz_data = generate_quiz(topic, vs_manager, llm_manager, num_questions, source_filter)
        if job_id in GLOBAL_BACKGROUND_JOBS:
            GLOBAL_BACKGROUND_JOBS[job_id]["status"] = "completed"
            GLOBAL_BACKGROUND_JOBS[job_id]["result"] = quiz_data
        add_history_entry(
            item_type="quiz",
            title=f"Practice Quiz: {topic}",
            details={"topic": topic, "questions": quiz_data, "num_questions": num_questions},
            source=source_filter
        )
    except Exception as e:
        logger.error(f"Background quiz error: {e}")
        if job_id in GLOBAL_BACKGROUND_JOBS:
            GLOBAL_BACKGROUND_JOBS[job_id]["status"] = "failed"
            GLOBAL_BACKGROUND_JOBS[job_id]["error"] = str(e)

def worker_generate_flashcards(job_id, topic, vs_manager, llm_manager, num_cards, source_filter):
    try:
        cards = generate_flashcards(topic, vs_manager, llm_manager, num_cards, source_filter)
        if job_id in GLOBAL_BACKGROUND_JOBS:
            GLOBAL_BACKGROUND_JOBS[job_id]["status"] = "completed"
            GLOBAL_BACKGROUND_JOBS[job_id]["result"] = cards
        add_history_entry(
            item_type="flashcard",
            title=f"Flashcard Deck: {topic}",
            details={"topic": topic, "cards": cards, "num_cards": num_cards},
            source=source_filter
        )
    except Exception as e:
        logger.error(f"Background flashcard error: {e}")
        if job_id in GLOBAL_BACKGROUND_JOBS:
            GLOBAL_BACKGROUND_JOBS[job_id]["status"] = "failed"
            GLOBAL_BACKGROUND_JOBS[job_id]["error"] = str(e)

def worker_generate_summary(job_id, topic, vs_manager, llm_manager, source_filter):
    try:
        summary_text = generate_summary(topic, vs_manager, llm_manager, source_filter)
        if job_id in GLOBAL_BACKGROUND_JOBS:
            GLOBAL_BACKGROUND_JOBS[job_id]["status"] = "completed"
            GLOBAL_BACKGROUND_JOBS[job_id]["result"] = summary_text
        add_history_entry(
            item_type="summary",
            title=f"Revision Summary: {topic}",
            details={"topic": topic, "summary_text": summary_text},
            source=source_filter
        )
    except Exception as e:
        logger.error(f"Background summary error: {e}")
        if job_id in GLOBAL_BACKGROUND_JOBS:
            GLOBAL_BACKGROUND_JOBS[job_id]["status"] = "failed"
            GLOBAL_BACKGROUND_JOBS[job_id]["error"] = str(e)


def main():
    logger.info("Application started.")
    
    # Sync global background jobs to session state
    for j_id, j_data in GLOBAL_BACKGROUND_JOBS.items():
        st.session_state.background_jobs[j_id] = j_data

    # Sidebar Navigation & Branding
    st.sidebar.markdown("<h2 style='margin-bottom: 0;'>📚 StudyRAG</h2>", unsafe_allow_html=True)
    st.sidebar.caption("Your Personal AI Study Companion")
    st.sidebar.markdown("---")
    
    # Modern Theme Toggle Switch
    dark_mode = st.sidebar.toggle("🌙 Dark Mode", value=(st.session_state.app_theme == "🌙 Dark Mode"))
    st.session_state.app_theme = "🌙 Dark Mode" if dark_mode else "☀️ Light Mode"
    apply_theme_css(st.session_state.app_theme)
    
    st.sidebar.markdown("---")
    
    page = st.sidebar.radio(
        "Navigation",
        ["💬 Ask Questions", "📝 Practice Quiz", "🗂 Flashcards", "📑 Topic Summaries", "📜 Study History", "📚 Study Library"],
        index=0
    )
    
    st.sidebar.markdown("---")
    
    # Background generation indicator
    running_jobs = [j for j in st.session_state.background_jobs.values() if j.get("status") == "running"]
    if running_jobs:
        st.sidebar.info(f"⏳ **{len(running_jobs)} task(s) generating in background...**")
        st.sidebar.caption("You can switch pages freely while your content builds!")

    # Pre-load vector store and check materials
    embedding_model = "minilm"
    vector_db_choice = "faiss"
    vs_mgr = ensure_vector_store_loaded(embedding_model, vector_db_choice)
    
    # Active Library Status for Student
    num_docs = max(0, len(st.session_state.document_sources) - 1)
    if num_docs > 0:
        st.sidebar.success(f"📖 **{num_docs} Study Material(s)** Active")
    else:
        st.sidebar.info("💡 Upload notes in **Study Library** to begin!")
        
    # Advanced Settings (collapsed by default)
    with st.sidebar.expander("⚙️ Advanced Settings", expanded=False):
        st.caption("Custom engine configuration for advanced users.")
        chunk_size = st.select_slider("Segment Size", options=[300, 500, 1000, 1500], value=default_chunk_size)
        chunk_overlap = 50
        
        default_provider = "groq" if os.getenv("GROQ_API_KEY") else "ollama"
        llm_provider = st.selectbox("AI Provider", options=["groq", "openai", "ollama"], index=0 if default_provider=="groq" else 2)
        
        if llm_provider == "groq":
            llm_model = st.selectbox("Model", options=["openai/gpt-oss-20b", "openai/gpt-oss-120b", "qwen/qwen3.8-27b"])
        elif llm_provider == "openai":
            llm_model = st.selectbox("Model", options=["gpt-3.5-turbo", "gpt-4o"])
        else:
            llm_model = st.selectbox("Model", options=["llama3", "mistral", "gemma"])
            
        llm_temperature = st.slider("Creativity (Temperature)", min_value=0.0, max_value=1.0, value=0.0, step=0.1)

    # Content Router
    if page == "📚 Study Library":
        render_documents_page(chunk_size, chunk_overlap, embedding_model, vector_db_choice)
    elif page == "💬 Ask Questions":
        render_chat_page(llm_provider, llm_model, llm_temperature, embedding_model, vector_db_choice)
    elif page == "📝 Practice Quiz":
        render_quiz_page(llm_provider, llm_model, llm_temperature, embedding_model, vector_db_choice)
    elif page == "🗂 Flashcards":
        render_flashcards_page(llm_provider, llm_model, llm_temperature, embedding_model, vector_db_choice)
    elif page == "📑 Topic Summaries":
        render_summary_page(llm_provider, llm_model, llm_temperature, embedding_model, vector_db_choice)
    elif page == "📜 Study History":
        render_history_page()

def render_documents_page(chunk_size, chunk_overlap, embedding_model, vector_db_choice):
    st.markdown("<div class='main-header'>📚 My Study Library</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Upload your textbooks, lecture notes, or syllabi (PDF) to build your instant AI study assistant.</div>", unsafe_allow_html=True)
    
    uploaded_files = st.file_uploader("Drag and drop your study materials (PDF format)", type=["pdf"], accept_multiple_files=True)
    if uploaded_files:
        for uploaded_file in uploaded_files:
            if uploaded_file.size == 0:
                st.warning(f"⚠️ {uploaded_file.name} is empty.")
                continue
            file_bytes = uploaded_file.read()
            file_hash = calculate_file_hash(file_bytes)
            if file_hash in st.session_state.processed_documents:
                st.success(f"✅ **{uploaded_file.name}** is loaded and ready.")
                continue
            upload_dir = Path("data/uploaded")
            upload_dir.mkdir(parents=True, exist_ok=True)
            save_path = upload_dir / uploaded_file.name
            with open(save_path, "wb") as f:
                f.write(file_bytes)
            status_placeholder = st.empty()
            status_placeholder.info(f"⏳ Indexing {uploaded_file.name} for instant study...")
            try:
                chunks = process_pdf(save_path, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                if not chunks:
                    status_placeholder.warning(f"⚠️ {uploaded_file.name} contains no readable text.")
                else:
                    st.session_state.processed_documents[file_hash] = chunks
                    status_placeholder.success(f"🎉 **{uploaded_file.name}** successfully indexed!")
                    # Rebuild vector store
                    vs_manager = VectorStoreManager(vector_db_choice, get_embedding_manager(embedding_model).get_embeddings())
                    all_chunks = []
                    for c_list in st.session_state.processed_documents.values():
                        all_chunks.extend(c_list)
                    docs = vs_manager.raw_chunks_to_documents(all_chunks)
                    vs_manager.add_documents(docs)
                    vs_manager.save()
                    st.session_state.vector_store_manager = vs_manager
                    update_document_sources()
                    add_history_entry("material", f"Uploaded Study Material: {uploaded_file.name}", {"doc_name": uploaded_file.name, "action": "upload"})
            except Exception as e:
                logger.error(f"Error processing {uploaded_file.name}: {e}")
                status_placeholder.error(f"❌ Unable to process {uploaded_file.name}.")
                
    st.markdown("---")
    st.markdown("### 📄 Currently Active Study Materials")
    sources = [s for s in st.session_state.document_sources if s != "All Materials"]
    if sources:
        for doc in sources:
            col_info, col_del = st.columns([5, 1])
            with col_info:
                st.markdown(f"📖 **{doc}**")
            with col_del:
                if st.button("🗑️ Delete", key=f"del_{doc}"):
                    delete_study_material(doc, embedding_model, vector_db_choice)
    else:
        st.info("No study materials uploaded yet. Upload a PDF above to get started!")

def render_chat_page(llm_provider, llm_model, llm_temperature, embedding_model, vector_db_choice):
    st.markdown("<div class='main-header'>💬 Ask Your Study Assistant</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Ask any question about your uploaded notes or textbooks. Get direct, factual answers with citations.</div>", unsafe_allow_html=True)
    
    vs_manager = ensure_vector_store_loaded(embedding_model, vector_db_choice)
    if not vs_manager:
        st.info("💡 **Getting Started:** Upload your study notes or textbooks in **📚 Study Library** to start asking questions!")
        return

    # User Scope Selection
    col1, col2 = st.columns([3, 1])
    with col1:
        source_filter_ui = st.selectbox("Focus Search On:", options=st.session_state.document_sources)
        source_filter = "All Documents" if source_filter_ui == "All Materials" else source_filter_ui
    with col2:
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_messages = []
            st.rerun()

    # Display Chat History
    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and "sources" in message and message["sources"]:
                with st.expander("📖 View Referenced Sources"):
                    for i, source in enumerate(message["sources"], 1):
                        st.markdown(f"**Reference {i}: {source['source']} (Page {source['page']})**")
                        st.markdown(f"> {source['text']}")

    # User Question Input
    if query := st.chat_input("Ask a question about your study material..."):
        st.session_state.chat_messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)
            
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("🤔 Reading your study materials...")
            
            try:
                llm_manager = get_llm_manager(llm_provider, llm_model, llm_temperature)
                response = generate_rag_response(
                    query=query,
                    vector_store=vs_manager,
                    llm_manager=llm_manager,
                    top_k=default_top_k,
                    score_threshold=default_threshold,
                    source_filter=source_filter
                )
                
                answer = response["answer"]
                sources = response.get("sources", [])
                
                message_placeholder.markdown(answer)
                
                if sources:
                    with st.expander("📖 View Referenced Sources"):
                        for i, source in enumerate(sources, 1):
                            st.markdown(f"**Reference {i}: {source['source']} (Page {source['page']})**")
                            st.markdown(f"> {source['text']}")
                            
                st.session_state.chat_messages.append({
                    "role": "assistant", 
                    "content": answer,
                    "sources": sources
                })
                
                # Add to persistent history
                add_history_entry(
                    item_type="chat",
                    title=f"Chat Question: {query[:50]}...",
                    details={"query": query, "answer": answer, "sources": sources},
                    source=source_filter
                )
                
            except Exception as e:
                error_msg = f"❌ Error retrieving answer: {e}"
                message_placeholder.error(error_msg)
                logger.error(f"Chat error: {e}")

def render_quiz_page(llm_provider, llm_model, llm_temperature, embedding_model, vector_db_choice):
    st.markdown("<div class='main-header'>📝 Practice Quiz Generator</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Test your understanding! Generate customized multiple-choice practice tests directly from your notes.</div>", unsafe_allow_html=True)
    
    vs_manager = ensure_vector_store_loaded(embedding_model, vector_db_choice)
    if not vs_manager:
        st.info("💡 **Getting Started:** Upload your study materials in **📚 Study Library** to generate practice quizzes!")
        return

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        topic = st.text_input("Quiz Topic / Chapter Name:", placeholder="e.g. SQL Queries, Database Normalization")
    with col2:
        num_questions = st.number_input("Questions", min_value=1, max_value=10, value=5)
    with col3:
        source_filter_ui = st.selectbox("Study Source:", options=st.session_state.document_sources, key="quiz_source")
        source_filter = "All Documents" if source_filter_ui == "All Materials" else source_filter_ui

    if st.button("🚀 Generate Practice Quiz", use_container_width=True):
        if not topic:
            st.error("Please enter a topic to quiz yourself on.")
            return

        job_id = f"quiz_{int(time.time())}"
        job_data = {
            "id": job_id,
            "type": "quiz",
            "title": topic,
            "status": "running",
            "result": None,
            "error": None
        }
        GLOBAL_BACKGROUND_JOBS[job_id] = job_data
        st.session_state.background_jobs[job_id] = job_data

        llm_manager = get_llm_manager(llm_provider, llm_model, llm_temperature)
        t = threading.Thread(
            target=worker_generate_quiz,
            args=(job_id, topic, vs_manager, llm_manager, num_questions, source_filter)
        )
        add_script_run_ctx(t)
        t.start()
        st.success(f"🚀 Started quiz generation for **'{topic}'** in the background! You can navigate to other pages anytime.")

    # Check for active or recent quiz jobs
    quiz_jobs = [j for j in st.session_state.background_jobs.values() if j.get("type") == "quiz"]
    if quiz_jobs:
        latest_job = quiz_jobs[-1]
        if latest_job["status"] == "running":
            st.info(f"⏳ **Quiz for '{latest_job['title']}' is currently generating in the background...** Feel free to use other pages!")
            if st.button("🔄 Refresh Page"):
                st.rerun()
        elif latest_job["status"] == "completed" and latest_job["result"]:
            st.session_state.current_quiz = latest_job["result"]
            st.session_state.quiz_topic = latest_job["title"]
        elif latest_job["status"] == "failed":
            st.error(f"❌ Quiz generation failed: {latest_job.get('error')}")

    if "current_quiz" in st.session_state and st.session_state.current_quiz:
        st.markdown(f"### 📋 Practice Test: {st.session_state.quiz_topic}")
        
        with st.form("quiz_form"):
            user_answers = {}
            for i, q in enumerate(st.session_state.current_quiz):
                st.markdown(f"**Question {i+1}: {q.get('question', '')}**")
                options = q.get('options', [])
                if options:
                    user_answers[i] = st.radio("Choose the best answer:", options=options, key=f"q_{i}")
                st.markdown("---")
                
            submitted = st.form_submit_button("Grade My Quiz 🎯")
            if submitted:
                st.session_state.quiz_submitted = True
                st.session_state.user_answers = user_answers
                
        if st.session_state.get("quiz_submitted", False):
            st.markdown("### 🏆 Quiz Results")
            score = 0
            for i, q in enumerate(st.session_state.current_quiz):
                correct_ans = q.get('answer', '')
                user_ans = st.session_state.user_answers.get(i, '')
                
                if user_ans and user_ans[0] == correct_ans[0]:
                    st.success(f"**Q{i+1}: Correct!** {correct_ans}")
                    score += 1
                else:
                    st.error(f"**Q{i+1}: Incorrect.** You selected: {user_ans}. Correct Answer: {correct_ans}")
                    
            st.balloons() if score == len(st.session_state.current_quiz) else None
            st.info(f"🎉 **Your Score: {score} / {len(st.session_state.current_quiz)}**")

def render_flashcards_page(llm_provider, llm_model, llm_temperature, embedding_model, vector_db_choice):
    st.markdown("<div class='main-header'>🗂 Flashcard Creator</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Memorize key terms, formulas, and concepts with automated flashcards.</div>", unsafe_allow_html=True)
    
    vs_manager = ensure_vector_store_loaded(embedding_model, vector_db_choice)
    if not vs_manager:
        st.info("💡 **Getting Started:** Upload your study materials in **📚 Study Library** to create flashcards!")
        return

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        topic = st.text_input("Topic or Key Concepts:", placeholder="e.g. Operating System Kernels, Data Structures")
    with col2:
        num_cards = st.number_input("Number of Cards", min_value=1, max_value=15, value=5)
    with col3:
        source_filter_ui = st.selectbox("Study Source:", options=st.session_state.document_sources, key="flashcard_source")
        source_filter = "All Documents" if source_filter_ui == "All Materials" else source_filter_ui

    if st.button("🗂 Create Flashcards", use_container_width=True):
        if not topic:
            st.error("Please enter a topic for flashcards.")
            return

        job_id = f"fc_{int(time.time())}"
        job_data = {
            "id": job_id,
            "type": "flashcard",
            "title": topic,
            "status": "running",
            "result": None,
            "error": None
        }
        GLOBAL_BACKGROUND_JOBS[job_id] = job_data
        st.session_state.background_jobs[job_id] = job_data

        llm_manager = get_llm_manager(llm_provider, llm_model, llm_temperature)
        t = threading.Thread(
            target=worker_generate_flashcards,
            args=(job_id, topic, vs_manager, llm_manager, num_cards, source_filter)
        )
        add_script_run_ctx(t)
        t.start()
        st.success(f"🗂 Started generating flashcards for **'{topic}'** in the background! You can navigate to other pages anytime.")

    fc_jobs = [j for j in st.session_state.background_jobs.values() if j.get("type") == "flashcard"]
    if fc_jobs:
        latest_job = fc_jobs[-1]
        if latest_job["status"] == "running":
            st.info(f"⏳ **Flashcards for '{latest_job['title']}' are currently generating in the background...**")
            if st.button("🔄 Refresh Page", key="ref_fc"):
                st.rerun()
        elif latest_job["status"] == "completed" and latest_job["result"]:
            st.session_state.current_flashcards = latest_job["result"]
            st.session_state.flashcard_topic = latest_job["title"]
        elif latest_job["status"] == "failed":
            st.error(f"❌ Flashcard creation failed: {latest_job.get('error')}")

    if "current_flashcards" in st.session_state and st.session_state.current_flashcards:
        st.markdown(f"### 🗂 Study Deck: {st.session_state.flashcard_topic}")
        for i, card in enumerate(st.session_state.current_flashcards):
            with st.expander(f"📌 Card {i+1}: {card.get('front', '')}", expanded=True):
                st.markdown(f"**Answer / Definition:**\n\n{card.get('back', '')}")

def render_summary_page(llm_provider, llm_model, llm_temperature, embedding_model, vector_db_choice):
    st.markdown("<div class='main-header'>📑 Instant Topic Summarizer</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Generate structured chapter summaries, key takeaways, and revision guides.</div>", unsafe_allow_html=True)
    
    vs_manager = ensure_vector_store_loaded(embedding_model, vector_db_choice)
    if not vs_manager:
        st.info("💡 **Getting Started:** Upload your study materials in **📚 Study Library** to generate summaries!")
        return

    col1, col2 = st.columns([3, 1])
    with col1:
        topic = st.text_input("Enter Topic / Chapter to Summarize:", placeholder="e.g. Relational Algebra, Database Constraints")
    with col2:
        source_filter_ui = st.selectbox("Study Source:", options=st.session_state.document_sources, key="summary_source")
        source_filter = "All Documents" if source_filter_ui == "All Materials" else source_filter_ui

    if st.button("📑 Generate Revision Summary", use_container_width=True):
        if not topic:
            st.error("Please enter a topic.")
            return

        job_id = f"sum_{int(time.time())}"
        job_data = {
            "id": job_id,
            "type": "summary",
            "title": topic,
            "status": "running",
            "result": None,
            "error": None
        }
        GLOBAL_BACKGROUND_JOBS[job_id] = job_data
        st.session_state.background_jobs[job_id] = job_data

        llm_manager = get_llm_manager(llm_provider, llm_model, llm_temperature)
        t = threading.Thread(
            target=worker_generate_summary,
            args=(job_id, topic, vs_manager, llm_manager, source_filter)
        )
        add_script_run_ctx(t)
        t.start()
        st.success(f"📑 Started summary generation for **'{topic}'** in the background! You can navigate to other pages anytime.")

    sum_jobs = [j for j in st.session_state.background_jobs.values() if j.get("type") == "summary"]
    if sum_jobs:
        latest_job = sum_jobs[-1]
        if latest_job["status"] == "running":
            st.info(f"⏳ **Summary for '{latest_job['title']}' is currently generating in the background...**")
            if st.button("🔄 Refresh Page", key="ref_sum"):
                st.rerun()
        elif latest_job["status"] == "completed" and latest_job["result"]:
            st.session_state.current_summary = latest_job["result"]
            st.session_state.summary_topic = latest_job["title"]
        elif latest_job["status"] == "failed":
            st.error(f"❌ Summary generation failed: {latest_job.get('error')}")

    if "current_summary" in st.session_state and st.session_state.current_summary:
        st.markdown(f"### 📖 Summary Guide: {st.session_state.summary_topic}")
        st.markdown(st.session_state.current_summary)

def render_history_page():
    st.markdown("<div class='main-header'>📜 Study History</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>View your past generated quizzes, flashcards, summaries, and chat sessions. Delete any item at any time.</div>", unsafe_allow_html=True)

    history_items = load_history()
    
    if not history_items:
        st.info("📜 No history entries yet. Start generating quizzes, flashcards, or asking questions to build your study history!")
        return

    col_filter, col_clear = st.columns([3, 1])
    with col_filter:
        filter_type = st.selectbox(
            "Filter History By:",
            ["All Activity", "📝 Practice Quizzes", "🗂 Flashcard Decks", "📑 Revision Summaries", "💬 Chat Questions", "📚 Library Actions"]
        )
    with col_clear:
        if st.button("🗑️ Clear All History", use_container_width=True):
            clear_all_history()
            st.success("All history cleared!")
            time.sleep(0.5)
            st.rerun()

    type_map = {
        "📝 Practice Quizzes": "quiz",
        "🗂 Flashcard Decks": "flashcard",
        "📑 Revision Summaries": "summary",
        "💬 Chat Questions": "chat",
        "📚 Library Actions": "material"
    }

    filtered_items = history_items
    if filter_type in type_map:
        target_type = type_map[filter_type]
        filtered_items = [h for h in history_items if h.get("type") == target_type]

    st.markdown(f"### 📋 Showing {len(filtered_items)} Activity Log(s)")

    for item in filtered_items:
        item_id = item.get("id")
        i_type = item.get("type", "activity")
        i_title = item.get("title", "Untitled")
        i_time = item.get("timestamp", "")
        i_source = item.get("source", "All Documents")
        i_details = item.get("details", {})

        icon_map = {
            "quiz": "📝",
            "flashcard": "🗂",
            "summary": "📑",
            "chat": "💬",
            "material": "📚"
        }
        icon = icon_map.get(i_type, "📌")

        with st.expander(f"{icon} **{i_title}** — *{i_time}*", expanded=False):
            col_info, col_del = st.columns([5, 1])
            with col_del:
                if st.button("🗑️ Delete Entry", key=f"del_hist_{item_id}"):
                    delete_history_entry(item_id)
                    st.success("Deleted from history.")
                    time.sleep(0.3)
                    st.rerun()

            with col_info:
                st.caption(f"**Source Material Filter:** {i_source}")
                
                if i_type == "quiz":
                    questions = i_details.get("questions", [])
                    st.markdown(f"**Total Questions:** {len(questions)}")
                    for q_idx, q in enumerate(questions, 1):
                        st.markdown(f"**Q{q_idx}: {q.get('question', '')}**")
                        for opt in q.get("options", []):
                            st.markdown(f"- {opt}")
                        st.markdown(f"✅ **Answer:** {q.get('answer', '')}")
                        st.markdown("---")

                elif i_type == "flashcard":
                    cards = i_details.get("cards", [])
                    st.markdown(f"**Total Cards:** {len(cards)}")
                    for c_idx, card in enumerate(cards, 1):
                        st.markdown(f"**Card {c_idx}:** {card.get('front', '')}")
                        st.markdown(f"> **Definition:** {card.get('back', '')}")

                elif i_type == "summary":
                    summary_text = i_details.get("summary_text", "")
                    st.markdown(summary_text)

                elif i_type == "chat":
                    st.markdown(f"**User Question:** {i_details.get('query', '')}")
                    st.markdown(f"**AI Response:** {i_details.get('answer', '')}")

                elif i_type == "material":
                    st.markdown(f"**Action:** {i_details.get('action', 'updated').title()}")
                    st.markdown(f"**File:** {i_details.get('doc_name', '')}")

if __name__ == "__main__":
    main()

