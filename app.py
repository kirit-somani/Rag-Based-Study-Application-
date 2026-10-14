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

from src.ui.text_provider import get_ui_text
from src.ui.styles import inject_app_styles
from src.ui.components import (
    render_hero_banner,
    render_brand_header,
    render_system_pill,
    render_stat_grid,
    render_doc_item,
    render_empty_state,
    render_prompt_chips,
    render_citation_box,
    render_quiz_card,
    render_score_card,
    render_flashcard_box,
    render_footer,
    render_spacer,
    render_glass_card_open,
    render_glass_card_close,
    render_welcome_hero,
    render_feature_card_html,
    render_steps_grid,
    render_section_title
)

GLOBAL_BACKGROUND_JOBS = {}

def load_config():
    config_path = Path("config/settings.yaml")
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}

config = load_config()
default_chunk_size = config.get("chunking", {}).get("chunk_size", 500)
default_chunk_overlap = config.get("chunking", {}).get("chunk_overlap", 50)
default_top_k = config.get("retrieval", {}).get("top_k", 5)
default_threshold = config.get("retrieval", {}).get("similarity_threshold", 0.20)

st.set_page_config(
    page_title=get_ui_text("app", "page_title"),
    page_icon=get_ui_text("app", "page_icon"),
    layout="wide",
    initial_sidebar_state="expanded"
)

if "processed_documents" not in st.session_state:
    st.session_state.processed_documents = {}
if "vector_store_manager" not in st.session_state:
    st.session_state.vector_store_manager = None
if "document_sources" not in st.session_state:
    st.session_state.document_sources = [get_ui_text("chat", "all_documents_choice")]
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []
if "app_theme" not in st.session_state:
    st.session_state.app_theme = "Dark"
if "background_jobs" not in st.session_state:
    st.session_state.background_jobs = {}
if "embedding_model" not in st.session_state:
    st.session_state.embedding_model = "minilm"
if "vector_db_choice" not in st.session_state:
    st.session_state.vector_db_choice = "faiss"
if "current_view" not in st.session_state:
    st.session_state.current_view = "welcome"
if "selected_tool" not in st.session_state:
    st.session_state.selected_tool = get_ui_text("sidebar", "nav_items", "chat")

@st.cache_resource
def get_embedding_manager(model_key: str):
    return EmbeddingManager(model_key)

@st.cache_resource
def get_llm_manager(provider: str, model_name: str, temperature: float):
    return LLMManager(provider=provider, model_name=model_name, temperature=temperature)

def update_document_sources():
    sources = set()
    all_label = get_ui_text("chat", "all_documents_choice")
    for chunks in st.session_state.processed_documents.values():
        for chunk in chunks:
            if "source" in chunk:
                sources.add(chunk["source"])
    st.session_state.document_sources = [all_label] + sorted(list(sources))

def ensure_vector_store_loaded(embedding_model_choice="minilm", vector_db_choice="faiss"):
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
    try:
        file_path = Path("data/uploaded") / doc_name
        if file_path.exists():
            file_path.unlink()

        hashes_to_remove = []
        for f_hash, chunks in st.session_state.processed_documents.items():
            if chunks and chunks[0].get("source") == doc_name:
                hashes_to_remove.append(f_hash)
        for f_hash in hashes_to_remove:
            del st.session_state.processed_documents[f_hash]

        update_document_sources()

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
        st.success(get_ui_text("library", "removed_success").format(doc_name=doc_name))
        time.sleep(0.5)
        st.rerun()
    except Exception as e:
        logger.error(f"Failed to delete {doc_name}: {e}")
        st.error(get_ui_text("library", "remove_error").format(doc_name=doc_name, error=e))

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

    for j_id, j_data in GLOBAL_BACKGROUND_JOBS.items():
        st.session_state.background_jobs[j_id] = j_data

    render_brand_header(
        brand_name=get_ui_text("app", "brand_name"),
        subtitle=get_ui_text("app", "brand_subtitle"),
        logo_glyph=get_ui_text("app", "page_icon")
    )

    is_dark = st.sidebar.toggle(
        get_ui_text("theme", "toggle_label"),
        value=(st.session_state.app_theme == "Dark")
    )
    st.session_state.app_theme = "Dark" if is_dark else "Light"

    inject_app_styles(theme_mode=st.session_state.app_theme)

    st.sidebar.markdown("---")

    nav_dict = get_ui_text("sidebar", "nav_items")
    tool_options = [
        nav_dict.get("chat"),
        nav_dict.get("quiz"),
        nav_dict.get("flashcards"),
        nav_dict.get("summaries"),
        nav_dict.get("history"),
        nav_dict.get("library")
    ]

    if st.session_state.current_view == "welcome":
        st.sidebar.markdown(f"### {get_ui_text('sidebar', 'nav_label')}")
        if st.sidebar.button(get_ui_text("sidebar", "enter_workspace_btn"), use_container_width=True, key="sidebar_enter_ws"):
            st.session_state.current_view = "workspace"
            st.session_state.selected_tool = nav_dict.get("chat")
            st.rerun()

        st.sidebar.caption("Click above or any module on the right to enter your workspace.")
    else:

        if st.sidebar.button(get_ui_text("sidebar", "return_to_welcome"), use_container_width=True, key="return_welcome_btn"):
            st.session_state.current_view = "welcome"
            st.rerun()

        st.sidebar.markdown("---")

        current_idx = 0
        if st.session_state.selected_tool in tool_options:
            current_idx = tool_options.index(st.session_state.selected_tool)

        page = st.sidebar.radio(
            get_ui_text("sidebar", "nav_label"),
            tool_options,
            index=current_idx,
            key="workspace_tools_radio"
        )
        st.session_state.selected_tool = page

    running_jobs = [j for j in st.session_state.background_jobs.values() if j.get("status") == "running"]
    if running_jobs:
        render_system_pill(
            title=get_ui_text("sidebar", "system_status", "background_generating"),
            value=get_ui_text("sidebar", "system_status", "active_count").format(count=len(running_jobs)),
            is_active=True,
            dot_color="dot-purple"
        )

    num_docs = max(0, len(st.session_state.document_sources) - 1)
    status_text = (
        get_ui_text("sidebar", "system_status", "docs_indexed").format(count=num_docs)
        if num_docs > 0
        else get_ui_text("sidebar", "system_status", "no_docs")
    )
    dot_class = "dot-green" if num_docs > 0 else "dot-purple"
    render_system_pill(
        title=get_ui_text("sidebar", "system_status", "library_status"),
        value=status_text,
        is_active=False,
        dot_color=dot_class
    )

    with st.sidebar.expander(get_ui_text("sidebar", "settings_drawer", "title"), expanded=False):
        st.caption(get_ui_text("sidebar", "settings_drawer", "caption_engine"))
        chunk_size = st.select_slider(
            get_ui_text("sidebar", "settings_drawer", "chunk_slider_label"),
            options=[300, 500, 1000, 1500],
            value=default_chunk_size
        )
        chunk_overlap = 50

        default_provider = "groq" if os.getenv("GROQ_API_KEY") else "ollama"
        llm_provider = st.selectbox(
            get_ui_text("sidebar", "settings_drawer", "provider_label"),
            options=["groq", "openai", "ollama"],
            index=0 if default_provider=="groq" else 2
        )

        if llm_provider == "groq":
            llm_model = st.selectbox(get_ui_text("sidebar", "settings_drawer", "model_label"), options=["openai/gpt-oss-20b", "openai/gpt-oss-120b", "qwen/qwen3.8-27b"])
        elif llm_provider == "openai":
            llm_model = st.selectbox(get_ui_text("sidebar", "settings_drawer", "model_label"), options=["gpt-3.5-turbo", "gpt-4o"])
        else:
            llm_model = st.selectbox(get_ui_text("sidebar", "settings_drawer", "model_label"), options=["llama3", "mistral", "gemma"])

        llm_temperature = st.slider(get_ui_text("sidebar", "settings_drawer", "temp_slider_label"), min_value=0.0, max_value=1.0, value=0.0, step=0.1)

        st.markdown("---")
        st.caption(get_ui_text("sidebar", "settings_drawer", "caption_retrieval"))

        emb_opts = get_ui_text("sidebar", "settings_drawer", "embedding_options")
        embedding_model = st.selectbox(
            get_ui_text("sidebar", "settings_drawer", "embedding_label"),
            options=["minilm", "bge"],
            format_func=lambda x: emb_opts.get(x, x),
        )

        vdb_opts = get_ui_text("sidebar", "settings_drawer", "vector_db_options")
        vector_db_choice = st.selectbox(
            get_ui_text("sidebar", "settings_drawer", "vector_db_label"),
            options=["faiss", "chroma"],
            format_func=lambda x: vdb_opts.get(x, x),
        )

    render_footer(get_ui_text("app", "version_tag"))

    vs_mgr = ensure_vector_store_loaded(embedding_model, vector_db_choice)

    if st.session_state.current_view == "welcome":
        render_welcome_page(nav_dict)
    else:
        active_tool = st.session_state.selected_tool
        if active_tool == nav_dict.get("library"):
            render_documents_page(chunk_size, chunk_overlap, embedding_model, vector_db_choice)
        elif active_tool == nav_dict.get("chat"):
            render_chat_page(llm_provider, llm_model, llm_temperature, embedding_model, vector_db_choice)
        elif active_tool == nav_dict.get("quiz"):
            render_quiz_page(llm_provider, llm_model, llm_temperature, embedding_model, vector_db_choice)
        elif active_tool == nav_dict.get("flashcards"):
            render_flashcards_page(llm_provider, llm_model, llm_temperature, embedding_model, vector_db_choice)
        elif active_tool == nav_dict.get("summaries"):
            render_summary_page(llm_provider, llm_model, llm_temperature, embedding_model, vector_db_choice)
        elif active_tool == nav_dict.get("history"):
            render_history_page()

def render_welcome_page(nav_dict):
    render_welcome_hero()

    col_btn1, col_btn2, col_spacer = st.columns([1.5, 1.5, 3])
    with col_btn1:
        if st.button(get_ui_text("welcome", "get_started_btn"), use_container_width=True, key="welcome_btn_library"):
            st.session_state.current_view = "workspace"
            st.session_state.selected_tool = nav_dict.get("library")
            st.rerun()
    with col_btn2:
        if st.button(get_ui_text("welcome", "quick_chat_btn"), use_container_width=True, key="welcome_btn_chat"):
            st.session_state.current_view = "workspace"
            st.session_state.selected_tool = nav_dict.get("chat")
            st.rerun()

    render_section_title(get_ui_text("welcome", "features_title"))

    features = get_ui_text("welcome", "features")
    cols = st.columns(4)
    for idx, f in enumerate(features):
        with cols[idx]:
            render_feature_card_html(icon=f.get("icon", "💡"), title=f.get("title", ""), desc=f.get("desc", ""))
            btn_key = f"welcome_card_btn_{f.get('key', idx)}"
            if st.button(f.get("btn_label", "Open →"), key=btn_key, use_container_width=True):
                target_key = f.get("key", "chat")
                st.session_state.current_view = "workspace"
                st.session_state.selected_tool = nav_dict.get(target_key, nav_dict.get("chat"))
                st.rerun()

    render_section_title(get_ui_text("welcome", "steps_title"))
    render_steps_grid(get_ui_text("welcome", "steps"))

def render_documents_page(chunk_size, chunk_overlap, embedding_model, vector_db_choice):
    render_hero_banner(
        tag=get_ui_text("library", "hero_tag"),
        title=get_ui_text("library", "hero_title"),
        subtitle=get_ui_text("library", "hero_subtitle")
    )

    total_chunks = sum(len(c) for c in st.session_state.processed_documents.values())
    num_docs = max(0, len(st.session_state.document_sources) - 1)

    stats = [
        {"value": str(num_docs), "label": get_ui_text("library", "stats", "active_docs")},
        {"value": str(total_chunks), "label": get_ui_text("library", "stats", "indexed_passages")},
        {"value": str(chunk_size), "label": get_ui_text("library", "stats", "token_size")},
        {"value": embedding_model.upper(), "label": get_ui_text("library", "stats", "dense_embeddings")},
    ]
    render_stat_grid(stats)

    uploaded_files = st.file_uploader(
        get_ui_text("library", "uploader_label"),
        type=["pdf"],
        accept_multiple_files=True,
        help=get_ui_text("library", "uploader_help")
    )
    if uploaded_files:
        for uploaded_file in uploaded_files:
            if uploaded_file.size == 0:
                st.warning(get_ui_text("library", "status_empty").format(filename=uploaded_file.name))
                continue
            file_bytes = uploaded_file.read()
            file_hash = calculate_file_hash(file_bytes)
            if file_hash in st.session_state.processed_documents:
                st.success(get_ui_text("library", "status_already_indexed").format(filename=uploaded_file.name))
                continue
            upload_dir = Path("data/uploaded")
            upload_dir.mkdir(parents=True, exist_ok=True)
            save_path = upload_dir / uploaded_file.name
            with open(save_path, "wb") as f:
                f.write(file_bytes)
            status_placeholder = st.empty()
            status_placeholder.info(get_ui_text("library", "status_indexing").format(filename=uploaded_file.name))
            try:
                chunks = process_pdf(save_path, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                if not chunks:
                    status_placeholder.warning(get_ui_text("library", "status_no_text").format(filename=uploaded_file.name))
                else:
                    st.session_state.processed_documents[file_hash] = chunks
                    status_placeholder.success(get_ui_text("library", "status_indexed_success").format(filename=uploaded_file.name))
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
                status_placeholder.error(get_ui_text("library", "status_index_error").format(filename=uploaded_file.name, error=e))

    st.markdown("---")
    st.markdown(f"### {get_ui_text('library', 'active_materials_heading')}")
    all_label = get_ui_text("chat", "all_documents_choice")
    sources = [s for s in st.session_state.document_sources if s != all_label]
    if sources:
        for doc in sources:
            render_doc_item(doc_name=doc, description=get_ui_text("library", "doc_card_desc"))
            col_spacer, col_del = st.columns([5, 1])
            with col_del:
                if st.button(get_ui_text("library", "remove_button"), key=f"del_{doc}"):
                    delete_study_material(doc, embedding_model, vector_db_choice)
    else:
        render_empty_state(
            title=get_ui_text("library", "empty_state", "title"),
            subtitle=get_ui_text("library", "empty_state", "subtitle")
        )

def render_chat_page(llm_provider, llm_model, llm_temperature, embedding_model, vector_db_choice):
    render_hero_banner(
        tag=get_ui_text("chat", "hero_tag"),
        title=get_ui_text("chat", "hero_title"),
        subtitle=get_ui_text("chat", "hero_subtitle")
    )

    vs_manager = ensure_vector_store_loaded(embedding_model, vector_db_choice)
    if not vs_manager:
        render_empty_state(
            title=get_ui_text("chat", "empty_state", "title"),
            subtitle=get_ui_text("chat", "empty_state", "subtitle"),
            icon="💡"
        )
        return

    col1, col2 = st.columns([3, 1])
    all_label = get_ui_text("chat", "all_documents_choice")
    with col1:
        source_filter_ui = st.selectbox(get_ui_text("chat", "focus_label"), options=st.session_state.document_sources)
        source_filter = "All Documents" if source_filter_ui == all_label else source_filter_ui
    with col2:
        render_spacer("spacer-28")
        if st.button(get_ui_text("chat", "clear_button"), use_container_width=True, key="chat_clear_dialogue_btn"):
            st.session_state.chat_messages = []
            st.rerun()

    if not st.session_state.chat_messages:
        render_prompt_chips(
            title=get_ui_text("chat", "suggested_prompts_title"),
            prompts=get_ui_text("chat", "prompts")
        )

    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and "sources" in message and message["sources"]:
                citations_title = get_ui_text("chat", "citations_expander").format(count=len(message["sources"]))
                with st.expander(citations_title):
                    for i, source in enumerate(message["sources"], 1):
                        render_citation_box(
                            index=i,
                            source=source["source"],
                            page=source["page"],
                            quote=source["text"]
                        )

    if query := st.chat_input(get_ui_text("chat", "input_placeholder")):
        st.session_state.chat_messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown(get_ui_text("chat", "thinking_message"))

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
                    citations_title = get_ui_text("chat", "citations_expander").format(count=len(sources))
                    with st.expander(citations_title):
                        for i, source in enumerate(sources, 1):
                            render_citation_box(
                                index=i,
                                source=source["source"],
                                page=source["page"],
                                quote=source["text"]
                            )

                st.session_state.chat_messages.append({
                    "role": "assistant", 
                    "content": answer,
                    "sources": sources
                })

                add_history_entry(
                    item_type="chat",
                    title=f"Chat Question: {query[:50]}...",
                    details={"query": query, "answer": answer, "sources": sources},
                    source=source_filter
                )

            except Exception as e:
                error_msg = get_ui_text("chat", "retrieval_error").format(error=e)
                message_placeholder.error(error_msg)
                logger.error(f"Chat error: {e}")

def render_quiz_page(llm_provider, llm_model, llm_temperature, embedding_model, vector_db_choice):
    render_hero_banner(
        tag=get_ui_text("quiz", "hero_tag"),
        title=get_ui_text("quiz", "hero_title"),
        subtitle=get_ui_text("quiz", "hero_subtitle")
    )

    vs_manager = ensure_vector_store_loaded(embedding_model, vector_db_choice)
    if not vs_manager:
        st.info(get_ui_text("quiz", "not_loaded_hint"))
        return

    col1, col2, col3 = st.columns([2, 1, 1])
    all_label = get_ui_text("chat", "all_documents_choice")
    with col1:
        topic = st.text_input(get_ui_text("quiz", "topic_label"), placeholder=get_ui_text("quiz", "topic_placeholder"))
    with col2:
        num_questions = st.number_input(get_ui_text("quiz", "count_label"), min_value=1, max_value=10, value=5)
    with col3:
        source_filter_ui = st.selectbox(get_ui_text("quiz", "source_label"), options=st.session_state.document_sources, key="quiz_source")
        source_filter = "All Documents" if source_filter_ui == all_label else source_filter_ui

    if st.button(get_ui_text("quiz", "generate_button"), use_container_width=True, key="quiz_gen_action_btn"):
        if not topic:
            st.error(get_ui_text("quiz", "topic_error"))
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
        st.success(get_ui_text("quiz", "generating_background").format(topic=topic))

    quiz_jobs = [j for j in st.session_state.background_jobs.values() if j.get("type") == "quiz"]
    if quiz_jobs:
        latest_job = quiz_jobs[-1]
        if latest_job["status"] == "running":
            st.info(f"{get_ui_text('quiz', 'compiling_notice')} '{latest_job['title']}'...")
            if st.button(get_ui_text("quiz", "refresh_button"), key="refresh_quiz"):
                st.rerun()
        elif latest_job["status"] == "completed" and latest_job["result"]:
            st.session_state.current_quiz = latest_job["result"]
            st.session_state.quiz_topic = latest_job["title"]
        elif latest_job["status"] == "failed":
            st.error(get_ui_text("quiz", "generation_failed").format(error=latest_job.get("error")))

    if "current_quiz" in st.session_state and st.session_state.current_quiz:
        st.markdown(get_ui_text("quiz", "exam_heading").format(topic=st.session_state.quiz_topic))

        with st.form("quiz_form"):
            user_answers = {}
            for i, q in enumerate(st.session_state.current_quiz):
                render_quiz_card(index=i+1, total=len(st.session_state.current_quiz), question=q.get("question", ""))
                options = q.get("options", [])
                if options:
                    user_answers[i] = st.radio(
                        get_ui_text("quiz", "option_radio_label").format(index=i+1),
                        options=options,
                        key=f"q_{i}",
                        label_visibility="collapsed"
                    )
                render_spacer("spacer-12")

            submitted = st.form_submit_button(get_ui_text("quiz", "submit_button"), use_container_width=True)
            if submitted:
                st.session_state.quiz_submitted = True
                st.session_state.user_answers = user_answers

        if st.session_state.get("quiz_submitted", False):
            st.markdown(get_ui_text("quiz", "results_heading"))
            score = 0
            for i, q in enumerate(st.session_state.current_quiz):
                correct_ans = q.get("answer", "")
                user_ans = st.session_state.user_answers.get(i, "")

                is_correct = user_ans and user_ans[0] == correct_ans[0]
                if is_correct:
                    score += 1
                    st.success(get_ui_text("quiz", "correct_label").format(index=i+1, answer=correct_ans))
                else:
                    st.error(get_ui_text("quiz", "incorrect_label").format(index=i+1, user_ans=user_ans, correct_ans=correct_ans))

            if score == len(st.session_state.current_quiz):
                st.balloons()

            percentage = int((score / len(st.session_state.current_quiz)) * 100)
            render_score_card(score=score, total=len(st.session_state.current_quiz), percentage=percentage, label=get_ui_text("quiz", "score_label"))

def render_flashcards_page(llm_provider, llm_model, llm_temperature, embedding_model, vector_db_choice):
    render_hero_banner(
        tag=get_ui_text("flashcards", "hero_tag"),
        title=get_ui_text("flashcards", "hero_title"),
        subtitle=get_ui_text("flashcards", "hero_subtitle")
    )

    vs_manager = ensure_vector_store_loaded(embedding_model, vector_db_choice)
    if not vs_manager:
        st.info(get_ui_text("flashcards", "not_loaded_hint"))
        return

    col1, col2, col3 = st.columns([2, 1, 1])
    all_label = get_ui_text("chat", "all_documents_choice")
    with col1:
        topic = st.text_input(get_ui_text("flashcards", "topic_label"), placeholder=get_ui_text("flashcards", "topic_placeholder"))
    with col2:
        num_cards = st.number_input(get_ui_text("flashcards", "count_label"), min_value=1, max_value=15, value=5)
    with col3:
        source_filter_ui = st.selectbox(get_ui_text("flashcards", "source_label"), options=st.session_state.document_sources, key="flashcard_source")
        source_filter = "All Documents" if source_filter_ui == all_label else source_filter_ui

    if st.button(get_ui_text("flashcards", "generate_button"), use_container_width=True, key="fc_gen_action_btn"):
        if not topic:
            st.error(get_ui_text("flashcards", "topic_error"))
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
        st.success(get_ui_text("flashcards", "generating_background").format(topic=topic))

    fc_jobs = [j for j in st.session_state.background_jobs.values() if j.get("type") == "flashcard"]
    if fc_jobs:
        latest_job = fc_jobs[-1]
        if latest_job["status"] == "running":
            st.info(get_ui_text("flashcards", "compiling_notice").format(topic=latest_job["title"]))
            if st.button(get_ui_text("flashcards", "refresh_button"), key="ref_fc"):
                st.rerun()
        elif latest_job["status"] == "completed" and latest_job["result"]:
            st.session_state.current_flashcards = latest_job["result"]
            st.session_state.flashcard_topic = latest_job["title"]
        elif latest_job["status"] == "failed":
            st.error(get_ui_text("flashcards", "generation_failed").format(error=latest_job.get("error")))

    if "current_flashcards" in st.session_state and st.session_state.current_flashcards:
        cards = st.session_state.current_flashcards
        st.markdown(get_ui_text("flashcards", "deck_heading").format(topic=st.session_state.flashcard_topic, count=len(cards)))

        for i, card in enumerate(cards, 1):
            render_flashcard_box(
                index=i,
                total=len(cards),
                front=card.get("front", ""),
                back=card.get("back", "")
            )

def render_summary_page(llm_provider, llm_model, llm_temperature, embedding_model, vector_db_choice):
    render_hero_banner(
        tag=get_ui_text("summary", "hero_tag"),
        title=get_ui_text("summary", "hero_title"),
        subtitle=get_ui_text("summary", "hero_subtitle")
    )

    vs_manager = ensure_vector_store_loaded(embedding_model, vector_db_choice)
    if not vs_manager:
        st.info(get_ui_text("summary", "not_loaded_hint"))
        return

    col1, col2 = st.columns([3, 1])
    all_label = get_ui_text("chat", "all_documents_choice")
    with col1:
        topic = st.text_input(get_ui_text("summary", "topic_label"), placeholder=get_ui_text("summary", "topic_placeholder"))
    with col2:
        source_filter_ui = st.selectbox(get_ui_text("summary", "source_label"), options=st.session_state.document_sources, key="summary_source")
        source_filter = "All Documents" if source_filter_ui == all_label else source_filter_ui

    if st.button(get_ui_text("summary", "generate_button"), use_container_width=True, key="summary_gen_action_btn"):
        if not topic:
            st.error(get_ui_text("summary", "topic_error"))
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
        st.success(get_ui_text("summary", "generating_background").format(topic=topic))

    sum_jobs = [j for j in st.session_state.background_jobs.values() if j.get("type") == "summary"]
    if sum_jobs:
        latest_job = sum_jobs[-1]
        if latest_job["status"] == "running":
            st.info(get_ui_text("summary", "compiling_notice").format(topic=latest_job["title"]))
            if st.button(get_ui_text("summary", "refresh_button"), key="ref_sum"):
                st.rerun()
        elif latest_job["status"] == "completed" and latest_job["result"]:
            st.session_state.current_summary = latest_job["result"]
            st.session_state.summary_topic = latest_job["title"]
        elif latest_job["status"] == "failed":
            st.error(get_ui_text("summary", "generation_failed").format(error=latest_job.get("error")))

    if "current_summary" in st.session_state and st.session_state.current_summary:
        st.markdown(get_ui_text("summary", "guide_heading").format(topic=st.session_state.summary_topic))
        render_glass_card_open()
        st.markdown(st.session_state.current_summary)
        render_glass_card_close()

def render_history_page():
    render_hero_banner(
        tag=get_ui_text("history", "hero_tag"),
        title=get_ui_text("history", "hero_title"),
        subtitle=get_ui_text("history", "hero_subtitle")
    )

    history_items = load_history()

    if not history_items:
        render_empty_state(
            title=get_ui_text("history", "empty_state", "title"),
            subtitle=get_ui_text("history", "empty_state", "subtitle"),
            icon="📜"
        )
        return

    col_filter, col_clear = st.columns([3, 1])
    with col_filter:
        filter_type = st.selectbox(
            get_ui_text("history", "filter_label"),
            get_ui_text("history", "filter_options")
        )
    with col_clear:
        render_spacer("spacer-28")
        if st.button(get_ui_text("history", "purge_button"), use_container_width=True, key="history_purge_all_btn"):
            clear_all_history()
            st.success(get_ui_text("history", "purge_success"))
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

    st.markdown(get_ui_text("history", "showing_label").format(count=len(filtered_items)))

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

        with st.expander(f"{icon} {i_title} — {i_time}", expanded=False):
            col_info, col_del = st.columns([5, 1])
            with col_del:
                if st.button(get_ui_text("history", "delete_entry_button"), key=f"del_hist_{item_id}"):
                    delete_history_entry(item_id)
                    st.success(get_ui_text("history", "deleted_success"))
                    time.sleep(0.3)
                    st.rerun()

            with col_info:
                st.caption(get_ui_text("history", "indexed_source_label").format(source=i_source))

                if i_type == "quiz":
                    questions = i_details.get("questions", [])
                    st.markdown(get_ui_text("history", "quiz_questions_label").format(count=len(questions)))
                    for q_idx, q in enumerate(questions, 1):
                        st.markdown(f"**Q{q_idx}: {q.get('question', '')}**")
                        for opt in q.get("options", []):
                            st.markdown(f"- {opt}")
                        st.markdown(f"✅ **Answer:** `{q.get('answer', '')}`")
                        st.markdown("---")

                elif i_type == "flashcard":
                    cards = i_details.get("cards", [])
                    st.markdown(get_ui_text("history", "deck_size_label").format(count=len(cards)))
                    for c_idx, card in enumerate(cards, 1):
                        st.markdown(f"**Card {c_idx}:** {card.get('front', '')}")
                        st.markdown(f"> **Definition / Answer:** {card.get('back', '')}")

                elif i_type == "summary":
                    summary_text = i_details.get("summary_text", "")
                    st.markdown(summary_text)

                elif i_type == "chat":
                    st.markdown(f"**Inquiry:** {i_details.get('query', '')}")
                    st.markdown(f"**Verified Response:** {i_details.get('answer', '')}")

                elif i_type == "material":
                    st.markdown(f"**Operation:** {i_details.get('action', 'updated').title()}")
                    st.markdown(f"**Document Name:** `{i_details.get('doc_name', '')}`")

if __name__ == "__main__":
    main()
