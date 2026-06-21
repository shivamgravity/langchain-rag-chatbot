import streamlit as st
import os
import warnings
import uuid

from src.rag_pipeline import init_rag_system, process_and_add_document
from src.document_manager import add_document, get_documents
from src.ui.sidebar import render_sidebar
from src.ui.dashboard import render_dashboard
from src.ui.chat import render_chat_history, handle_chat_input
from src.auth import authenticate_user, register_user, log_action

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Enterprise AI Assistant", layout="wide")

# --- Basic Authentication ---
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.user_id = None

if not st.session_state.logged_in:
    st.title("🔒 Enterprise Login")
    tab1, tab2 = st.tabs(["Login", "Register"])
    
    with tab1:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login")
            if submitted:
                user_id = authenticate_user(username, password)
                if user_id:
                    st.session_state.logged_in = True
                    st.session_state.user_id = user_id
                    log_action(user_id, "login")
                    st.rerun()
                else:
                    st.error("Invalid credentials")
                    
    with tab2:
        with st.form("register_form"):
            new_username = st.text_input("New Username")
            new_password = st.text_input("New Password", type="password")
            reg_submitted = st.form_submit_button("Register")
            if reg_submitted:
                if register_user(new_username, new_password):
                    st.success("Registration successful! Please login.")
                else:
                    st.error("Username already exists or error occurred.")
    st.stop()

# Custom CSS
st.markdown("""
<style>
div.stButton > button {
    width: 100%;
    text-align: left !important;
}
div.stButton > button p {
    text-align: left !important;
}
div[data-testid="column"]:has(#sticky-anchor),
div[data-testid="stColumn"]:has(#sticky-anchor),
div.block-container div[data-testid="stHorizontalBlock"]:first-of-type > div:nth-child(2) {
    align-self: flex-start !important;
    position: -webkit-sticky !important;
    position: sticky !important;
    top: 3rem !important;
    height: fit-content !important;
    max-height: calc(100vh - 6rem) !important;
    overflow-y: auto !important;
}
@media (min-width: 50.5rem) {
    div[data-testid="stChatInput"] {
        width: calc((100% - 2rem) * 2.5 / 3.5) !important;
    }
}
</style>
""", unsafe_allow_html=True)

st.title("📚 Enterprise Knowledge Assistant")
st.caption("Upload documents, generate insights, and ask questions with 100% data privacy.")

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "rag" not in st.session_state:
    st.session_state.rag = None
if "summary" not in st.session_state:
    st.session_state.summary = None
if "num_documents" not in st.session_state:
    st.session_state.num_documents = 0
if "num_pages" not in st.session_state:
    st.session_state.num_pages = 0
if "suggested_questions" not in st.session_state:
    st.session_state.suggested_questions = None
if "uploaded_doc_names" not in st.session_state:
    st.session_state.uploaded_doc_names = []
if "selected_question" not in st.session_state:
    st.session_state.selected_question = None
if "num_chunks" not in st.session_state:
    st.session_state.num_chunks = 0
if "embedding_model" not in st.session_state:
    st.session_state.embedding_model = None
if "llm_model" not in st.session_state:
    st.session_state.llm_model = None
if "selected_documents" not in st.session_state:
    st.session_state.selected_documents = set()
if "last_selection_set" not in st.session_state:
    st.session_state.last_selection_set = set()
if "active_sources" not in st.session_state:
    st.session_state.active_sources = None

uploaded_files = render_sidebar()

if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()

# 1. Process newly uploaded PDFs
if uploaded_files:
    new_files = [f for f in uploaded_files if f.file_id not in st.session_state.processed_files]
    if new_files:
        os.makedirs("data/documents", exist_ok=True)

        for file in new_files:
            document_id = uuid.uuid4().hex[:8]
            path = os.path.join("data/documents", f"{document_id}.pdf")
            with open(path, "wb") as f:
                f.write(file.read())

            with st.spinner(f"🔄 Processing and embedding {file.name}..."):
                try:
                    doc_metadata = process_and_add_document(path, document_id)
                    
                    add_document(
                        filename=file.name,
                        pages=doc_metadata["num_pages"],
                        chunks=doc_metadata["num_chunks"],
                        user_id=st.session_state.user_id,
                        summary=doc_metadata["summary"],
                        questions=doc_metadata["suggested_questions"],
                        document_id=document_id
                    )
                    log_action(st.session_state.user_id, "upload_doc", file.name)
                    st.session_state.selected_documents.add(document_id)
                    st.session_state.processed_files.add(file.file_id)
                except Exception as e:
                    st.error(f"Failed to process {file.name}: {str(e)}")

        st.success(f"{len(new_files)} PDFs uploaded and processed!")
        st.rerun() # Refresh to update sidebar

# 2. Add selected documents from the library
documents_metadata = get_documents(st.session_state.user_id)

selected_ids = list(st.session_state.selected_documents)
selected_ids = [did for did in selected_ids if did in documents_metadata]

current_selection_set = set(selected_ids)

# Check if selection changed or if LLM toggle changed
use_local_llm = st.session_state.get("use_local_llm", False)
if (st.session_state.last_selection_set != current_selection_set) or (st.session_state.get("last_llm_state") != use_local_llm):
    if current_selection_set:
        with st.spinner("🔄 Initializing RAG pipeline..."):
            rag_system = init_rag_system(list(current_selection_set), documents_metadata, use_local_llm)

            st.session_state.rag = rag_system["query_fn"]
            
            total_pages = sum([documents_metadata[did]["pages"] for did in current_selection_set])
            total_chunks = sum([documents_metadata[did]["chunks"] for did in current_selection_set])
            
            combined_summary = "\n\n".join([f"**{documents_metadata[did]['display_name']}**:\n{documents_metadata[did].get('summary', '')}" for did in current_selection_set])
            all_questions = []
            for did in current_selection_set:
                all_questions.extend(documents_metadata[did].get("questions", []))
            
            st.session_state.summary = combined_summary
            st.session_state.num_documents = len(current_selection_set)
            st.session_state.num_pages = total_pages
            st.session_state.suggested_questions = list(set(all_questions))[:4]
            st.session_state.num_chunks = total_chunks
            st.session_state.embedding_model = rag_system["embedding_model"]
            st.session_state.llm_model = rag_system["llm_model"]
            
            st.session_state.uploaded_doc_names = [documents_metadata[did]["display_name"] for did in current_selection_set]
            
            st.session_state.last_selection_set = current_selection_set
            st.session_state.last_llm_state = use_local_llm
    else:
        st.session_state.rag = None
        st.session_state.summary = None
        st.session_state.num_documents = 0
        st.session_state.num_pages = 0
        st.session_state.suggested_questions = None
        st.session_state.num_chunks = 0
        st.session_state.uploaded_doc_names = []
        st.session_state.last_selection_set = set()
        st.session_state.last_llm_state = use_local_llm

main_col, side_col = st.columns([2.5, 1], gap="large")

with main_col:
    render_dashboard(bool(current_selection_set))

    if not current_selection_set:
        st.info("👈 Upload or select one or more PDFs from the sidebar to start chatting.")

    chat_container = st.container()
    with chat_container:
        render_chat_history()

with side_col:
    st.markdown('<div id="sticky-anchor"></div>', unsafe_allow_html=True)
    st.subheader("📄 Source Context")
    if st.session_state.active_sources:
        seen = set()
        for src in st.session_state.active_sources:
            key = (src["file"], src["page"])
            if key in seen:
                continue
            seen.add(key)
            citation = f"{src['file']} (Page {src['page']})"
            content = src.get("content", "").strip()
            if len(content) > 100:
                content = content[:100] + "..."
            with st.container(border=True):
                st.markdown(f"**{citation}**")
                st.markdown(f"> {content}")
    else:
        st.caption("Click 'View Sources' on a chat message to see the context here.")

handle_chat_input(chat_container)