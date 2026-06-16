import streamlit as st
import os
import warnings

from src.rag_pipeline import create_rag_chain
from src.document_manager import add_document, get_documents
from src.ui.sidebar import render_sidebar
from src.ui.dashboard import render_dashboard
from src.ui.chat import render_chat

# Do not show the warnings in console output
warnings.filterwarnings("ignore")

st.set_page_config(page_title="AI Assistant", layout="centered")

# Custom CSS to left-align buttons
st.markdown("""
<style>
div.stButton > button {
    width: 100%;
    text-align: left !important;
}

div.stButton > button p {
    text-align: left !important;
}
</style>
""", unsafe_allow_html=True)

# Header
st.title("📚 AI Knowledge Assistant")
st.caption(
    "Upload documents, generate insights, and ask questions."
)

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

# Render sidebar and receive the file uploader widget value.
# uploaded_files must be captured here because the main page
# body references it to gate dashboard rendering and RAG processing.
uploaded_files = render_sidebar()

# Initialize lists for paths and names
file_paths = []
doc_names = []

# 1. Process newly uploaded PDFs
if uploaded_files:
    os.makedirs(
        "data/documents",
        exist_ok=True
    )

    for file in uploaded_files:
        path = os.path.join(
            "data/documents",
            file.name
        )
        with open(path, "wb") as f:
            f.write(file.read())

        file_paths.append(path)
        doc_names.append(file.name)

    st.success(f"{len(uploaded_files)} PDFs uploaded!")

# 2. Add selected documents from the library
documents_metadata = get_documents()

for doc_id in st.session_state.selected_documents:
    if doc_id in documents_metadata:
        info = documents_metadata[doc_id]
        doc_name = info["display_name"]
        
        # Avoid duplicates if the same file is uploaded and selected
        if doc_name not in doc_names:
            doc_names.append(doc_name)
            path = os.path.join("data/documents", doc_name)
            if os.path.exists(path):
                file_paths.append(path)

# Recreate RAG if active files have changed
if set(st.session_state.uploaded_doc_names) != set(doc_names):
    if file_paths:
        with st.spinner("🔄 Processing documents..."):

            # Generating vector space embeddings to understand the document
            rag_system = create_rag_chain(file_paths)

            # Updating the session states with fresh values

            st.session_state.rag = rag_system["query_fn"]
            st.session_state.summary = rag_system["summary"]
            st.session_state.num_documents = rag_system["num_documents"]
            st.session_state.num_pages = rag_system["num_pages"]
            st.session_state.suggested_questions = rag_system["suggested_questions"]
            st.session_state.num_chunks = rag_system["num_chunks"]
            st.session_state.embedding_model = rag_system["embedding_model"]
            st.session_state.llm_model = rag_system["llm_model"]
            st.session_state.uploaded_doc_names = doc_names  # Update session state with new doc names

            # Adding the document to the persist directory
            # Currently, adding only the first file if multiple are uploaded - for testing purpose.

            if uploaded_files and len(uploaded_files) == 1:

                document_id, pdf_path = add_document(
                    filename=uploaded_files[0].name,
                    pages=rag_system["num_pages"],
                    chunks=rag_system["num_chunks"]
                )
    else:
        # Clear RAG if no files are selected/uploaded
        st.session_state.rag = None
        st.session_state.summary = None
        st.session_state.num_documents = 0
        st.session_state.num_pages = 0
        st.session_state.suggested_questions = None
        st.session_state.num_chunks = 0
        st.session_state.uploaded_doc_names = []

# Render document overview dashboard (metrics, summary, suggested questions)
render_dashboard(bool(file_paths))

# Empty state
if not file_paths:
    st.info("👈 Upload or select one or more PDFs from the sidebar to start chatting.")

# Render chat (input, history, sources, download)
render_chat()