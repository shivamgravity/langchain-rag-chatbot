import streamlit as st
from src.document_manager import get_documents


def render_sidebar():
    with st.sidebar:
        st.header("⚙️ Configuration")
        st.session_state.use_local_llm = st.toggle("Use Local LLM (Ollama)", value=st.session_state.get("use_local_llm", False))
        if st.session_state.use_local_llm:
            st.caption("🔒 100% Private (Air-gapped). Make sure Ollama is running locally.")
        else:
            st.caption("☁️ Using Groq Cloud API for maximum speed.")
            
        st.divider()

        # Document upload section
        st.header("📂 Upload Documents")
        uploaded_files = st.file_uploader(
            "Upload PDFs",
            type="pdf",
            accept_multiple_files=True
        )

        # Document Library
        documents = get_documents(st.session_state.user_id)
        st.subheader(f"📚 Document Library \u2022 {len(documents)}")

        if documents:
            for document_id, info in documents.items():
                clean_name = info["display_name"].replace(".pdf", "").replace("-", " ").replace("_", " ")
                display_metric = f"\n({info['pages']} pages \u2022 {info['chunks']} chunks)"
                display_name = clean_name + display_metric

                if st.checkbox(f"{display_name}", key=f"{document_id}"):
                    st.session_state.selected_documents.add(document_id)
                else:
                    st.session_state.selected_documents.discard(document_id)
        else:
            st.caption("No saved documents yet.")

        if st.session_state.selected_documents:
            st.markdown("### ✅ Selected Documents")
            for document_id in st.session_state.selected_documents:
                if document_id in documents:
                    st.badge(f"📄 {documents[document_id]['display_name']}")

        def clear_selection():
            st.session_state.selected_documents.clear()
            for doc_id in get_documents(st.session_state.user_id):
                if doc_id in st.session_state:
                    st.session_state[doc_id] = False

        st.button("❌ Clear Selection", use_container_width=True, on_click=clear_selection)

        if st.button("🧹 Clear Chat"):
            st.session_state.chat_history = []
            st.session_state.active_sources = None
            # We don't reset rag system here, just the chat history
            
        st.divider()
        
        if st.button("🚪 Log Out", use_container_width=True, type="primary"):
            st.session_state.clear()
            st.rerun()

    return uploaded_files
