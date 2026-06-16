import streamlit as st
from src.document_manager import get_documents


def render_sidebar():
    """
    Renders the sidebar: file uploader, Document Library, selection controls,
    and chat-clearing button.

    Returns
    -------
    uploaded_files : list[UploadedFile] | None
        The raw return value of st.file_uploader, needed by the main page
        to gate dashboard rendering and RAG processing.
    """
    with st.sidebar:

        # Document upload section
        st.header("📂 Upload Documents")

        # Document file uploader button
        uploaded_files = st.file_uploader(
            "Upload PDFs",
            type="pdf",
            accept_multiple_files=True
        )

        # Document Library - documents that are already uploaded
        documents = get_documents()

        st.subheader(f"📚 Document Library \u2022 {len(documents)}")

        if documents:
            for document_id, info in documents.items():

                # Cleaning the display name of the file
                clean_name = (
                    info["display_name"]
                    .replace(".pdf", "")
                    .replace("-", " ")
                    .replace("_", " ")
                )
                display_metric = (
                    f"\n"
                    f"                ({info['pages']} pages "
                    f"\u2022 "
                    f"{info['chunks']} chunks)"
                )

                display_name = clean_name + display_metric

                if st.checkbox(
                    f"{display_name}",
                    key=f"{document_id}",
                ):
                    # Adding selected document to selected_documents active state using document id
                    st.session_state.selected_documents.add(document_id)

                    # debugging
                    print(st.session_state.selected_documents)

                else:
                    # Removing the document id from the selected_documents session state
                    st.session_state.selected_documents.discard(document_id)

                    # debugging
                    print(st.session_state.selected_documents)

        else:
            st.caption(
                "No saved documents yet."
            )

        # Show current selected documents
        if st.session_state.selected_documents:

            st.markdown("### ✅ Selected Documents")

            documents = get_documents()

            for document_id in (
                st.session_state.selected_documents
            ):

                st.write(
                    f"📄 "
                    f"{documents[document_id]['display_name']}"
                )

        def clear_selection():
            st.session_state.selected_documents.clear()
            for doc_id in get_documents():
                if doc_id in st.session_state:
                    st.session_state[doc_id] = False

        # Clear all the selected documents from session state
        st.button(
            "❌ Clear Selection",
            use_container_width=True,
            on_click=clear_selection
        )

        # Button to clear the chat
        if st.button("🧹 Clear Chat"):
            st.session_state.chat_history = []

            st.session_state.rag = None
            st.session_state.summary = None
            st.session_state.num_documents = 0
            st.session_state.num_pages = 0
            st.session_state.suggested_questions = None
            st.session_state.uploaded_doc_names = []

    return uploaded_files
