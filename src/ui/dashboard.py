import streamlit as st


def render_dashboard(has_files):
    """
    Renders the document overview dashboard and suggested questions panel.
    Only shown when has_files is truthy and a summary exists in session state.

    Parameters
    ----------
    has_files : bool
        A flag indicating if there are uploaded or selected documents.
    """
    if not has_files:
        return

    if not st.session_state.summary:
        return

    st.subheader("📊 Document Overview")

    col11, col12, col13, col14 = st.columns(4)

    with col11:
        st.metric(
            "Documents Uploaded",
            st.session_state.num_documents
        )

    with col12:
        st.metric(
            "Pages Indexed",
            st.session_state.num_pages
        )

    with col13:
        st.metric(
            "Chunks",
            st.session_state.num_chunks
        )

    with col14:
        chunks_per_page = round(
            st.session_state.num_chunks / st.session_state.num_pages,
            2
        )
        st.metric(
            "Chunks/Page",
            chunks_per_page
        )

    with st.expander("⚙️ System Information"):

        col21, col22, col23 = st.columns(3)

        with col21:
            st.metric(
                "Embedding",
                "MiniLM-L6-v2"
            )

        with col22:
            st.metric(
                "LLM",
                "Llama 3.1 8B"
            )

        with col23:
            st.metric(
                "Retriever",
                "ChromaDB"
            )

    # Showing uploaded documents names in much better readable format
    st.markdown("### 📂 Uploaded Documents")

    for doc in st.session_state.uploaded_doc_names:
        clean_name = doc.replace("_", " ").replace(".pdf", "")
        st.markdown(f"📄 {clean_name}")

    # Executive summary
    with st.expander(
        "📄 Executive Summary",
    ):
        # Summary of the extracted chunks
        st.markdown(st.session_state.summary)

    if st.session_state.suggested_questions:

        st.markdown("### 💡 Suggested Questions")

        col1, col2 = st.columns(2)

        for idx, question in enumerate(
            st.session_state.suggested_questions
        ):

            target_col = col1 if idx % 2 == 0 else col2

            with target_col:

                if st.button(
                    question,
                    key=f"suggested_question_{idx}",
                    use_container_width=True
                ):
                    st.session_state.selected_question = question
                    st.rerun()
