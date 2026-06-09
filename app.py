import streamlit as st
import os
from src.rag_pipeline import create_rag_chain

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

# Sidebar
with st.sidebar:
    st.header("📂 Upload Documents")

    uploaded_files = st.file_uploader(
        "Upload PDFs",
        type="pdf",
        accept_multiple_files=True
    )

    if st.button("🧹 Clear Chat"):
        st.session_state.chat_history = []

        st.session_state.rag = None
        st.session_state.summary = None
        st.session_state.num_documents = 0
        st.session_state.num_pages = 0
        st.session_state.suggested_questions = None
        st.session_state.uploaded_doc_names = []

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

# Process uploaded PDFs
if uploaded_files:
    if not os.path.exists("temp_docs"):
        os.makedirs("temp_docs")

    file_paths = []
    uploaded_doc_names = []

    for file in uploaded_files:
        path = os.path.join("temp_docs", file.name)

        with open(path, "wb") as f:
            f.write(file.read())

        file_paths.append(path)
        uploaded_doc_names.append(file.name)

    st.success(f"{len(file_paths)} PDFs uploaded!")

    # Recreate RAG if new files uploaded
    if (
        set(st.session_state.uploaded_doc_names) != set(uploaded_doc_names)
    ):
        with st.spinner("🔄 Processing documents..."):
            rag_system = create_rag_chain(file_paths)

            st.session_state.rag = rag_system["query_fn"]
            st.session_state.summary = rag_system["summary"]
            st.session_state.num_documents = rag_system["num_documents"]
            st.session_state.num_pages = rag_system["num_pages"]
            st.session_state.suggested_questions = rag_system["suggested_questions"]
            st.session_state.uploaded_doc_names = uploaded_doc_names # Update session state with new doc names

    # Displaying the summary of uploaded documents

    if st.session_state.summary:

        st.subheader("📊 Document Overview")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Documents Uploaded",
                st.session_state.num_documents
            )

        with col2:
            st.metric(
                "Pages Indexed",
                st.session_state.num_pages
            )
        
        st.markdown("### 📂 Uploaded Documents")

        for doc in st.session_state.uploaded_doc_names:
            clean_name = doc.replace("_", " ").replace(".pdf", "")
            st.markdown(f"📄 {clean_name}")

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

# Empty state
if not uploaded_files:
    st.info("👈 Upload one or more PDFs from the sidebar to start chatting.")

# Prompting the user to ask questions after summary generation

if st.session_state.rag:

    st.divider()

    st.subheader(
        "\u2753 Ask Questions"
    )

    st.caption(
        "Answers are generated from the uploaded documents."
    )

# Chat input

query = st.chat_input("Ask a question...")

# If user clicks on a suggested question, use that as the query

if (
    query is None
    and st.session_state.selected_question
):
    query = st.session_state.selected_question
    st.session_state.selected_question = None

if query and st.session_state.rag:
    with st.spinner("🤖 Thinking..."):
        result = st.session_state.rag(query)

    answer = result["answer"]
    sources = result["sources"]

    # Save chat

    st.session_state.chat_history.append(
        {
            "role": "user",
            "message": query
        }
    )

    st.session_state.chat_history.append(
        {
            "role": "assistant",
            "message": answer,
            "sources": sources
        }
    )

# Display Chat
for chat in st.session_state.chat_history:

    if chat["role"] == "user":

        with st.chat_message("user"):
            st.markdown(chat["message"])

    else:

        with st.chat_message("assistant"):

            st.markdown(chat["message"])

            if chat.get("sources"):

                with st.expander("📄 Sources"):

                    seen = set()

                    for src in chat["sources"]:

                        citation = (
                            f"{src['file']} "
                            f"(Page {src['page']})"
                        )

                        if citation not in seen:
                            seen.add(citation)
                            st.write(citation)