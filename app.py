import streamlit as st
from rag_pipeline import create_rag_chain

st.set_page_config(page_title="RAG Chatbot", layout="centered")

# st.title("📚 Chat with Your PDF")
st.title("🤖 RAG Chatbot")
st.caption("Chat with your documents using AI")

# Upload PDF
uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)

import os

if uploaded_files:
    if not os.path.exists("temp_docs"):
        os.makedirs("temp_docs")

    file_paths = []

    for file in uploaded_files:
        path = os.path.join("temp_docs", file.name)
        with open(path, "wb") as f:
            f.write(file.read())
        file_paths.append(path)

    st.success(f"{len(file_paths)} PDFs uploaded!")

    # Create chain once
    if "rag" not in st.session_state:
        st.session_state.rag = create_rag_chain(file_paths)

    # Chat input
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    query = st.chat_input("Ask a question:")

    if query:
        # Call the RAG function
        result = st.session_state.rag(query)

        answer = result["answer"]
        sources = result["sources"]

        # Save chat (only answer)
        st.session_state.chat_history.append(("You", query))
        st.session_state.chat_history.append(("Bot", answer))

        # Save sources separately
        st.session_state.last_sources = sources

    # Display chat
    for i, (sender, message) in enumerate(st.session_state.chat_history):
        if sender == "You":
            st.chat_message("user").write(message)
        else:
            with st.chat_message("assistant"):
                st.write(message)

                # Show sources ONLY for last response
                if i == len(st.session_state.chat_history) - 1 and "last_sources" in st.session_state:
                    with st.expander("📄 Sources"):
                        for src in st.session_state.last_sources:
                            st.write(src)