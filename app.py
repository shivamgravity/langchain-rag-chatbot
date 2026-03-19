import streamlit as st
import os
from rag_pipeline import create_rag_chain

st.set_page_config(page_title="RAG Chatbot", layout="centered")

# 🎯 Header
st.title("🤖 RAG Chatbot")
st.caption("Chat with your documents using AI • Powered by Groq + LangChain")

# 🧩 Sidebar
with st.sidebar:
    st.header("📂 Upload Documents")

    uploaded_files = st.file_uploader(
        "Upload PDFs",
        type="pdf",
        accept_multiple_files=True
    )

    if st.button("🧹 Clear Chat"):
        st.session_state.chat_history = []
        st.session_state.last_sources = []

# 🧠 Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "last_sources" not in st.session_state:
    st.session_state.last_sources = []

if "rag" not in st.session_state:
    st.session_state.rag = None

# 📄 Process uploaded PDFs
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

    # 🔥 Recreate RAG if new files uploaded
    if st.session_state.rag is None:
        with st.spinner("🔄 Processing documents..."):
            st.session_state.rag = create_rag_chain(file_paths)

# 💤 Empty state
if not uploaded_files:
    st.info("👈 Upload one or more PDFs from the sidebar to start chatting.")

# 💬 Chat Input
query = st.chat_input("Ask a question...")

if query and st.session_state.rag:
    with st.spinner("🤖 Thinking..."):
        result = st.session_state.rag(query)

    answer = result["answer"]
    sources = result["sources"]

    # Save chat
    st.session_state.chat_history.append(("You", query))
    st.session_state.chat_history.append(("Bot", answer))

    # Save sources
    st.session_state.last_sources = sources

# 🧾 Display Chat
for i, (sender, message) in enumerate(st.session_state.chat_history):
    if sender == "You":
        with st.chat_message("user"):
            st.markdown(message)
    else:
        with st.chat_message("assistant"):
            st.markdown(message)

            # Show sources only for last bot message
            if i == len(st.session_state.chat_history) - 1:
                if st.session_state.last_sources:
                    with st.expander("📄 Sources"):
                        for src in st.session_state.last_sources:
                            st.write(src)