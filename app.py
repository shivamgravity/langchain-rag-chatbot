import streamlit as st
from rag_pipeline import create_rag_chain

st.set_page_config(page_title="RAG Chatbot", layout="centered")

st.title("📚 Chat with Your PDF")

# Upload PDF
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    st.success("PDF uploaded successfully!")

    # Create chain once
    if "rag" not in st.session_state:
        st.session_state.rag = create_rag_chain("temp.pdf")

    # Chat input
    # query = st.text_input("Ask a question:") # real time interaction without a memory
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    query = st.text_input("Ask a question:")

    if query:
        response = st.session_state.rag(query)

        st.session_state.chat_history.append(("You", query))
        st.session_state.chat_history.append(("Bot", response))

    for sender, message in st.session_state.chat_history:
        st.write(f"**{sender}:** {message}")

    if query:
        response = st.session_state.rag(query)
        st.write("### 🤖 Answer:")
        st.write(response)