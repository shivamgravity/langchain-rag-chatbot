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
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    query = st.chat_input("Ask a question:")

    if query:
        # Call only once
        response = st.session_state.rag(query)

        # Save chat
        st.session_state.chat_history.append(("You", query))
        st.session_state.chat_history.append(("Bot", response))

    # Display chat
    for sender, message in st.session_state.chat_history:
        if sender == "You":
            st.chat_message("user").write(message)
        else:
            st.chat_message("assistant").write(message)