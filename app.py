import streamlit as st
from rag_pipeline import create_rag_chain

st.title("📚 RAG Chatbot")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    st.success("PDF uploaded successfully!")

    if "chain" not in st.session_state:
        st.session_state.chain = create_rag_chain("temp.pdf")

    query = st.text_input("Ask a question:")

    if query:
        response = st.session_state.chain.run(query)
        st.write(response)