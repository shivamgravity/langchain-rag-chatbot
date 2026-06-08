from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

import shutil

from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Embeddings (HuggingFace)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)

def create_rag_chain(pdf_paths):
    # Load PDFs
    documents = []
    for path in pdf_paths:
        loader = PyPDFLoader(path)
        documents.extend(loader.load())

    # Split text into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    docs = splitter.split_documents(documents)

    # Vector DB (Persistent Chroma)

    persist_directory = "./chroma_db"

    if os.path.exists(persist_directory):
        shutil.rmtree(persist_directory)

    db = Chroma.from_documents(
        docs,
        embeddings,
        persist_directory=persist_directory
    )

    # Retriever (Improved)
    # retriever = db.as_retriever(
    #     search_type="similarity_score_threshold",
    #     search_kwargs={
    #         "k": 5,
    #         "score_threshold": 0.5
    #     }
    # )
    retriever = db.as_retriever(
        search_kwargs={"k": 5}
    )

    # Groq LLM
    llm = ChatGroq(model_name="llama-3.1-8b-instant")

    # Memory (conversation history)
    chat_history = []

    # RAG Query Function
    def rag_query(question):
        # Retrieve relevant docs
        docs = retriever.invoke(question)

        if not docs:
            return {
                "answer": "I couldn't find relevant information in the document.",
                "sources": []
            }

        # Build context
        context = "\n\n".join([doc.page_content for doc in docs])
        
        sources = []

        for doc in docs:
            sources.append({
                "file": os.path.basename(
                    doc.metadata.get("source", "Unknown")
                ),
                "page": doc.metadata.get("page", "Unknown") + 1
            })

        # Use last few exchanges for memory
        history_text = "\n".join(chat_history[-4:])  # last 2 Q&A pairs

        # Prompt
        prompt = f"""
You are an intelligent AI assistant.

Use the provided context to answer the question.

Conversation so far:
{history_text}

If the question is general (like "what is this document about"),
summarize the document based on the context.

If context is partial, still try to answer using available information.

Be clear, structured, and helpful.

If the answer is not in the context, say "I don't know".

Context:
{context}

Question:
{question}
"""

        # LLM response
        response = llm.invoke(prompt)

        # Save conversation
        chat_history.append(f"User: {question}")
        chat_history.append(f"Assistant: {response.content}")

        return {
            "answer": response.content,
            "sources": sources
        }

    return rag_query


# Test block
if __name__ == "__main__":
    rag = create_rag_chain(["sample.pdf"])  # MUST be list
    result = rag("What is this document about?")
    print(result["answer"])
    print("Sources:", result["sources"])