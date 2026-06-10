from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from src.document_summary import generate_document_summary
from src.question_suggestions import generate_suggested_questions

import shutil
import uuid

from dotenv import load_dotenv
import os

import warnings

# Ignore warnings for cleaner output
warnings.filterwarnings("ignore")

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
        chunk_size=1000,
        chunk_overlap=150,
        separators=[
            "\n\n", 
            "\n", 
            ". ", 
            " ", 
            ""
        ]
    )
    docs = splitter.split_documents(documents)

    # document summary

    document_summary = generate_document_summary(docs)

    # question suggestions

    suggested_questions = generate_suggested_questions(docs)

    # Vector DB (Persistent Chroma)

    persist_directory = "./chroma_db"

    collection_name = f"docs_{uuid.uuid4().hex[:8]}"

    db = Chroma.from_documents(
        docs,
        embeddings,
        persist_directory=persist_directory,
        collection_name=collection_name
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
        search_type="mmr",
        search_kwargs={
            "k": 5,
            "fetch_k": 15
        }
    )

    # Groq LLM
    llm = ChatGroq(model_name="llama-3.1-8b-instant")

    # Memory (conversation history)
    chat_history = []

    # RAG Query Function
    def rag_query(question):

        # Use last few exchanges for memory
        history_text = "\n".join(chat_history[-4:])  # last 2 Q&A pairs

        # Question Rewriting Prompt (for better retrieval in follow-up questions)
        rewrite_prompt = f"""
        Given the conversation history and latest user question,
        rewrite the latest question so it is fully standalone.

        Conversation:

        {history_text}

        Latest Question:

        {question}

        Return only the rewritten question.
        """

        if history_text:
            # Rewritten question (for better retrieval in follow-up questions)
            rewritten_question = llm.invoke(
                rewrite_prompt
            ).content.strip()
        else:
            # No history, use original question
            rewritten_question = question

        # Retrieve relevant docs
        docs = retriever.invoke(
            rewritten_question
        )

        # Confidence score based on the number of chunks retrieved
        if len(docs) < 3:
            confidence = "Low"
        elif len(docs) < 5:
            confidence = "Medium"
        else:
            confidence = "High"

        if not docs:
            return {
                "answer": "I couldn't find relevant information in the document.",
                "sources": []
            }

        # Build context
        context = "\n\n".join([doc.page_content for doc in docs])
        
        sources = []

        for doc in docs:

            sources.append(
                {
                    "file": os.path.basename(
                        doc.metadata["source"]
                    ),
                    "page": doc.metadata.get("page", 0) + 1,
                    "content": doc.page_content
                }
            )

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

        If the answer is not in the context, say "I couldn't find enough information in the uploaded documents."

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
            "sources": sources,
            "retrieved_chunks": len(docs),
            "confidence": confidence
        }

    return {
        "query_fn": rag_query,
        "summary": document_summary,
        "suggested_questions": suggested_questions,
        "num_documents": len(pdf_paths),
        "num_chunks": len(docs),
        "num_pages": len(documents)
    }


# Test block
if __name__ == "__main__":
    rag = create_rag_chain(["sample.pdf"])  # MUST be list
    result = rag("What is this document about?")
    print(result["answer"])
    print("Sources:", result["sources"])