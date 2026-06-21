from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from src.document_summary import generate_document_summary
from src.question_suggestions import generate_suggested_questions

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

def process_and_add_document(pdf_path, document_id):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    docs = splitter.split_documents(documents)
    
    for doc in docs:
        doc.metadata["document_id"] = document_id
        
    num_chunks = len(docs)
    num_pages = len(documents)
    
    # Store in persistent collection
    db = Chroma.from_documents(
        docs,
        embeddings,
        persist_directory="./chroma_db",
        collection_name="enterprise_docs"
    )
    
    document_summary = generate_document_summary(docs)
    suggested_questions = generate_suggested_questions(docs)
    
    return {
        "num_pages": num_pages,
        "num_chunks": num_chunks,
        "summary": document_summary,
        "suggested_questions": suggested_questions
    }


def init_rag_system(selected_document_ids, document_metadata_map, use_local_llm=False):
    db = Chroma(
        persist_directory="./chroma_db",
        collection_name="enterprise_docs",
        embedding_function=embeddings
    )
    
    filter_dict = {"document_id": {"$in": selected_document_ids}} if selected_document_ids else {}
    
    retriever = db.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 5,
            "fetch_k": 15,
            "filter": filter_dict
        }
    )
    
    if use_local_llm:
        llm = ChatOllama(model="gemma3:1b")
        llm_model_name = "Ollama (Local)"
    else:
        llm = ChatGroq(model_name="llama-3.1-8b-instant")
        llm_model_name = "Groq (Cloud)"
        
    def rag_query(question, session_history=None):
        if session_history is None:
            session_history = []

        history_text = "\n".join(
            [f"{msg['role'].capitalize()}: {msg['message']}" for msg in session_history[-4:]]
        )

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
            try:
                rewritten_question = llm.invoke(rewrite_prompt).content.strip()
            except Exception:
                rewritten_question = question
        else:
            rewritten_question = question

        docs = retriever.invoke(rewritten_question)

        if len(docs) < 3:
            confidence = "Low"
        elif len(docs) < 5:
            confidence = "Medium"
        else:
            confidence = "High"

        if not docs:
            def empty_stream():
                yield "I couldn't find relevant information in the selected documents."
            return {
                "answer_stream": empty_stream(),
                "sources": [],
                "retrieved_chunks": 0,
                "confidence": confidence
            }

        context = "\n\n".join([doc.page_content for doc in docs])
        
        sources = []
        for doc in docs:
            doc_id = doc.metadata.get("document_id")
            display_name = document_metadata_map.get(doc_id, {}).get("display_name", os.path.basename(doc.metadata.get("source", "Unknown")))
            sources.append({
                "file": display_name,
                "page": doc.metadata.get("page", 0) + 1,
                "content": doc.page_content
            })

        prompt = f"""
        You are an intelligent AI assistant.

        Use the provided context to answer the question.

        Conversation so far:
        {history_text}

        If the question is general, summarize the document based on the context.
        If context is partial, still try to answer using available information.
        Be clear, structured, and helpful.

        Context:
        {context}

        Question:
        {question}
        """

        try:
            response_stream = llm.stream(prompt)
        except Exception as e:
            def error_stream():
                yield type('obj', (object,), {'content': f"⚠️ I encountered an error while generating the response. Please try again. (Error: {str(e)})"})()
            response_stream = error_stream()

        return {
            "answer_stream": response_stream,
            "sources": sources,
            "retrieved_chunks": len(docs),
            "confidence": confidence
        }
    
    return {
        "query_fn": rag_query,
        "embedding_model": "all-MiniLM-L6-v2",
        "llm_model": llm_model_name
    }