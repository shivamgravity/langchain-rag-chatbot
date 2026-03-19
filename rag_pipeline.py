from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

from dotenv import load_dotenv

load_dotenv()

def create_rag_chain(pdf_paths):
    # 1. Load PDF
    documents = []

    for path in pdf_paths:
        loader = PyPDFLoader(path)
        documents.extend(loader.load())

    # 2. Split text
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    docs = splitter.split_documents(documents)

    # 3. Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # 4. Vector DB
    db = Chroma.from_documents(docs, embeddings)

    # 5. Retriever
    retriever = db.as_retriever(search_kwargs={"k": 5})

    # 6. Groq LLM
    llm = ChatGroq(model_name="llama-3.1-8b-instant")

    # 🔥 NEW RAG FUNCTION (no chains)
    def rag_query(question):
        docs = retriever.invoke(question)

        context = "\n\n".join([doc.page_content for doc in docs])
        sources = [doc.metadata.get("source", "Unknown") for doc in docs]

        if not docs:
            return "I couldn't find relevant information in the document."

        context = "\n\n".join([doc.page_content for doc in docs])

        prompt = f"""
        You are a helpful AI assistant.

        If the question is general (like "what is this document about"),
        try to summarize the document using the context.

        Answer ONLY from the context.
        If no relevant info is found, say "I don't know".

        Context:
        {context}

        Question:
        {question}
        """

        response = llm.invoke(prompt)

        return {
            "answer": response.content,
            "sources": sources
        }

    return rag_query


# 🔥 Test
if __name__ == "__main__":
    rag = create_rag_chain("sample.pdf")
    print(rag("What is this document about?"))