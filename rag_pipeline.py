from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

from langchain.chains import RetrievalQA


def create_rag_chain(pdf_path):
    # 1. Load PDF
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # 2. Split text
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    docs = splitter.split_documents(documents)

    # 3. Embeddings (FREE)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # 4. Vector DB
    db = Chroma.from_documents(docs, embeddings)

    # 5. Retriever
    retriever = db.as_retriever(search_kwargs={"k": 3})

    # 6. Groq LLM (FREE + FAST)
    llm = ChatGroq(
        model_name="llama3-8b-8192"
    )

    # 7. RAG Chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever
    )

    return qa_chain


# 🔥 Test it
if __name__ == "__main__":
    chain = create_rag_chain("sample.pdf")
    print(chain.invoke("What is this document about?"))