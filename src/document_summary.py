from langchain_groq import ChatGroq


def generate_document_summary(docs):
    """
    Generate a high-level summary of uploaded documents.
    """

    # Take a sample of documents from the beginning and middle of the list for context
    middle = len(docs) // 2

    sample_docs = (
        docs[:3] +
        docs[middle:middle+5]
    )

    sample_text = "\n\n".join(
        [doc.page_content for doc in sample_docs]
    )

    llm = ChatGroq(
        model_name="llama-3.1-8b-instant"
    )

    prompt = f"""
You are analyzing uploaded documents.

Provide:

1. A concise overview (3-5 sentences)
2. Key topics (bullet list)
3. Important concepts (bullet list)

Document Content:

{sample_text}
"""

    response = llm.invoke(prompt)

    return response.content