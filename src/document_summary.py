from langchain_groq import ChatGroq


def generate_document_summary(docs):
    """
    Generate a high-level summary of uploaded documents.
    """

    # Take first few chunks to keep prompt small
    sample_text = "\n\n".join(
        [doc.page_content for doc in docs[:10]]
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