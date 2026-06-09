from langchain_groq import ChatGroq


def generate_suggested_questions(docs):

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
    Generate exactly 3 useful exploratory questions.

    Rules:
    - Maximum 12 words per question.
    - Keep questions concise.
    - Focus on key concepts and takeaways.
    - Return only questions.
    - One question per line.
    - End every question with a question mark.

    Focus on:
    - key concepts
    - practical takeaways
    - important ideas
    - recommendations
    - summaries

    Do not ask about:
    - authors
    - acknowledgements
    - copyright pages
    - ISBN numbers
    - publication details

    Document Content:
    
    {sample_text}
    """

    response = llm.invoke(prompt)

    questions = [
        q.strip()
        for q in response.content.split("\n")
        if q.strip()
    ]

    questions = list(dict.fromkeys(questions))

    return questions[:3]