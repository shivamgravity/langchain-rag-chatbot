import streamlit as st
from datetime import datetime


def build_chat_export():
    """
    Builds and returns the exportable Markdown string from the current
    chat history in session state. Mirrors the original export logic exactly.

    Returns
    -------
    str
        A Markdown-formatted string of the full conversation.
    """
    chat_export = ""

    for chat in st.session_state.chat_history:

        if chat["role"] == "user":

            chat_export += (
                f"## User\n\n"
                f"{chat['message']}\n\n"
            )

        else:

            chat_export += (
                f"## Assistant\n\n"
                f"{chat['message']}\n\n"
            )

            if chat.get("sources"):

                chat_export += "### Sources\n\n"

                seen = set()

                for src in chat["sources"]:

                    key = (
                        src["file"],
                        src["page"]
                    )

                    if key in seen:
                        continue

                    seen.add(key)

                    chat_export += (
                        f"- {src['file']} "
                        f"(Page {src['page']})\n"
                    )

                chat_export += "\n"

    return chat_export


def render_chat():
    """
    Handles the chat input, runs the RAG query, appends to history,
    renders the full conversation with sources, and shows the download button.
    Mirrors the original chat block exactly.

    Returns
    -------
    str
        The exportable Markdown string of the current conversation,
        used by the download button.
    """
    # Prompting the user to ask questions after summary generation
    if st.session_state.rag:

        st.divider()

        st.subheader(
            "\u2753 Ask Questions"
        )

        st.caption(
            "Answers are generated from the uploaded documents."
        )

    # Chat input
    query = st.chat_input("Ask a question...")

    # If user clicks on a suggested question, use that as the query
    if (
        query is None
        and st.session_state.selected_question
    ):
        query = st.session_state.selected_question
        st.session_state.selected_question = None

    if query and st.session_state.rag:
        with st.spinner("🤖 Thinking..."):
            result = st.session_state.rag(query)

        answer = result["answer"]
        sources = result["sources"]
        retrieved_chunks = result["retrieved_chunks"]
        confidence = result["confidence"]

        # Save chat
        st.session_state.chat_history.append(
            {
                "role": "user",
                "message": query
            }
        )

        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "message": answer,
                "sources": sources,
                "retrieved_chunks": retrieved_chunks,
                "confidence": confidence
            }
        )
        # No explicit rerun needed; Streamlit will render the updated chat history in this run

    # Build exportable chat content
    chat_export = build_chat_export()

    # Chat rendering with sources and retrieved chunk count
    # Display Chat
    for chat in st.session_state.chat_history:

        if chat["role"] == "user":

            # Display user message
            with st.chat_message("user"):
                st.markdown(chat["message"])

        else:

            # Display assistant message
            with st.chat_message("assistant"):

                st.markdown(chat["message"])

                # Show number of sources used
                number_of_sources = None
                if chat.get("retrieved_chunks"):
                    number_of_sources = chat["retrieved_chunks"]

                # Show confidence in result
                confidence_symbol = None
                if chat.get("confidence"):
                    confidence = chat["confidence"]
                    if confidence == "Low":
                        confidence_symbol = "\U0001F534"
                    elif confidence == "Medium":
                        confidence_symbol = "\U0001F7E1"
                    else:
                        confidence_symbol = "\U0001F7E2"

                if number_of_sources and confidence_symbol:
                    st.caption(
                        f"📚 {chat['retrieved_chunks']} sources used  \t  {confidence_symbol} Confidence: {confidence}"
                    )
                elif number_of_sources:
                    st.caption(
                        f"📚 {chat['retrieved_chunks']} sources used"
                    )
                else:
                    st.caption(
                        f"{confidence_symbol} Confidence: {confidence}"
                    )

                if chat.get("sources"):

                    with st.expander("📄 View Sources"):

                        col1, col2 = st.columns(2)

                        seen = set()

                        button_idx = 0

                        for src in chat["sources"]:

                            # Avoid showing duplicate sources if multiple chunks come from the same page
                            key = (
                                src["file"],
                                src["page"]
                            )

                            if key in seen:
                                continue

                            seen.add(key)

                            # Create a citation string like "Document (Page 3)"
                            citation = (
                                f"{src['file']} "
                                f"(Page {src['page']})"
                            )

                            # Create a snippet for each source
                            content = src["content"].strip().replace("\n", " ")
                            first_period = content.find(".")

                            if first_period != -1:
                                snippet = content[first_period + 1:].strip()
                            else:
                                snippet = content

                            snippet = snippet[:150]

                            if len(snippet) < 100:
                                snippet = content[:150]

                            if len(src["content"]) > 100:
                                snippet += "..."

                            # Combine citation and snippet for button content
                            button_content = f"**📄 {citation}**\n\n{snippet}"

                            if button_idx % 2 == 0:
                                target_col = col1
                            else:
                                target_col = col2

                            with target_col:

                                with st.container(border=True):

                                    st.markdown(citation)
                                    st.caption(snippet)

                            button_idx += 1

    # Download conversation as markdown file
    if st.session_state.chat_history:

        st.download_button(
            label="📥 Download Conversation",
            data=chat_export,
            file_name=(
                f"chat_"
                f"{datetime.now():%Y%m%d_%H%M%S}.md"
            ),
            mime="text/markdown"
        )
