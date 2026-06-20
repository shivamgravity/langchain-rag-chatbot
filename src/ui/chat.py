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


def _render_sources(chat):
    number_of_sources = chat.get("retrieved_chunks")
    confidence_symbol = None
    if chat.get("confidence"):
        confidence = chat["confidence"]
        if confidence == "Low":
            confidence_symbol = "🔴"
        elif confidence == "Medium":
            confidence_symbol = "🟡"
        else:
            confidence_symbol = "🟢"

    if number_of_sources and confidence_symbol:
        st.caption(f"📚 {number_of_sources} sources used  \t  {confidence_symbol} Confidence: {confidence}")
    elif number_of_sources:
        st.caption(f"📚 {number_of_sources} sources used")
    elif confidence_symbol:
        st.caption(f"{confidence_symbol} Confidence: {confidence}")

    if chat.get("sources"):
        # We need a unique key for the button, we can use the hash of the message content or just an index
        # Since we don't pass an index right now, we can use a hash of the first source content as a pseudo-unique key
        unique_key = f"src_btn_{hash(chat['message'][:50])}"
        if st.button("📄 View Sources", key=unique_key):
            st.session_state.active_sources = chat["sources"]

def render_chat_history():
    if st.session_state.rag:
        st.divider()
        st.subheader("\u2753 Ask Questions")
        st.caption("Answers are generated from the uploaded documents.")

    for chat in st.session_state.chat_history:
        if chat["role"] == "user":
            with st.chat_message("user"):
                st.markdown(chat["message"])
        else:
            with st.chat_message("assistant"):
                st.markdown(chat["message"])
                _render_sources(chat)

    chat_export = build_chat_export()

    if st.session_state.chat_history:
        st.download_button(
            label="📥 Download Conversation",
            data=chat_export,
            file_name=f"chat_{datetime.now():%Y%m%d_%H%M%S}.md",
            mime="text/markdown"
        )


def handle_chat_input(chat_container):
    query = st.chat_input("Ask a question...", disabled=not bool(st.session_state.rag))

    if query is None and st.session_state.selected_question:
        query = st.session_state.selected_question
        st.session_state.selected_question = None

    if query and st.session_state.rag:
        with chat_container:
            with st.chat_message("user"):
                st.markdown(query)
                
            with st.chat_message("assistant"):
                with st.spinner("🤖 Thinking..."):
                    result = st.session_state.rag(query, st.session_state.chat_history)
                
                def stream_response():
                    for chunk in result["answer_stream"]:
                        if hasattr(chunk, 'content'):
                            yield chunk.content
                        else:
                            yield str(chunk)

                answer = st.write_stream(stream_response)
                
                assistant_msg = {
                    "role": "assistant",
                    "message": answer,
                    "sources": result["sources"],
                    "retrieved_chunks": result["retrieved_chunks"],
                    "confidence": result["confidence"]
                }
                
                _render_sources(assistant_msg)
                
                st.session_state.chat_history.append({"role": "user", "message": query})
                st.session_state.chat_history.append(assistant_msg)
                
                st.session_state.active_sources = result["sources"]
                st.rerun()
