import streamlit as st
import os
from src.rag_pipeline import create_rag_chain
from src.document_manager import (
    add_document,
    get_documents
)

from datetime import datetime
import warnings

# Do not show the warnings in console output
warnings.filterwarnings("ignore")

# Chat export initialization
chat_export = f"""
# AI Knowledge Assistant Conversation

Generated:
{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

"""

st.set_page_config(page_title="AI Assistant", layout="centered")

# Custom CSS to left-align buttons
st.markdown("""
<style>
div.stButton > button {
    width: 100%;
    text-align: left !important;
}

div.stButton > button p {
    text-align: left !important;
}
</style>
""", unsafe_allow_html=True)

# Header
st.title("📚 AI Knowledge Assistant")
st.caption(
    "Upload documents, generate insights, and ask questions."
)

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "rag" not in st.session_state:
    st.session_state.rag = None

if "summary" not in st.session_state:
    st.session_state.summary = None

if "num_documents" not in st.session_state:
    st.session_state.num_documents = 0

if "num_pages" not in st.session_state:
    st.session_state.num_pages = 0

if "suggested_questions" not in st.session_state:
    st.session_state.suggested_questions = None

if "uploaded_doc_names" not in st.session_state:
    st.session_state.uploaded_doc_names = []

if "selected_question" not in st.session_state:
    st.session_state.selected_question = None

if "num_chunks" not in st.session_state:
    st.session_state.num_chunks = 0

if "embedding_model" not in st.session_state:
    st.session_state.embedding_model = None

if "llm_model" not in st.session_state:
    st.session_state.llm_model = None

if "selected_documents" not in st.session_state:
    st.session_state.selected_documents = set()

# Sidebar
with st.sidebar:
    
    # Document upload section
    st.header("📂 Upload Documents")

    # Document file uploader button
    uploaded_files = st.file_uploader(
        "Upload PDFs",
        type="pdf",
        accept_multiple_files=True
    )

    # # Selected Documents

    # st.subheader("\U0001F516 Selected Documents")

    # if st.session_state.selected_documents:
    #     selected_docs = ""
    #     line = 1
    #     for selected_doc in st.session_state.selected_documents:
    #         # Cleaning the name
    #         cleaned_name = (
    #             selected_doc
    #             .replace(".pdf", "")
    #             .replace("-", " ")
    #             .replace("_", " ")
    #         )
    #         last_char = "\n" if line < len(st.session_state.selected_documents) else ""
    #         selected_docs += str(line) + ". " + cleaned_name + last_char
    #         line += 1
    #     st.info(
    #         f"{selected_docs}"
    #     )

    # Document Library - documents that are already uploaded

    documents = get_documents()

    st.subheader(f"📚 Document Library \u2022 {len(documents)}")

    if documents:
        for document_id, info in documents.items():

            # Cleaning the display name of the file
            clean_name = (
                info["display_name"]
                .replace(".pdf", "")
                .replace("-", " ")
                .replace("_", " ")
            )
            display_metric = f"\n\
                ({info["pages"]} pages \
                \u2022 \
                {info["chunks"]} chunks)"

            display_name = clean_name + display_metric

            if st.checkbox(
                f"{display_name}",
                value = True,
                key = f"{document_id}",
            ):
                # Adding selected document to selected_documents active state using document id
                st.session_state.selected_documents.add(document_id)
                
                # debugging
                print(st.session_state.selected_documents)

            else:
                # Removing the document id from the selected_documents session state
                st.session_state.selected_documents.discard(document_id)

                # debugging
                print(st.session_state.selected_documents)

    else:
        st.caption(
            "No saved documents yet."
        )
    
    # Show current selected documents

    if st.session_state.selected_documents:

        st.markdown("### ✅ Selected Documents")

        documents = get_documents()

        for document_id in (
            st.session_state.selected_documents
        ):

            st.write(
                f"📄 "
                f"{documents[document_id]['display_name']}"
            )
    
    # Clear all the selected documents from session state
    
    if st.button(
        "❌ Clear Selection",
        use_container_width=True
    ):

        st.session_state.selected_documents.clear()

        st.rerun()
    
    # # Show the number of documents selected
    # st.caption(
    #     f"Selected Documents: {len(st.session_state.selected_documents)}"
    # )

    # Button to clear the chat
    if st.button("🧹 Clear Chat"):
        st.session_state.chat_history = []

        st.session_state.rag = None
        st.session_state.summary = None
        st.session_state.num_documents = 0
        st.session_state.num_pages = 0
        st.session_state.suggested_questions = None
        st.session_state.uploaded_doc_names = []

# Process uploaded PDFs
if uploaded_files:
    os.makedirs(
        "data/documents",
        exist_ok=True
    )

    file_paths = []
    uploaded_doc_names = []

    for file in uploaded_files:
        path = os.path.join(
            "data/documents",
            file.name
        )
        with open(path, "wb") as f:
            f.write(file.read())

        file_paths.append(path)
        uploaded_doc_names.append(file.name)

    st.success(f"{len(file_paths)} PDFs uploaded!")

    # Recreate RAG if new files uploaded
    if (
        set(st.session_state.uploaded_doc_names) != set(uploaded_doc_names)
    ):
        with st.spinner("🔄 Processing documents..."):

            # Generating vector space embeddings to understand the document
            rag_system = create_rag_chain(file_paths)

            # Updating the session states with fresh values

            st.session_state.rag = rag_system["query_fn"]
            st.session_state.summary = rag_system["summary"]
            st.session_state.num_documents = rag_system["num_documents"]
            st.session_state.num_pages = rag_system["num_pages"]
            st.session_state.suggested_questions = rag_system["suggested_questions"]
            st.session_state.num_chunks = rag_system["num_chunks"]
            st.session_state.embedding_model = rag_system["embedding_model"]
            st.session_state.llm_model = rag_system["llm_model"]
            st.session_state.uploaded_doc_names = uploaded_doc_names # Update session state with new doc names

            # Adding the document to the persist directory
            # Currently, adding only the first file if mutiple are uploaded - for testing purpose.
            
            if len(uploaded_files) == 1:

                document_id, pdf_path = add_document(
                    filename=uploaded_files[0].name,
                    pages=rag_system["num_pages"],
                    chunks=rag_system["num_chunks"]
                )

    # Displaying the summary of uploaded documents

    if st.session_state.summary:

        st.subheader("📊 Document Overview")

        col11, col12, col13, col14 = st.columns(4)

        with col11:
            st.metric(
                "Documents Uploaded",
                st.session_state.num_documents
            )

        with col12:
            st.metric(
                "Pages Indexed",
                st.session_state.num_pages
            )
        
        with col13:
            st.metric(
                "Chunks",
                st.session_state.num_chunks
            )
        
        with col14:
            chunks_per_page = round(
                st.session_state.num_chunks / st.session_state.num_pages,
                2
            )
            st.metric(
                "Chunks/Page",
                chunks_per_page
            )
        
        with st.expander("⚙️ System Information"):
            
            col21, col22, col23 = st.columns(3)
            
            with col21:
                st.metric(
                    "Embedding",
                    "MiniLM-L6-v2"
                )
            
            with col22:
                st.metric(
                    "LLM",
                    "Llama 3.1 8B"
                )
            
            with col23:
                st.metric(
                    "Retriever",
                    "ChromaDB"
                )

        # Showing uploaded documents names in much better readable format
        
        st.markdown("### 📂 Uploaded Documents")

        for doc in st.session_state.uploaded_doc_names:
            clean_name = doc.replace("_", " ").replace(".pdf", "")
            st.markdown(f"📄 {clean_name}")
        
        # Executive summary
        with st.expander(
            "📄 Executive Summary",
        ):
            # Summary of the extracted chunks
            st.markdown(st.session_state.summary)

        if st.session_state.suggested_questions:

            st.markdown("### 💡 Suggested Questions")

            col1, col2 = st.columns(2)

            for idx, question in enumerate(
                st.session_state.suggested_questions
            ):

                target_col = col1 if idx % 2 == 0 else col2

                with target_col:

                    if st.button(
                        question,
                        key=f"suggested_question_{idx}",
                        use_container_width=True
                    ):
                        st.session_state.selected_question = question
                        st.rerun()

# Empty state
if not uploaded_files:
    st.info("👈 Upload one or more PDFs from the sidebar to start chatting.")

# Adding documents to export before starting conversation
chat_export += "## Documents\n\n"

for doc in st.session_state.uploaded_doc_names:
    chat_export += f"- {doc}\n"

chat_export += "\n---\n\n"

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

# Build exportable chat content
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
                    f"📚 {chat["retrieved_chunks"]} sources used  \t  {confidence_symbol} Confidence: {confidence}"
                )
            elif number_of_sources:
                st.caption(
                    f"📚 {chat["retrieved_chunks"]} sources used"
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