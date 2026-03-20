# 🤖 RAG Chatbot (Multi-PDF AI Assistant)

A production-ready **Retrieval-Augmented Generation (RAG)** chatbot that allows users to upload multiple PDFs and ask questions based on their content.

Built using **LangChain, Groq LLM, HuggingFace embeddings, and Streamlit**, this app provides fast, accurate, and context-aware answers with source attribution.

---

## 🚀 Live Demo

🔗 https://shivamgravity-rag-chatbot.streamlit.app/

---

## ✨ Features

* 📂 **Multi-PDF Upload** — Query multiple documents at once
* 🧠 **RAG Pipeline** — Context-aware answers using retrieved document chunks
* ⚡ **Fast LLM Inference** — Powered by Groq (LLaMA 3)
* 🔍 **Source Attribution** — See which documents were used for answers
* 💬 **Conversational Memory** — Supports follow-up questions
* 🎨 **ChatGPT-style UI** — Clean and interactive interface
* ☁️ **Deployed on Streamlit Cloud**

---

## 🏗️ Tech Stack

* **Frontend:** Streamlit
* **LLM:** Groq (LLaMA 3)
* **Framework:** LangChain (modular)
* **Embeddings:** HuggingFace (sentence-transformers)
* **Vector Database:** ChromaDB
* **Document Loader:** PyPDF

---

## 🧠 Architecture

```
User Query
   ↓
Retriever (ChromaDB)
   ↓
Relevant Chunks
   ↓
Prompt + Context
   ↓
Groq LLM
   ↓
Answer + Sources
```

---

## ⚙️ Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/shivamgravity/langchain-rag-chatbot.git
cd rag-chatbot
```

---

### 2. Create virtual environment

```bash
python -m venv venv
venv\Scripts\activate   # Windows
# or
source venv/bin/activate  # Mac/Linux
```

---

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

### 4. Add API Key

Create a `.env` file:

```
GROQ_API_KEY=your_api_key_here
```

---

### 5. Run the app

```bash
streamlit run app.py
```

---

## 🌐 Deployment

The app is deployed using **Streamlit Community Cloud**.

To deploy:

1. Push code to GitHub
2. Connect repo to Streamlit Cloud
3. Add `GROQ_API_KEY` in **Secrets**
4. Deploy 🚀

---

## 📂 Project Structure

```
rag-chatbot/
│
├── app.py                # Streamlit UI
├── rag_pipeline.py       # RAG logic
├── requirements.txt      # Dependencies
├── .env                  # API keys (not committed)
└── temp_docs/            # Uploaded PDFs (ignored)
```

---

## 🎯 Key Learnings

* Implemented **Retrieval-Augmented Generation (RAG)** from scratch
* Worked with **LangChain’s modular architecture**
* Integrated **Groq LLM for fast inference**
* Built **multi-document querying system**
* Designed **chat-based UI with memory and sources**

---

## 🏆 Future Improvements

* 🔍 Highlight exact text used for answers (Explainability)
* 📊 Confidence scoring
* ⚡ Faster retrieval optimization
* 👤 User authentication
* 🌐 Full-stack deployment (FastAPI + React)

---

## 👨‍💻 Author

**Shivam**

* GitHub: https://github.com/shivamgravity
* LinkedIn: https://www.linkedin.com/in/shivam-gravity

---

## ⭐ If you like this project

Give it a star ⭐ on GitHub!

---
