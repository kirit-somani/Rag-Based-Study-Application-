# 📚 StudyRAG: AI-Powered Study Assistant

An intelligent, retrieval-augmented generation (RAG) powered study assistant designed for students and educators. Upload course materials, lecture slides, or textbooks (PDF format) and instantly unlock interactive Q&A with direct citation tracking, practice quizzes with auto-grading, digital flashcards, chapter summaries, and persistent study history.

---

## 🌟 Key Features

- 📖 **PDF Document Ingestion**: Upload course materials with background chunking, text cleaning, and instant indexing.
- 💬 **Interactive RAG Q&A**: Ask detailed questions based on your notes and receive factual answers with exact document and page-level references.
- 📝 **Practice Quiz Generator**: Generate custom multiple-choice quizzes (MCQs) for any topic in your library with built-in instant grading.
- 🗂 **Digital Flashcards**: Automatically convert complex concepts, definitions, and terms into study card decks.
- 📑 **Topic Summarizer**: Create structured revision guides and key takeaway summaries on demand.
- 📜 **Study History & Persistence**: Store, filter, view, or delete past generated quizzes, flashcards, summaries, and chat sessions.
- ⚙️ **Multi-Model Support**: Seamless integration with **Groq**, **OpenAI**, and local **Ollama** LLMs paired with `sentence-transformers` embeddings and **FAISS** / **ChromaDB** vector indices.
- 🎨 **Modern Student-Centric Interface**: Fully responsive UI supporting custom Light & Dark modes, non-blocking background worker threads, and smooth navigation.

---

## 🏗️ Architecture Overview

```
rag-study-assistant/
├── app.py                      # Main Streamlit application & interactive UI
├── config/
│   └── settings.yaml           # Default chunking, vector store, and retrieval parameters
├── src/
│   ├── embeddings/             # SentenceTransformer embedding manager
│   ├── evaluation/             # Retrieval performance metrics & evaluation
│   ├── generation/             # LLM provider integration & study content generators
│   ├── ingestion/              # PDF extraction (PyMuPDF), document cleaning, recursive chunking
│   ├── rag/                    # Main RAG response pipeline
│   ├── retrieval/              # Vector query processor, score filtering, and retrieval pipeline
│   ├── utils/                  # File utils, logger, and persistent history manager
│   └── vectorstore/            # FAISS and ChromaDB vector store implementations
├── tests/                      # Unit test suite for ingestion, embeddings, vector stores, and RAG
├── requirements.txt            # Python dependencies
└── pytest.ini                  # Pytest test suite configuration
```

---

## 🚀 Quick Start Guide

### 1. Prerequisites
- Python `3.10+` recommended
- `pip` package manager
- (Optional) API keys for **Groq** (`GROQ_API_KEY`) or **OpenAI** (`OPENAI_API_KEY`), or a local **Ollama** instance.

### 2. Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/kirit-somani/Rag-Based-Study-Application-.git
   cd Rag-Based-Study-Application-
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Configuration:**
   Copy `.env.example` to `.env` and set your API keys if applicable:
   ```bash
   cp .env.example .env
   ```
   *Example `.env`:*
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   OPENAI_API_KEY=your_openai_api_key_here
   ```

---

## 💻 Running the Application

Launch the Streamlit web application:

```bash
streamlit run app.py
```

Open your browser and navigate to `http://localhost:8501`.

---

## 🧪 Running Unit Tests

Execute the test suite using `pytest`:

```bash
pytest
```

---

## 🛠️ Tech Stack & Dependencies

- **Frontend & App Framework**: [Streamlit](https://streamlit.io/)
- **RAG & Orchestration**: [LangChain](https://www.langchain.com/)
- **Embeddings**: `sentence-transformers` (`all-MiniLM-L6-v2`)
- **Vector Search**: [FAISS](https://github.com/facebookresearch/faiss) / [ChromaDB](https://www.trychroma.com/)
- **PDF Extraction**: PyMuPDF (`fitz`)
- **LLM Integrations**: Groq Cloud, OpenAI, Ollama
- **Data & Testing**: Pandas, PyYAML, Pytest

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).
