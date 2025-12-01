# Agentic RAG Application

A hybrid Retrieval-Augmented Generation (RAG) system with agentic capabilities that intelligently routes queries between local document knowledge and web search to provide accurate answers.

## Overview

This application combines traditional RAG with agentic AI to create a smart question-answering system. It first searches through local PDF documents using vector similarity search. If the local knowledge base doesn't contain sufficient information, it automatically deploys AI agents to search the web and retrieve relevant information.

## Features

- **Hybrid Knowledge Retrieval**: Automatically determines whether to use local documents or web search
- **Semantic Caching**: In-memory cache with similarity search to avoid redundant LLM calls (20-item capacity)
- **Vector Database**: FAISS-based vector store for efficient document similarity search
- **Multi-LLM Support**: Uses Groq (Llama) for document Q&A and Gemini for web agent tasks
- **Web Scraping Agents**: CrewAI-powered agents for intelligent web search and content extraction
- **Multiple Interfaces**:
  - FastAPI REST API for programmatic access
  - Streamlit web interface for interactive queries
  - CLI for direct execution
- **Smart Routing**: LLM-powered decision making to determine answer source

## Architecture

```
User Query
    ↓
[Semantic Cache Check] → Cache Hit? → Return Cached Answer
    ↓ (Cache Miss)
[Vector Database Search] → Local Documents
    ↓
[Router LLM] → Can answer locally?
    ├─ Yes → Generate answer from local context
    └─ No → [Web Scraping Agents] → Generate answer from web content
```

## Installation

### Prerequisites

- Python >3.10, <3.13. 3.12.7 recommended.
- [uv](https://github.com/astral-sh/uv) package manager

### Setup

1. **Clone the repository** (if applicable) or navigate to the project directory
2. **Install dependencies:**

   ```bash
   uv sync
   ```
3. **Create a `.env` file** in the project root with your API keys:

   ```env
   GROQ_API_KEY=your_groq_api_key
   GEMINI_API_KEY=your_gemini_api_key
   SERPER_API_KEY=your_serper_api_key
   ```
4. **Add PDF documents** to the `documents/` directory

## Usage

### FastAPI REST API

Start the API server:

```bash
fastapi run serving_api/main.py --host 0.0.0.0 --port 8000 # use main_v2.py if
# you want to use langgraph instead of CrewAI
```

**API Endpoints:**

- `GET /health` - Health check endpoint
- `POST /query` - Submit a query
  ```json
  {
    "query": "What is Agentic RAG?"
  }
  ```

**Interactive API Docs:**

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### Streamlit Web Interface

Launch the interactive web interface:

```bash
streamlit run app/main.py
```

Navigate to `http://localhost:8501` in your browser.

### Command Line

Run queries directly:

```bash
python pipeline/agentic_rag.py # Use agentic-rag_v2.py to use langgraph instead of CrewAI
```

## Project Structure

```
agentic_RAG/
├── app/                    # Streamlit web interface
│   └── main.py
├── serving_api/            # FastAPI REST API
│   └── main.py
├── pipeline/               # Core RAG pipeline
│   └── agentic_rag.py      # Main processing logic
├── documents/              # PDF documents for RAG
│   └── *.pdf
├── pyproject.toml          # Project dependencies
└── README.md
```

## Technologies Used

- **LangChain**: Document processing and vector stores
- **FAISS**: Vector similarity search
- **CrewAI**: Multi-agent orchestration for web search
- **Groq**: Fast LLM inference (Llama models)
- **Google Gemini**: Agent LLM for web tasks
- **FastAPI**: REST API framework
- **Streamlit**: Web interface
- **HuggingFace**: Sentence transformers for embeddings
- **Serper**: Web search API

## How It Works

1. **Query Processing**: User submits a query
2. **Cache Check**: System checks semantic cache for similar queries
3. **Document Search**: If cache miss, searches local PDF documents using embeddings
4. **Routing Decision**: LLM determines if local documents contain sufficient information
5. **Answer Generation**:
   - If local: Generates answer from document context
   - If not: Deploys web scraping agents to find information online
6. **Caching**: Stores query-answer pairs with embeddings for future use

## Configuration

Key parameters in `pipeline/agentic_rag.py`:

- `max_cache_size = 20`: Maximum cached queries (adjustable)
- `similarity_threshold = 0.85`: Cache hit threshold
- `chunk_size = 1000`: Document chunk size for embeddings
- `chunk_overlap = 200`: Overlap between chunks

## Contributing

The base for the RAG-Agent idea was taken from https://www.datacamp.com/tutorial/agentic-rag-tutorial
