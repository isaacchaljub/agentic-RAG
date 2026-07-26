"""main_v4.py — chat-enabled FastAPI for the fully agentic RAG pipeline.

Adds multi-turn conversations on top of the v4 agent. Each conversation id maps to its own
persistent RAGAgent, so the model keeps that conversation's history in context across requests
(follow-ups work). A per-conversation lock serializes turns within one conversation while
different conversations still run concurrently.

Endpoints:
- GET  /health              -> readiness
- POST /chat                -> {conversation_id?, message} -> {conversation_id, answer}
- POST /query               -> {query} -> {answer}   (stateless one-shot, no history)

Caveat: conversation state is in-memory, so it lives in a single process. Running multiple
workers would split conversations across them; a shared store (e.g. Redis) is needed for that.
"""

import asyncio
import uuid
from contextlib import asynccontextmanager
from logging import getLogger
from pathlib import Path

from fastapi import FastAPI
from fastapi import HTTPException
from fastapi import Request
from fastapi import Response
from pydantic import BaseModel
from pydantic import Field

from pipeline.agentic_rag_v4 import RAGAgent
from pipeline.agentic_rag_v4 import initialize_vectorstore
from pipeline.agentic_rag_v4 import process_query

logger = getLogger(__name__)

_agents: dict[str, RAGAgent] = {}
_locks: dict[str, asyncio.Lock] = {}


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, description="The user's message for this turn.")
    conversation_id: str | None = Field(
        default=None, description="Omit to start a new conversation; pass the returned id to continue one."
    )


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, description="A one-shot question (no conversation history).")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global vector_db, ready
    try:
        vector_db = initialize_vectorstore(str(Path(__file__).parent.parent / 'documents'))
        logger.info("Vector database loaded successfully")
        ready = True
    except Exception as e:
        ready = False
        logger.error(f"Error initializing vector database: {e}")
        print(f"LIFESPAN ERROR: {e}")
        raise Exception(f"Error initializing vector database: {e}")

    yield

    logger.info("Shutting down vector database")
    _agents.clear()
    _locks.clear()
    vector_db = None
    logger.info("Vector database shut down successfully")


app = FastAPI(
    lifespan=lifespan,
    title="Agentic RAG (chat)",
    description="A conversational API for the fully agentic RAG pipeline",
    version="1.0.0",
)


@app.get("/")
async def root():
    """Index so a browser GET / is not a bare 404. The chat/query endpoints are POST."""
    return {
        "service": "Agentic RAG (chat)",
        "endpoints": {
            "GET /health": "readiness check",
            "POST /chat": "{message, conversation_id?} -> {conversation_id, answer}",
            "POST /query": "{query} -> {answer} (stateless, no history)",
            "GET /docs": "interactive Swagger UI to try the POST endpoints",
        },
    }


@app.get("/health")
async def get_health():
    if ready:
        return Response(content="OK", status_code=200)
    return Response(content="NOT OK", status_code=500)


@app.post("/chat")
async def chat(req: ChatRequest):
    """Continue (or start) a conversation. Omit conversation_id to start a new one; pass the
    returned id on later turns to keep context."""
    try:
        conversation_id = req.conversation_id or str(uuid.uuid4())
        if conversation_id not in _agents:
            _agents[conversation_id] = RAGAgent(vector_db)
        lock = _locks.setdefault(conversation_id, asyncio.Lock())

        async with lock:
            answer = await asyncio.to_thread(_agents[conversation_id].run, req.message)

        return {"conversation_id": conversation_id, "answer": answer}

    except Exception as e:
        logger.error(f"Error in chat: {e}")
        print(f"CHAT ERROR: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query")
async def answer_query(req: QueryRequest):
    """Stateless one-shot query — no conversation history."""
    try:
        answer = await asyncio.to_thread(process_query, req.query, vector_db)
        return {"answer": answer}

    except Exception as e:
        logger.error(f"Error answering query: {e}")
        print(f"QUERY ERROR: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with consistent error format"""
    logger.error(f"HTTP exception: {exc.status_code} - {exc.detail}")
    return Response(status_code=exc.status_code, content=f"HTTP error: {exc.status_code} - {exc.detail}")
