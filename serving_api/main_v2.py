from fastapi import FastAPI, Request, Response, HTTPException
from ..pipeline.agentic_rag_v2 import process_query, initialize_vectorstore
import asyncio
from logging import getLogger
from contextlib import asynccontextmanager
import os
from pathlib import Path


#Logger for the serving API
logger = getLogger(__name__)

# Lifespan for the serving API
@asynccontextmanager
async def lifespan(app: FastAPI):
    global vector_db, ready
    try:
        vector_db = initialize_vectorstore(str(Path(__file__).parent.parent / 'documents'))
        logger.info("Vector database loaded successfully (graph compiled at module import)")
        ready = True
    except Exception as e:
        ready = False
        logger.error(f"Error initializing vector database: {e}")
        print(f"LIFESPAN ERROR: {e}")
        raise Exception(f"Error initializing vector database: {e}")
        

    yield

    #Shutdown the model
    logger.info("Shutting down vector database")
    vector_db = None
    logger.info("Vector database shut down successfully")

#Create the FastAPI app
app=FastAPI(lifespan=lifespan,
title="Agentic RAG",
description="A simple API for the Agentic RAG pipeline",
version="1.0.0")

@app.get("/health")
async def get_health():
    if ready:
        return Response(content="OK", status_code=200)
    else:
        return Response(content="NOT OK", status_code=500)

@app.post("/query")
async def answer_query(request: Request):
    try:
        data = await request.json()
        query = data.get("query")
        
        if not query:
            raise HTTPException(status_code=400, detail="Missing 'query' in request body")
        
        # Run sync function in thread pool - llm.invoke() blocks even though it's waiting for I/O
        answer = await asyncio.to_thread(process_query, query, vector_db)
        
        return {"answer": answer}
    
    except Exception as e:
        logger.error(f"Error answering query: {e}")
        print(f"QUERY ERROR: {e}")
        raise HTTPException(status_code=500, detail=str(e))



# Handle HTTP exceptions with consistent error format
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with consistent error format"""
    logger.error(f"HTTP exception: {exc.status_code} - {exc.detail}")
    return Response(status_code=exc.status_code, content=f"HTTP error: {exc.status_code} - {exc.detail}")