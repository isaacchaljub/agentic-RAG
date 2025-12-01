import os
from typing import TypedDict, Annotated, Literal, Optional
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langchain_litellm import ChatLiteLLM
from crewai_tools import SerperDevTool, ScrapeWebsiteTool
from numpy.linalg import norm
from numpy import dot
import numpy as np

# Load the environment variables
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

global_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Initialize LLMs
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    max_tokens=500,
    timeout=None,
    max_retries=2,
    api_key=GROQ_API_KEY,
)

# Define custom AgentState for RAG pipeline
class AgentState(TypedDict):
    """Custom state for the RAG pipeline workflow"""
    query: str
    local_context: str
    web_context: str
    context: str
    can_answer_locally: bool
    answer: str
    cache_hit: bool
    vector_db: Optional[FAISS]  # Optional, can be None

# LLM for web agent tasks (using LiteLLM with Gemini)
web_llm = ChatLiteLLM(
    model="gemini/gemini-2.5-flash",
    temperature=0.7,
    max_tokens=2000,
    timeout=None,
    max_retries=2,
    api_key=GEMINI_API_KEY,  # Pass directly, but LiteLLM will also check env
)

# Semantic caching
query_cache = []
max_cache_size = 20
similarity_threshold = 0.85


def cosine_similarity(query_embedding, cached_embedding):
    """Calculate cosine similarity between two embeddings"""
    return dot(query_embedding, cached_embedding) / (norm(query_embedding) * norm(cached_embedding))


def check_cache(state: AgentState) -> AgentState:
    """Check if query is in semantic cache"""
    query = state["query"]
    
    if len(query_cache) == 0:
        return {**state, "cache_hit": False}
    
    embedded_query = global_embeddings.embed_query(query)
    similarity_scores = [cosine_similarity(embedded_query, cached_item[0]) for cached_item in query_cache]
    
    if np.max(similarity_scores) > similarity_threshold:
        cached_answer = query_cache[np.argmax(similarity_scores)][2]
        return {
            **state,
            "cache_hit": True,
            "answer": cached_answer
        }
    
    return {**state, "cache_hit": False}


def search_local_documents(state: AgentState) -> AgentState:
    """Search local vector database for relevant documents"""
    query = state["query"]
    vector_db = state.get("vector_db")
    
    if vector_db is None:
        # Initialize if not provided
        vector_db = initialize_vectorstore()
    
    local_content = vector_db.similarity_search(query, k=5)
    local_context = " ".join([doc.page_content for doc in local_content])
    
    return {**state, "local_context": local_context}


def router_decision(state: AgentState) -> AgentState:
    """Use LLM to determine if we can answer from local documents"""
    query = state["query"]
    local_context = state["local_context"]
    
    prompt = '''Role: Question-Answering Assistant
Task: Determine whether the system can answer the user's question based on the provided text.
Instructions:
    - Analyze the text and identify if it contains the necessary information to answer the user's question.
    - Provide a clear and concise response indicating whether the system can answer the question or not.
    - Your response should include only a single word. Nothing else, no other text, information, header/footer. 
Output Format:
    - Answer: Yes/No
Study the below examples and based on that, respond to the last question. 
Examples:
    Input: 
        Text: The capital of France is Paris.
        User Question: What is the capital of France?
    Expected Output:
        Answer: Yes
    Input: 
        Text: The population of the United States is over 330 million.
        User Question: What is the population of China?
    Expected Output:
        Answer: No
    Input:
        User Question: {query}
        Text: {text}
'''
    formatted_prompt = prompt.format(text=local_context, query=query)
    response = llm.invoke(formatted_prompt)
    can_answer_locally = response.content.strip().lower() == "yes"
    
    return {**state, "can_answer_locally": can_answer_locally}


def should_use_web(state: AgentState) -> Literal["use_local", "use_web"]:
    """Conditional edge function to decide routing"""
    if state.get("can_answer_locally", False):
        return "use_local"
    return "use_web"


def use_local_context(state: AgentState) -> AgentState:
    """Use local context for answer generation"""
    return {**state, "context": state["local_context"]}


# Global web agent (created once and reused)
_web_agent = None

def get_web_agent():
    """Get or create the LangChain agent using create_agent"""
    global _web_agent
    if _web_agent is None:
        tools = [SerperDevTool(), ScrapeWebsiteTool()]
        
        # Create the agent using langchain.agents.create_agent
        _web_agent = create_agent(
            model=web_llm,
            tools=tools,
            system_prompt="""You are an expert web research assistant. Your task is to:
1. Search the web for relevant information about the user's query
2. Scrape and analyze the most relevant web pages
3. Provide a comprehensive summary of the findings

Use the available tools to search and scrape websites. Be thorough and accurate."""
        )
    return _web_agent


def search_web(state: AgentState) -> AgentState:
    """Search the web for information using LangChain agent"""
    query = state["query"]
    
    # Get the web agent
    agent = get_web_agent()
    
    # Create a message from the query
    user_message = HumanMessage(
        content=f"Search the web and find comprehensive information about: {query}"
    )
    
    # Invoke the agent with messages (AgentState expects "messages" key)
    result = agent.invoke({"messages": [user_message]})
    
    # Extract the final message content from the agent's response
    # The agent returns state with "messages" containing the conversation
    messages = result.get("messages", [])
    if messages:
        # Get the last message (agent's response)
        last_message = messages[-1]
        web_context = last_message.content if hasattr(last_message, 'content') else str(last_message)
    else:
        web_context = "No response from web agent"
    
    return {**state, "web_context": web_context, "context": web_context}


def generate_answer(state: AgentState) -> AgentState:
    """Generate final answer using LLM"""
    context = state["context"]
    query = state["query"]
    
    messages = [
        SystemMessage(content="You are a helpful assistant. Use the provided context to answer the query accurately."),
        SystemMessage(content=f"Context: {context}"),
        HumanMessage(content=query),
    ]
    
    response = llm.invoke(messages)
    answer = response.content
    
    # Update cache
    embedded_query = global_embeddings.embed_query(query)
    query_cache.append((embedded_query, query, answer))
    
    # Prune cache if too large
    if len(query_cache) > max_cache_size:
        query_cache.pop(0)
    
    return {**state, "answer": answer}


# Load all documents from the documents directory
def load_all_documents(documents_directory="documents"):
    """Load all PDF documents from the directory"""
    all_documents = []
    for file in os.listdir(documents_directory):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(documents_directory, file))
            documents = loader.load()
            all_documents.extend(documents)
    return all_documents


# Create the vector store from the documents
def create_vector_database(documents):
    """Create a vector store from the documents"""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)

    # Create the embeddings
    embeddings = global_embeddings
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore


# Initialize the vectorstore (call this once, then reuse it)
def initialize_vectorstore(documents_directory="documents"):
    """Load all documents and create a single vectorstore"""
    all_documents = load_all_documents(documents_directory)
    if not all_documents:
        raise ValueError(f"No PDF documents found in the documents directory: {documents_directory}")
    vectorstore = create_vector_database(all_documents)
    return vectorstore


def create_agentic_rag_graph():
    """Create the LangGraph workflow for agentic RAG"""
    
    # Create the graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("check_cache", check_cache)
    workflow.add_node("search_local", search_local_documents)
    workflow.add_node("router", router_decision)
    workflow.add_node("use_local", use_local_context)
    workflow.add_node("search_web", search_web)
    workflow.add_node("generate_answer", generate_answer)
    
    # Set entry point
    workflow.set_entry_point("check_cache")
    
    # Add edges
    workflow.add_conditional_edges(
        "check_cache",
        lambda state: END if state.get("cache_hit", False) else "search_local"
    )
    
    workflow.add_edge("search_local", "router")
    
    workflow.add_conditional_edges(
        "router",
        should_use_web,
        {
            "use_local": "use_local",
            "use_web": "search_web"
        }
    )
    
    workflow.add_edge("use_local", "generate_answer")
    workflow.add_edge("search_web", "generate_answer")
    workflow.add_edge("generate_answer", END)
    
    # Compile the graph
    app = workflow.compile()
    
    return app


# Global graph instance - initialized once at module load (not on first request)
# This is important for API usage - graph is compiled at startup, not on first query
_rag_graph = create_agentic_rag_graph()

def process_query(query: str, vector_db=None):
    """Main function to process user query using LangGraph"""
    print(f"Processing query: {query}")
    
    # Graph is already initialized at module level, just use it
    
    # Initialize state
    initial_state = {
        "query": query,
        "local_context": "",
        "web_context": "",
        "context": "",
        "can_answer_locally": False,
        "answer": "",
        "cache_hit": False,
        "vector_db": vector_db
    }
    
    # Run the graph
    result = _rag_graph.invoke(initial_state)
    
    return result["answer"]


# Main function to run the RAG pipeline
def main():
    """Main function to run the RAG pipeline"""
    # Path to the documents directory
    documents_directory = "documents"

    # Create the vector database
    vector_db = initialize_vectorstore(documents_directory)

    # Define the query
    query = "What is Agentic RAG?"
    # query = "What is the highest city in Colombia?"

    # Answer the query
    answer = process_query(query, vector_db)
    print(f"Answer: {answer}")


if __name__ == "__main__":
    main()

