"""agentic_rag_v3.py — Agentic RAG without LangGraph.

Two lessons drive this rewrite:

1. The outer RAG flow is a DETERMINISTIC pipeline, not an agent. cache -> retrieve ->
   route -> (local | web) -> generate is straight-line control flow, so it is a plain
   function (`process_query`), not a StateGraph. v2's graph was a linear pipeline wearing
   a graph costume — the nodes/edges/conditional-edges ceremony added indirection without
   adding branching that plain `if` statements can't express more clearly.

2. The web-research step IS a genuine ReAct agent (search -> scrape -> decide when done).
   That, and only that, gets the agent treatment: a small `while` loop over `llm.invoke()`
   with the tools bound to the model, terminated by a terminal tool (`submit_findings`)
   instead of routed graph edges: `bind_tools` attaches the tools, `final_tool_names` is the
   exit condition, and all state lives on `self`.

Provider note: v2 ran the web agent on gemini/gemini-2.5-flash, which Google now gates off
for new keys (404). v3 points the web agent at gemini-flash-latest — an alias that always
tracks Google's current flash model, so it won't deprecate out from under us (a pinned
gemini-3-flash-preview is heavily rate-limited on free-tier keys). We keep the original
two-model split — Groq llama for document reasoning, Gemini for web research. The document
path never calls tools, so it stays on llama; the agent needs reliable tool-calling, which
the Gemini flash models provide and llama-3.3 on Groq does not (it emits malformed function
calls the API rejects). Caveat: Gemini's free tier is ~5 requests/min per model, which a
multi-step agent can exhaust (429) — see v4 for a Groq fallback that rides that out.
"""

import os
import time
from abc import ABC
from abc import abstractmethod

import numpy as np
from crewai_tools import ScrapeWebsiteTool
from crewai_tools import SerperDevTool
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.messages import BaseMessage
from langchain_core.messages import HumanMessage
from langchain_core.messages import SystemMessage
from langchain_core.messages import ToolCall
from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool
from langchain_core.tools import StructuredTool
from langchain_groq import ChatGroq
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_litellm import ChatLiteLLM
from langchain_text_splitters import RecursiveCharacterTextSplitter
from numpy import dot
from numpy.linalg import norm
from pydantic import BaseModel
from pydantic import Field

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

global_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    max_tokens=500,
    timeout=None,
    max_retries=2,
    api_key=GROQ_API_KEY,
)

web_llm = ChatLiteLLM(
    model="gemini/gemini-flash-latest",
    temperature=0.3,
    max_tokens=2000,
    timeout=None,
    max_retries=2,
    api_key=GEMINI_API_KEY,
)


class BaseAgent(ABC):
    """A ReAct agent is a loop, not a graph.

    Invoke the model, run whatever tools it asked for, append the results, repeat — until
    the model calls a terminal tool (one named in `final_tool_names`) or the iteration
    budget runs out. State lives on `self`, so there is no state dict to thread between
    nodes and no edges to wire: the control flow is the `while` below. `bind_tools` in
    __init__ attaches the tools to the model so it can emit tool calls.
    """

    def __init__(
        self,
        llm: BaseChatModel,
        tools: list[BaseTool],
        system_prompt: str,
        final_tool_names: list[str],
        max_iter: int = 6,
        max_retries: int = 3,
    ) -> None:
        self.llm = llm.bind_tools(tools)
        self.tools = {tool.name: tool for tool in tools}
        self.final_tool_names = final_tool_names
        self.max_iter = max_iter
        self.max_retries = max_retries
        self.messages: list[BaseMessage] = [SystemMessage(content=system_prompt)]
        self._final_payload: str | None = None

    def run(self, user_message: str) -> str:
        """Drive the loop until a terminal tool fires or the iteration budget is spent.

        The model call is retried with exponential backoff because model calls fail
        transiently — a provider rate limit (which clears only after a wait, so an immediate
        retry is useless), or a malformed tool-call the API rejects.
        """
        self.messages.append(HumanMessage(content=user_message))
        n_iter = 0
        n_retries = 0
        done = False
        while not done and n_iter < self.max_iter:
            try:
                resp = self.llm.invoke(self.messages)
            except Exception as e:
                if n_retries >= self.max_retries:
                    raise
                n_retries += 1
                wait_secs = min(5 * 2 ** (n_retries - 1), 30)
                print(f"LLM call failed ({e.__class__.__name__}), retry {n_retries}/{self.max_retries} in {wait_secs}s")
                time.sleep(wait_secs)
                continue
            n_retries = 0
            self.messages.append(resp)
            done = self._run_tools(resp.tool_calls) if resp.tool_calls else True
            n_iter += 1
        return self._process_output()

    def _run_tools(self, tool_calls: list[ToolCall]) -> bool:
        """Execute each requested tool, appending its ToolMessage. Return True once a
        terminal tool has fired so the loop can stop."""
        reached_final = False
        for tool_call in tool_calls:
            selected_tool = self.tools.get(tool_call["name"])
            if selected_tool is None:
                self.messages.append(
                    ToolMessage(
                        content=f'Unknown tool {tool_call["name"]}; pick one of {list(self.tools)}.',
                        tool_call_id=tool_call["id"],
                        name=tool_call["name"],
                    )
                )
                continue
            try:
                tool_msg = selected_tool.invoke(tool_call)
            except Exception as e:
                tool_msg = ToolMessage(
                    content=f'Error calling {tool_call["name"]}: {e}',
                    tool_call_id=tool_call["id"],
                    name=tool_call["name"],
                    status="error",
                )
            self.messages.append(tool_msg)
            if tool_call["name"] in self.final_tool_names:
                self._final_payload = str(tool_msg.content)
                reached_final = True
        return reached_final

    @abstractmethod
    def _process_output(self) -> str:
        raise NotImplementedError


class _WebSearchInput(BaseModel):
    search_query: str = Field(description="What to search the web for.")


class _ScrapeInput(BaseModel):
    website_url: str = Field(description="URL of a promising page to fetch and read.")


class _SubmitFindingsInput(BaseModel):
    summary: str = Field(
        description="A thorough, self-contained summary of what you found that answers the question."
    )


_serper = SerperDevTool()
_scraper = ScrapeWebsiteTool()


def _web_search(search_query: str) -> str:
    return str(_serper.run(search_query=search_query))


def _scrape_website(website_url: str) -> str:
    return str(_scraper.run(website_url=website_url))


def _submit_findings(summary: str) -> str:
    return summary


def build_web_tools() -> list[BaseTool]:
    """Wrap the CrewAI search/scrape tools as plain LangChain tools, plus the terminal
    `submit_findings` tool the agent calls to end its loop."""
    return [
        StructuredTool.from_function(
            func=_web_search,
            name="web_search",
            description="Search the web (Google via Serper) and return result snippets with links.",
            args_schema=_WebSearchInput,
        ),
        StructuredTool.from_function(
            func=_scrape_website,
            name="scrape_website",
            description="Fetch a URL and return its extracted text. Use on the most relevant search result.",
            args_schema=_ScrapeInput,
        ),
        StructuredTool.from_function(
            func=_submit_findings,
            name="submit_findings",
            description="Return your final researched summary. Call this exactly once, when you have enough info.",
            args_schema=_SubmitFindingsInput,
        ),
    ]


_WEB_RESEARCH_SYSTEM_PROMPT = """You are an expert web research assistant. For the user's question:
1. Call web_search to find relevant pages.
2. Call scrape_website on the most promising result(s) to read the actual content.
3. When you have enough information, call submit_findings with a thorough, self-contained summary
   that directly answers the question, citing the key facts you found.
Always finish by calling submit_findings — do not answer in plain text."""


class WebResearchAgent(BaseAgent):
    """Bounded search -> scrape -> summarize loop behind the `submit_findings` terminal tool."""

    def __init__(self) -> None:
        super().__init__(
            llm=web_llm,
            tools=build_web_tools(),
            system_prompt=_WEB_RESEARCH_SYSTEM_PROMPT,
            final_tool_names=["submit_findings"],
            max_iter=6,
        )

    def _process_output(self) -> str:
        if self._final_payload is not None:
            return self._final_payload
        last_ai = next((m for m in reversed(self.messages) if isinstance(m, AIMessage)), None)
        return str(last_ai.content) if last_ai else "No web findings."


query_cache: list[tuple[list[float], str, str]] = []
max_cache_size = 20
similarity_threshold = 0.85


def cosine_similarity(query_embedding, cached_embedding):
    return dot(query_embedding, cached_embedding) / (norm(query_embedding) * norm(cached_embedding))


def check_cache(query: str) -> str | None:
    """Return a cached answer for a sufficiently similar past query, else None."""
    if not query_cache:
        return None
    embedded_query = global_embeddings.embed_query(query)
    scores = [cosine_similarity(embedded_query, item[0]) for item in query_cache]
    if np.max(scores) > similarity_threshold:
        return query_cache[int(np.argmax(scores))][2]
    return None


def update_cache(query: str, answer: str) -> None:
    query_cache.append((global_embeddings.embed_query(query), query, answer))
    if len(query_cache) > max_cache_size:
        query_cache.pop(0)


_ROUTER_PROMPT = """Role: Question-Answering Assistant
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
"""


def retrieve_local_context(query: str, vector_db: FAISS) -> str:
    local_content = vector_db.similarity_search(query, k=5)
    return " ".join(doc.page_content for doc in local_content)


def can_answer_locally(query: str, local_context: str) -> bool:
    response = llm.invoke(_ROUTER_PROMPT.format(text=local_context, query=query))
    return response.content.strip().lower() == "yes"


def research_web(query: str) -> str:
    """Run the web-research agent for one query and return its findings."""
    return WebResearchAgent().run(f"Research and answer this question: {query}")


def generate_final_answer(context: str, query: str) -> str:
    messages = [
        SystemMessage(content="You are a helpful assistant. Use the provided context to answer the query accurately."),
        SystemMessage(content=f"Context: {context}"),
        HumanMessage(content=query),
    ]
    return llm.invoke(messages).content


def load_all_documents(documents_directory="documents"):
    """Load all PDF documents from the directory"""
    all_documents = []
    for file in os.listdir(documents_directory):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(documents_directory, file))
            all_documents.extend(loader.load())
    return all_documents


def create_vector_database(documents):
    """Create a vector store from the documents"""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    return FAISS.from_documents(chunks, global_embeddings)


def initialize_vectorstore(documents_directory="documents"):
    """Load all documents and create a single vectorstore"""
    all_documents = load_all_documents(documents_directory)
    if not all_documents:
        raise ValueError(f"No PDF documents found in the documents directory: {documents_directory}")
    return create_vector_database(all_documents)


def process_query(query: str, vector_db: FAISS | None = None) -> str:
    """Answer a query: cache -> retrieve -> route -> (local | web agent) -> generate."""
    print(f"Processing query: {query}")

    if (cached_answer := check_cache(query)) is not None:
        print("Cache hit")
        return cached_answer

    if vector_db is None:
        vector_db = initialize_vectorstore()

    local_context = retrieve_local_context(query, vector_db)
    if can_answer_locally(query, local_context):
        print("Answering from local documents")
        context = local_context
    else:
        print("Escalating to the web-research agent")
        context = research_web(query)

    answer = generate_final_answer(context, query)
    update_cache(query, answer)
    return answer


def main():
    """Main function to run the RAG pipeline"""
    documents_directory = "documents"
    vector_db = initialize_vectorstore(documents_directory)

    query = "What is Agentic RAG?"
    # query = "What is the highest city in Colombia?"

    answer = process_query(query, vector_db)
    print(f"Answer: {answer}")


if __name__ == "__main__":
    main()
