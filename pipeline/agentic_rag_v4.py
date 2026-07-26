"""agentic_rag_v4.py — fully agentic RAG, self-contained (no LangGraph, no cross-file imports).

Both layers are driven by the model:

1. A top-level orchestrator agent (`RAGAgent`) is given two tools — search_documents (local
   FAISS) and research_web (a web sub-agent, used AS a tool) — and decides the retrieval
   strategy at run time: search the docs, judge whether they answer the question, escalate to
   the web only if needed, then answer. There is no hardcoded router and no graph.
2. The web step is itself an agent (`WebResearchAgent`): search -> scrape -> submit_findings,
   a while-loop over the model with a terminal tool. It is delegated to as a tool — an agent
   nested inside another agent.

Model strategy (both matter on free-tier keys):
- The web agent tries gemini-3-flash-preview first; on failure it waits 20s, then 60s, then
  falls back to Groq's gpt-oss-120b. Gemini's free tier is 20 requests/min, which two nested
  agents exhaust quickly, so the escalating backoff + cross-provider fallback keeps it alive.
  gpt-oss-120b is the fallback (not llama-3.3, which emits malformed function calls Groq
  rejects); it does tool-calling cleanly.
- The orchestrator runs on gpt-oss-120b, off Gemini, so its calls never compete for Gemini's
  request quota.
- Tool outputs are truncated (scraped pages are huge) so a request stays under Groq's
  free-tier 8000 tokens/min ceiling.

The retry + fallback live in `ResilientChat.invoke`, so the agent loop that calls it stays a
plain while.

Run: `python pipeline/agentic_rag_v4.py` (from the repo root) or import process_query.
"""

import os
import time
from abc import ABC
from abc import abstractmethod

import numpy as np
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.messages import AIMessage
from langchain_core.messages import BaseMessage
from langchain_core.messages import HumanMessage
from langchain_core.messages import SystemMessage
from langchain_core.messages import ToolCall
from langchain_core.messages import ToolMessage
from langchain_core.messages import trim_messages
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.runnables import Runnable
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

web_llm = ChatLiteLLM(
    model="gemini/gemini-3-flash-preview",
    temperature=0.3,
    max_tokens=2000,
    timeout=None,
    max_retries=0,
    api_key=GEMINI_API_KEY,
)

web_backup_llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.3,
    max_tokens=2000,
    timeout=None,
    max_retries=0,
    api_key=GROQ_API_KEY,
)

orchestrator_llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
    max_tokens=1500,
    timeout=None,
    max_retries=0,
    api_key=GROQ_API_KEY,
)


class ResilientChat:
    """A primary chat model with a retry schedule and an optional backup it falls back to.

    On each failed primary `invoke`, waits the next interval in `retry_waits` and retries the
    primary; rate limits clear only after a wait, so an immediate retry is useless — the waits
    are the point. Once the schedule is spent, it switches to the backup and STAYS there for
    the rest of this instance's life (sticky) — otherwise a primary that is down for the whole
    run would re-incur the full wait schedule on every turn. The backup gets its own immediate
    retries to ride out stochastic failures (e.g. a model hallucinating a tool name Groq 400s
    on). Retry and fallback live here, so the agent loop that calls invoke() stays a plain
    while.
    """

    def __init__(
        self,
        primary: Runnable,
        backup: Runnable | None = None,
        retry_waits: tuple[int, ...] = (5, 15, 30),
        backup_retries: int = 2,
    ):
        self.primary = primary
        self.backup = backup
        self.retry_waits = retry_waits
        self.backup_retries = backup_retries
        self._use_backup = False

    def invoke(self, messages: list[BaseMessage]) -> AIMessage:
        last_error: Exception | None = None
        if not self._use_backup:
            for i in range(len(self.retry_waits) + 1):
                try:
                    return self.primary.invoke(messages)
                except Exception as e:
                    last_error = e
                    if i < len(self.retry_waits):
                        wait = self.retry_waits[i]
                        print(f"Primary model failed ({e.__class__.__name__}); retry {i + 1}/{len(self.retry_waits)} in {wait}s")
                        time.sleep(wait)
            if self.backup is None:
                raise last_error
            print("Primary model exhausted; switching to backup for the rest of this run")
            self._use_backup = True
        for _ in range(self.backup_retries + 1):
            try:
                return self.backup.invoke(messages)
            except Exception as e:
                last_error = e
        raise last_error


class BaseAgent(ABC):
    """A ReAct agent is a loop, not a graph.

    Invoke the model, run whatever tools it asked for, append the results, repeat — until the
    model calls a terminal tool (one named in `final_tool_names`) or the iteration budget runs
    out. State lives on `self`, so there is no state dict to thread between nodes and no edges
    to wire: the control flow is the `while` below. Resilience (retry, backoff, backup model)
    lives in `self.llm`, a ResilientChat, keeping this loop clean.
    """

    def __init__(
        self,
        llm: Runnable,
        tools: list[BaseTool],
        system_prompt: str,
        final_tool_names: list[str],
        max_iter: int = 6,
        backup_llm: Runnable | None = None,
        retry_waits: tuple[int, ...] = (5, 15, 30),
        max_history_tokens: int = 4000,
    ) -> None:
        primary = llm.bind_tools(tools)
        backup = backup_llm.bind_tools(tools) if backup_llm is not None else None
        self.llm = ResilientChat(primary, backup, retry_waits)
        self.tools = {tool.name: tool for tool in tools}
        self.final_tool_names = final_tool_names
        self.max_iter = max_iter
        self.max_history_tokens = max_history_tokens
        self.messages: list[BaseMessage] = [SystemMessage(content=system_prompt)]
        self._final_payload: str | None = None

    def run(self, user_message: str) -> str:
        self.messages.append(HumanMessage(content=user_message))
        n_iter = 0
        done = False
        while not done and n_iter < self.max_iter:
            resp = self.llm.invoke(self._windowed_messages())
            self.messages.append(resp)
            done = self._run_tools(resp.tool_calls) if resp.tool_calls else True
            n_iter += 1
        return self._process_output()

    def _windowed_messages(self) -> list[BaseMessage]:
        """Sliding-window view of the history actually sent to the model: the system prompt plus
        the most recent messages that fit in `max_history_tokens`, dropping older turns.
        `self.messages` keeps the full record — only what is *sent* is bounded, so token cost and
        context stay flat as a conversation grows. `start_on='human'` keeps whole turns, so an
        AIMessage's tool calls are never split from their ToolMessage results (which a provider
        rejects as an orphaned pair).

        Guard: if even the latest turn is larger than the budget, `trim_messages` would drop it
        and leave only the system prompt — an invalid request (nothing for the model to answer).
        In that case we fall back to the system prompt plus the whole current turn, uncut."""
        windowed = trim_messages(
            self.messages,
            max_tokens=self.max_history_tokens,
            strategy="last",
            token_counter=count_tokens_approximately,
            include_system=True,
            start_on="human",
            allow_partial=False,
        )
        if any(isinstance(message, HumanMessage) for message in windowed):
            return windowed
        last_human = max(i for i, message in enumerate(self.messages) if isinstance(message, HumanMessage))
        return [self.messages[0], *self.messages[last_human:]]

    def _run_tools(self, tool_calls: list[ToolCall]) -> bool:
        """Execute each requested tool, appending its ToolMessage. Return True once a terminal
        tool has fired so the loop can stop."""
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


_MAX_SEARCH_CHARS = 2000
_MAX_SCRAPE_CHARS = 3000
_MAX_DOCS_CHARS = 4000


class _WebSearchInput(BaseModel):
    search_query: str = Field(description="What to search the web for.")


class _ScrapeInput(BaseModel):
    website_url: str = Field(description="URL of a promising page to fetch and read.")


class _SubmitFindingsInput(BaseModel):
    summary: str = Field(
        description="A thorough, self-contained summary of what you found that answers the question."
    )


_BROWSER_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/122.0 Safari/537.36 agentic-rag"
)


def _web_search(search_query: str) -> str:
    """Google search via the Serper API (POST) — returns the top organic results as text."""
    response = requests.post(
        "https://google.serper.dev/search",
        headers={"X-API-KEY": SERPER_API_KEY or "", "Content-Type": "application/json"},
        json={"q": search_query},
        timeout=20,
    )
    response.raise_for_status()
    results = response.json().get("organic", [])
    rendered = [f"{r.get('title', '')} — {r.get('link', '')}\n{r.get('snippet', '')}" for r in results[:6]]
    return ("\n\n".join(rendered) or "No results found.")[:_MAX_SEARCH_CHARS]


def _web_scrape(website_url: str) -> str:
    """Fetch a page and extract its visible text (scripts/nav/boilerplate stripped)."""
    response = requests.get(website_url, headers={"User-Agent": _BROWSER_USER_AGENT}, timeout=30)
    response.raise_for_status()
    soup = BeautifulSoup(response.content, "html.parser")
    for tag in soup(["script", "style", "nav", "header", "footer", "noscript"]):
        tag.decompose()
    return soup.get_text(" ", strip=True)[:_MAX_SCRAPE_CHARS]


def _submit_findings(summary: str) -> str:
    return summary


def build_web_tools() -> list[BaseTool]:
    """Expose the search/scrape functions plus the terminal submit_findings tool as LangChain
    tools. Search and scrape outputs are truncated so accumulated tool results stay under
    free-tier token-per-minute ceilings (a full scraped page alone can blow past Groq's 8000 TPM)."""
    return [
        StructuredTool.from_function(
            func=_web_search,
            name="web_search",
            description="Search the web (Google via Serper) and return result snippets with links.",
            args_schema=_WebSearchInput,
        ),
        StructuredTool.from_function(
            func=_web_scrape,
            name="web_scrape",
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
2. Call web_scrape on the most promising result(s) to read the actual content.
3. When you have enough information, call submit_findings with a thorough, self-contained summary
   that directly answers the question, citing the key facts you found.
Always finish by calling submit_findings — do not answer in plain text."""


class WebResearchAgent(BaseAgent):
    """Bounded search -> scrape -> summarize loop behind the submit_findings terminal tool.

    Tries Gemini first, waits 20s then 60s on failure, then falls back to Groq's gpt-oss-120b —
    keeping web research alive through Gemini's tight free-tier request quota."""

    def __init__(self) -> None:
        super().__init__(
            llm=web_llm,
            tools=build_web_tools(),
            system_prompt=_WEB_RESEARCH_SYSTEM_PROMPT,
            final_tool_names=["submit_findings"],
            max_iter=6,
            backup_llm=web_backup_llm,
            retry_waits=(20, 60),
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


class _SearchDocsInput(BaseModel):
    query: str = Field(description="A focused search query for the local document knowledge base.")


class _ResearchWebInput(BaseModel):
    query: str = Field(description="The question to research on the web.")


_RAG_SYSTEM_PROMPT = """You are a helpful research assistant having an ongoing conversation with the user.

You have two tools:
- search_documents: search a local PDF knowledge base.
- research_web: delegate to a web-research agent that searches and scrapes the internet (slower — use only when needed).

How to work:
1. ALWAYS call search_documents first to see what the local knowledge base contains.
2. Judge for yourself whether those passages actually answer the question. If they do, answer from them.
3. Only if the documents are insufficient or off-topic, call research_web to gather the information online.
4. When you have enough information, write your final answer as a normal message with no tool call.

This is a conversation: earlier questions and answers stay in context, so handle follow-ups naturally and reuse what
you already found instead of re-researching it. Base your answers strictly on what the tools returned, and be accurate
and concise. End every turn by briefly asking the user whether they would like more detail or have a follow-up question,
then stop and wait for their reply — do not keep working or call more tools until they respond."""


class RAGAgent(BaseAgent):
    """Top-level orchestrator: decides whether to answer from local docs or escalate to the web.

    Replaces a hardcoded router + branch. The loop ends when the model stops calling tools and
    returns the answer, so no terminal tool is needed (`final_tool_names` is empty)."""

    def __init__(self, vector_db: FAISS) -> None:
        self.vector_db = vector_db
        super().__init__(
            llm=orchestrator_llm,
            tools=self._build_tools(),
            system_prompt=_RAG_SYSTEM_PROMPT,
            final_tool_names=[],
            max_iter=8,
        )

    def _process_output(self) -> str:
        """Return the model's answer: the most recent AIMessage that carries actual text.

        Scanning for non-empty text (rather than the very last AIMessage) matters when the loop
        exhausts max_iter mid-tool-call — e.g. a tool kept failing — where the last AIMessage is
        a tool-call turn with empty content. Falling back to a message beats returning ''."""
        for message in reversed(self.messages):
            if isinstance(message, AIMessage) and str(message.content).strip():
                return str(message.content)
        return "I could not find enough information to answer that question."

    def _build_tools(self) -> list[BaseTool]:
        return [
            StructuredTool.from_function(
                func=self._search_documents,
                name="search_documents",
                description="Search the local PDF knowledge base and return the most relevant passages. Call this first.",
                args_schema=_SearchDocsInput,
            ),
            StructuredTool.from_function(
                func=self._research_web,
                name="research_web",
                description="Delegate to a web-research agent that searches and scrapes the internet. Use only when the local documents lack the answer.",
                args_schema=_ResearchWebInput,
            ),
        ]

    def _search_documents(self, query: str) -> str:
        docs = self.vector_db.similarity_search(query, k=5)
        if not docs:
            return "No relevant passages found in the local documents."
        return "\n\n".join(doc.page_content for doc in docs)[:_MAX_DOCS_CHARS]

    def _research_web(self, query: str) -> str:
        return WebResearchAgent().run(f"Research and answer this question: {query}")


def process_query(query: str, vector_db: FAISS | None = None) -> str:
    """Answer a query by letting the orchestrator agent decide how to retrieve.

    The cache is a deterministic fast-path around the agent; everything else — search local,
    judge sufficiency, escalate to web, answer — is the model's call."""
    print(f"Processing query: {query}")

    if (cached_answer := check_cache(query)) is not None:
        print("Cache hit")
        return cached_answer

    if vector_db is None:
        vector_db = initialize_vectorstore()

    answer = RAGAgent(vector_db).run(query)
    update_cache(query, answer)
    return answer


def chat(documents_directory: str = "documents") -> None:
    """Interactive, event-driven assistant.

    One RAGAgent persists for the whole session, so its message history carries across turns:
    the model keeps the full conversation in context, reuses earlier findings, and — per the
    system prompt — ends each turn by checking whether you want more. The loop is driven by
    your input events; it answers one message, then waits for the next."""
    vector_db = initialize_vectorstore(documents_directory)
    agent = RAGAgent(vector_db)
    print("Assistant ready. Ask a question, or type 'exit' to quit.")
    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            break
        if not user_input:
            continue
        print(f"\nAssistant: {agent.run(user_input)}")


def main():
    """Answer a single hardcoded query (non-interactive)."""
    vector_db = initialize_vectorstore("documents")
    query = "What is Agentic RAG?"
    answer = process_query(query, vector_db)
    print(f"Answer: {answer}")


if __name__ == "__main__":
    chat()
