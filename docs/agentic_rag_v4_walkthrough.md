# `agentic_rag_v4.py` — Complete Code Walkthrough

> A line-by-line explanation of the fully agentic RAG pipeline, written so you can re-read it
> months later and immediately understand *what* every piece does and *why* it exists.

---

## 0. The one-paragraph mental model

This is a **question-answering assistant** over your PDF documents that can fall back to the
**web** when the docs don't have the answer. What makes it *agentic* (not a fixed pipeline) is
that **a language model decides the control flow at run time**: it chooses whether to search the
docs, whether they're good enough, whether to go to the web, and when to answer. There is **no
graph** and **no hardcoded router** — just a `while` loop that hands the model tools and does
what it asks, until it stops asking.

There are **two agents**, one nested inside the other:

```
RAGAgent (orchestrator)          ← decides: docs vs web, and writes the final answer
  ├── tool: search_documents      → FAISS similarity search over your PDFs (not an agent)
  └── tool: research_web           → runs WebResearchAgent (an agent, used as a tool)
        WebResearchAgent
          ├── tool: web_search      → Google via Serper
          ├── tool: web_scrape      → fetch + extract a page's text
          └── tool: submit_findings → TERMINAL tool: ends the web loop with a summary
```

Both agents share one engine (`BaseAgent`) and one resilience layer (`ResilientChat`).

---

## 1. Two ideas you must hold in your head

### 1a. "Agentic" means the model owns the control flow

- **Non-agentic (a pipeline):** *you* wrote the order of steps. The model is called like a
  function — it returns text, and your code decides what happens next. A router that outputs
  "yes/no" and feeds an `if` is still non-agentic: it's a *learned `if`*, evaluated once.
- **Agentic (a loop):** the model's output *is a decision that steers execution* — a tool call
  the runtime dispatches, whose result feeds back so the model decides again, looping an unknown
  number of times until it decides to stop. **The number and order of steps isn't known until
  the model runs.**

`RAGAgent` is agentic: it decides retrieval strategy live. `search_documents` (a FAISS lookup)
and the cache are *not* agentic — they're plain functions the agent calls.

### 1b. Two ways a loop can end (this trips everyone up)

Both agents run the *same* loop, but they **terminate differently**:

| Agent | `final_tool_names` | How its loop ends |
|-------|--------------------|-------------------|
| `WebResearchAgent` | `["submit_findings"]` | The model calls the **terminal tool** `submit_findings`. Its argument becomes the return value. |
| `RAGAgent` | `[]` (empty) | The model **stops calling tools and just writes text**. That text is the answer. |

So: **the web sub-agent ends with a final tool call; the orchestrator ends by answering in
plain text.** Why the difference?

- The web agent needs to hand a *clean, structured string* back to its caller (the summary), so
  we force it through a named tool (`submit_findings`) whose argument we capture.
- The orchestrator is talking to a *human*. The natural end of an assistant's turn is "it says
  the answer." Requiring a tool call there would be artificial. In the loop, "no tool calls
  this turn" is the terminal condition (see `BaseAgent.run`).

This is why the orchestrator prompt says *"write your final answer as a normal message with no
tool call"* while the web prompt says *"Always finish by calling submit_findings."*

---

## 2. Line-by-line walkthrough

### 2.1 Module docstring (lines 1–28)

```python
"""agentic_rag_v4.py — fully agentic RAG, self-contained (no LangGraph, no cross-file imports).
...
"""
```
Pure documentation. The important claims: **self-contained** (no imports from other project
files, so you can copy this one file into a new repo), and the **model strategy** (Gemini for
web with a Groq fallback; Groq for the orchestrator; truncated tool outputs). Nothing executes.

### 2.2 Imports (lines 30–57)

```python
import os
import time
from abc import ABC
from abc import abstractmethod
```
- `os` — read environment variables and list the documents directory.
- `time` — `time.sleep(...)` for the retry backoff in `ResilientChat`.
- `ABC`, `abstractmethod` — make `BaseAgent` an *abstract base class* (see §2.4). `ABC` is a
  Python built-in that marks a class as not-directly-instantiable; `@abstractmethod` marks a
  method that subclasses **must** implement.

```python
import numpy as np
from numpy import dot
from numpy.linalg import norm
```
Numpy, used only by the semantic cache to compute cosine similarity between query embeddings.

```python
from crewai_tools import ScrapeWebsiteTool
from crewai_tools import SerperDevTool
```
The actual web-search (`SerperDevTool`, Google via the Serper API) and page-scraping
(`ScrapeWebsiteTool`) implementations. We wrap these in LangChain tools later.

```python
from dotenv import load_dotenv
```
Loads `.env` into environment variables so `os.getenv(...)` finds your API keys.

```python
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
```
- `PyPDFLoader` — reads a PDF into LangChain `Document` objects.
- `FAISS` — an in-memory vector store for similarity search over document chunks.

```python
from langchain_core.messages import AIMessage
from langchain_core.messages import BaseMessage
from langchain_core.messages import HumanMessage
from langchain_core.messages import SystemMessage
from langchain_core.messages import ToolCall
from langchain_core.messages import ToolMessage
from langchain_core.messages import trim_messages
from langchain_core.messages.utils import count_tokens_approximately
```
The message types that make up a conversation with a chat model:
- `SystemMessage` — instructions/persona (the system prompt).
- `HumanMessage` — a user turn.
- `AIMessage` — a model turn. May contain text (`.content`) and/or tool calls (`.tool_calls`).
- `ToolMessage` — the *result* of running a tool, fed back to the model. Carries a
  `tool_call_id` that links it to the `AIMessage`'s tool call.
- `ToolCall` — the typed dict shape of a single tool call (`{"name","args","id","type"}`).
- `BaseMessage` — the common base type, used for type hints on `list[BaseMessage]`.
- `trim_messages` — LangChain's utility to shrink a message list to a token budget. Used to
  implement the sliding window (§2.5); it knows how to keep the system message and avoid
  splitting a tool call from its result.
- `count_tokens_approximately` — a fast, tokenizer-free token estimate used as `trim_messages`'
  token counter (no model call, no download).

```python
from langchain_core.runnables import Runnable
```
`Runnable` is LangChain's universal "callable component" interface — anything with `.invoke()`.
Used only as a type hint here (chat models and `bind_tools(...)` results are `Runnable`s).

```python
from langchain_core.tools import BaseTool
from langchain_core.tools import StructuredTool
```
- `BaseTool` — the base class of all LangChain tools. Used as a **type hint** (`list[BaseTool]`).
- `StructuredTool` — a concrete tool built from a Python function with a typed argument schema.
  `StructuredTool.from_function(...)` is how we actually create every tool. (See §3 for why we
  use this instead of the `@tool` decorator.)

```python
from langchain_groq import ChatGroq
from langchain_litellm import ChatLiteLLM
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
```
- `ChatGroq` — chat model client for Groq (runs `gpt-oss-120b`).
- `ChatLiteLLM` — a chat model client that speaks to many providers via LiteLLM; here it talks
  to Google Gemini.
- `HuggingFaceEmbeddings` — turns text into vectors (for FAISS and the cache).
- `RecursiveCharacterTextSplitter` — splits long documents into overlapping chunks.

```python
from pydantic import BaseModel
from pydantic import Field
```
Used to declare each tool's **argument schema** (e.g. "this tool takes a `search_query: str`").
The model reads these schemas to know how to call the tool.

### 2.3 Configuration and models (lines 59–92)

```python
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")
```
Load `.env`, then read the three keys. (`SERPER_API_KEY` isn't passed explicitly anywhere — the
CrewAI Serper tool reads it from the environment itself — but it's surfaced here for clarity.)

```python
global_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
```
One shared embedding model, created once at import. Used both to embed document chunks (for
FAISS) and to embed queries (for the semantic cache). `all-MiniLM-L6-v2` is a small, fast,
384-dimensional sentence-embedding model that downloads on first use.

```python
web_llm = ChatLiteLLM(
    model="gemini/gemini-3-flash-preview",
    temperature=0.3,
    max_tokens=2000,
    timeout=None,
    max_retries=0,
    api_key=GEMINI_API_KEY,
)
```
The **primary** model for the web research agent — Gemini. `temperature=0.3` gives slightly
creative-but-focused research. `max_tokens=2000` bounds each response. **`max_retries=0` is
deliberate**: we do *not* want LiteLLM's own hidden retry loop, because `ResilientChat` owns all
retry/backoff logic. If the client retried too, our clean 20s/60s schedule would be muddied.

```python
web_backup_llm = ChatGroq(
    model="openai/gpt-oss-120b",
    ...
    max_retries=0,
    api_key=GROQ_API_KEY,
)
```
The **fallback** model for web research — Groq's `gpt-oss-120b`. Chosen because it does
tool-calling cleanly (unlike `llama-3.3-70b` on Groq, which emits a malformed `<function=…>`
format that Groq's API rejects). Again `max_retries=0` — `ResilientChat` is the single retry
authority.

```python
orchestrator_llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
    max_tokens=1500,
    ...
)
```
The model for the top-level `RAGAgent`. `temperature=0` for deterministic routing decisions. It
runs on Groq, deliberately **off Gemini**, so the orchestrator's calls never eat into Gemini's
tight free-tier request quota (which the web agent needs).

### 2.4 `ResilientChat` — the resilience layer (lines 95–142)

This class wraps a model so the agent loop above it doesn't have to think about failures.

```python
class ResilientChat:
    def __init__(self, primary, backup=None, retry_waits=(5, 15, 30), backup_retries=2):
        self.primary = primary
        self.backup = backup
        self.retry_waits = retry_waits
        self.backup_retries = backup_retries
        self._use_backup = False
```
- `primary` — the main model (already tool-bound; see §2.5).
- `backup` — an optional second model to switch to when the primary is exhausted.
- `retry_waits` — the wait schedule **between primary attempts**. `(20, 60)` for the web agent
  means: try, wait 20s, try, wait 60s, try. Then give up on primary.
- `backup_retries` — how many *extra* immediate attempts to give the backup (for stochastic
  failures like a hallucinated tool name).
- `_use_backup` — the **sticky flag**. Once we fall back, this becomes `True` and stays `True`,
  so future turns skip the primary entirely (see below for why).

```python
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
```
The primary path. `range(len(retry_waits) + 1)` = one initial attempt plus one per wait. On each
failure we remember the error; if there's still a wait left, we sleep and loop. `time.sleep`
matters because rate limits **only clear after real time passes** — an immediate retry is
useless. If any attempt succeeds, we `return` immediately (an `AIMessage`).

```python
            if self.backup is None:
                raise last_error
            print("Primary model exhausted; switching to backup for the rest of this run")
            self._use_backup = True
```
If the primary is exhausted and there's **no** backup, re-raise the last error (fail loudly). If
there **is** a backup, flip the sticky flag. **Why sticky?** If the primary (Gemini) is
rate-limited for the whole run, and we re-tried it every turn, every turn would pay the full
`20+60 = 80s` of waits before falling back. Sticky means we pay that **once**, then use Groq for
the rest of this agent's run.

```python
        for _ in range(self.backup_retries + 1):
            try:
                return self.backup.invoke(messages)
            except Exception as e:
                last_error = e
        raise last_error
```
The backup path (reached either by falling through above, or directly on later turns once
`_use_backup` is `True`). It tries the backup up to `backup_retries + 1` times with **no waits**
(these failures are stochastic, not rate limits). If all fail, re-raise.

> **Design note:** this mirrors what production LLM clients do — a retrying, model-falling-back
> "chain" — but kept tiny and readable. Because all of this lives in `invoke`, the agent loop
> that calls `self.llm.invoke(...)` never sees a failure and stays a plain `while`.

### 2.5 `BaseAgent` — the agent engine (lines 145–217)

```python
class BaseAgent(ABC):
```
`ABC` = **Abstract Base Class**. This makes `BaseAgent` a *template* you must subclass — you
cannot do `BaseAgent(...)` directly. Combined with the `@abstractmethod` on `_process_output`
below, Python raises `TypeError: Can't instantiate abstract class` if a subclass forgets to
implement that method. It turns "please remember to implement this" into an enforced contract.

```python
    def __init__(self, llm, tools, system_prompt, final_tool_names,
                 max_iter=6, backup_llm=None, retry_waits=(5, 15, 30),
                 max_history_tokens=4000):
        primary = llm.bind_tools(tools)
        backup = backup_llm.bind_tools(tools) if backup_llm is not None else None
        self.llm = ResilientChat(primary, backup, retry_waits)
```
`llm.bind_tools(tools)` is the key line. `llm` is a LangChain chat model (`ChatGroq` or
`ChatLiteLLM`), both subclasses of `BaseChatModel` where `bind_tools` is defined. It **does not
call the model** — it returns a *new* `Runnable` with the tools' JSON schemas attached, so that a
later `.invoke(messages)` lets the model emit tool calls. We bind the same tools to the backup,
then wrap both in a `ResilientChat`. From here on, `self.llm.invoke(...)` means
`ResilientChat.invoke` — **our** retry/fallback wrapper, *not* a raw model call.

```python
        self.tools = {tool.name: tool for tool in tools}
        self.final_tool_names = final_tool_names
        self.max_iter = max_iter
        self.max_history_tokens = max_history_tokens
        self.messages: list[BaseMessage] = [SystemMessage(content=system_prompt)]
        self._final_payload: str | None = None
```
- `self.tools` — a name→tool lookup, used to dispatch a tool call to the right object.
- `self.final_tool_names` — which tool(s), if called, end the loop (empty = "end when the model
  stops calling tools"; see §1b).
- `self.max_iter` — a hard cap on loop iterations, so a misbehaving model can't loop forever.
- `self.max_history_tokens` — the token budget for the sliding window (§2.5.1): the most the
  message history may occupy in a single model call before older turns are dropped.
- `self.messages` — **the entire conversation state**, seeded with the system prompt. Everything
  the model says and every tool result is appended here. This is the *full* record; the sliding
  window only bounds what is *sent* per call, not what is stored.
- `self._final_payload` — where a terminal tool's result is stashed (used by `WebResearchAgent`).

```python
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
```
**This is the whole agent.** Step by step:
1. Append the user's message to the history.
2. Loop, bounded by `max_iter`:
   - `self.llm.invoke(self._windowed_messages())` — send a **token-bounded sliding window** of
     the history to the model (via `ResilientChat`), get back an `AIMessage` (`resp`). Note we
     send `_windowed_messages()`, *not* `self.messages` — the full history is kept on `self`, but
     only a recent window is sent (§2.5.1).
   - Append `resp` to the **full** history (`self.messages`).
   - **Branch on whether the model called tools:**
     - If `resp.tool_calls` is non-empty → run them via `_run_tools(...)`, which returns `True`
       only if a *terminal* tool fired. So `done` = "did a terminal tool fire?"
     - If there are **no** tool calls → the model answered in plain text → `done = True`. This is
       the orchestrator's normal exit.
3. When the loop ends, `_process_output()` extracts the answer.

> Note: because `run` *appends* to `self.messages`, calling `agent.run(...)` again on the **same
> instance** continues the conversation with full history. That's exactly how the interactive
> `chat()` and the FastAPI `/chat` endpoint get multi-turn memory — one agent, many `run` calls.

#### 2.5.1 `_windowed_messages` — the sliding window (context management)

Without this, `self.messages` grows forever and the **whole** history is re-sent on every model
call — cost and latency climb each turn, and eventually you blow past the model's context window
or a per-minute token limit (we hit Groq's 8000 tokens/min this way). This method bounds what is
*sent* while keeping the full record on `self`.

A **sliding window** is one trimming strategy: keep the system prompt + the most recent messages,
drop the oldest. (The other common strategy is *summarization/compaction* — replace old turns
with an LLM-written summary. We use the simpler window.)

```python
    def _windowed_messages(self) -> list[BaseMessage]:
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
```

`trim_messages` (a LangChain utility) does the trimming; the parameters are the whole story:
- `max_tokens=self.max_history_tokens` — the budget (default 4000). Counted with
  `count_tokens_approximately` (`token_counter=...`), a fast tokenizer-free estimate.
- `strategy="last"` — keep the **most recent** messages (that's what makes it a *sliding* window
  rather than keeping the oldest).
- `include_system=True` — always keep the leading `SystemMessage` (the agent's instructions),
  even though it's the oldest message. It never gets trimmed away.
- `start_on="human"` — **the correctness lynchpin.** The kept window must begin on a
  `HumanMessage`. This guarantees we never start on a dangling `ToolMessage` or an `AIMessage`
  whose tool calls were dropped — a tool call and its result must always travel together, or the
  provider rejects the request (an "orphaned tool call"). Starting on a human message means we
  only ever cut at clean turn boundaries.
- `allow_partial=False` — don't keep a *fragment* of a message; keep whole messages only.

**The guard (last three lines).** `trim_messages` has one failure mode we must handle: if even
the *latest* turn is bigger than the budget (very possible for the web agent, whose single turn
can include several scraped pages), it drops that turn too and returns just `[SystemMessage]` —
which is an invalid request (nothing for the model to answer). So: if the trimmed window contains
no `HumanMessage`, we fall back to "system prompt + the entire current turn" (everything from the
last `HumanMessage` onward), uncut. Better to exceed the budget than to send a broken request.

**Where trimming actually bites:** because `start_on="human"` keeps whole turns, and a single
`run` call has exactly one `HumanMessage` at its start, trimming never splits an in-progress turn
— it only drops **older, completed turns**. So it does nothing within one `WebResearchAgent` run
(one turn), and does its real work across the **many turns** of a `chat()` / `/chat` conversation,
dropping the oldest exchanges once history exceeds the budget. `self.messages` still holds
everything; `_process_output` reads the full record; only the per-call payload is bounded.

```python
    def _run_tools(self, tool_calls: list[ToolCall]) -> bool:
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
```
For each tool the model asked for, look it up by name. If it named a tool that doesn't exist
(models occasionally hallucinate names), we don't crash — we append an error `ToolMessage`
telling the model the valid options, and move on. The `tool_call_id` **must** match the model's
call id, or the provider rejects the next request (every tool call needs exactly one result).

```python
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
```
**`selected_tool.invoke(tool_call)` runs the tool.** This is `BaseTool.invoke` (LangChain), a
*different method from `ResilientChat.invoke`* that happens to share the name. It validates the
call's `args` against the tool's schema, runs the underlying Python function, and returns a
`ToolMessage`. If the tool raises, we convert the exception into an error `ToolMessage` (so the
model sees "that failed" and can adapt) instead of crashing the loop. Either way, the result is
appended to the history.

> **The two `.invoke`s, side by side** — this is the single most confusing part:
> - `self.llm.invoke(messages)` → `ResilientChat.invoke` → an `AIMessage` (a *model* call).
> - `selected_tool.invoke(tool_call)` → `BaseTool.invoke` → a `ToolMessage` (a *tool* call).
>
> They connect only *transitively*: when the tool is `research_web`, its function spins up a
> `WebResearchAgent`, whose own loop calls *its* `ResilientChat.invoke`. So a `tool.invoke` reaches
> a `ResilientChat.invoke` only because that particular tool is itself an agent.

```python
            if tool_call["name"] in self.final_tool_names:
                self._final_payload = str(tool_msg.content)
                reached_final = True
        return reached_final
```
After running a tool, check whether it's a **terminal** tool. If so, remember its output as the
final payload and mark `reached_final = True`. The method returns that flag, which becomes `done`
in `run`. For the orchestrator (`final_tool_names=[]`) this is never true, so its loop only ends
via the "no tool calls" branch.

```python
    @abstractmethod
    def _process_output(self) -> str:
        raise NotImplementedError
```
The one method every subclass **must** provide: how to turn the finished conversation into the
string returned by `run`. Declaring it `@abstractmethod` is what makes `BaseAgent` abstract.

### 2.6 Web tools (lines 220–279)

```python
_MAX_SEARCH_CHARS = 2000
_MAX_SCRAPE_CHARS = 3000
_MAX_DOCS_CHARS = 4000
```
Truncation caps. Tool outputs are appended to `self.messages` and re-sent every turn, so an
untruncated scraped page (tens of thousands of characters) would blow past model token limits
(we hit Groq's 8000-tokens/minute ceiling before adding these). These caps bound each tool
result's size.

```python
class _WebSearchInput(BaseModel):
    search_query: str = Field(description="What to search the web for.")

class _ScrapeInput(BaseModel):
    website_url: str = Field(description="URL of a promising page to fetch and read.")

class _SubmitFindingsInput(BaseModel):
    summary: str = Field(description="A thorough, self-contained summary ...")
```
Each tool's **argument schema**. The field name (`search_query`, `website_url`, `summary`) is the
argument the model must supply, and the `description` tells the model what to put there. These
schemas are what the model sees when deciding how to call a tool.

```python
_serper = SerperDevTool()
_scraper = ScrapeWebsiteTool()

def _web_search(search_query: str) -> str:
    return str(_serper.run(search_query=search_query))[:_MAX_SEARCH_CHARS]

def _web_scrape(website_url: str) -> str:
    return str(_scraper.run(website_url=website_url))[:_MAX_SCRAPE_CHARS]

def _submit_findings(summary: str) -> str:
    return summary
```
The actual tool implementations:
- `_web_search` calls the CrewAI Serper tool and truncates the result.
- `_web_scrape` fetches a URL's text and truncates it.
- `_submit_findings` is the **terminal tool's function**. It just returns the `summary` the model
  passed. That return value becomes the `ToolMessage.content`, which `_run_tools` stashes into
  `_final_payload` — i.e. the model *writes* the final research summary as this tool's argument.

```python
def build_web_tools() -> list[BaseTool]:
    return [
        StructuredTool.from_function(
            func=_web_search,
            name="web_search",
            description="Search the web (Google via Serper) and return result snippets with links.",
            args_schema=_WebSearchInput,
        ),
        StructuredTool.from_function(func=_web_scrape, name="web_scrape", ...),
        StructuredTool.from_function(func=_submit_findings, name="submit_findings", ...),
    ]
```
Wraps each function as a LangChain `StructuredTool`, giving it the `name` the model will call, a
`description` (the model's only clue about *when* to use it), and the `args_schema` from above.
The tool is named `web_scrape` (not `scrape_website`) on purpose: the model pattern-matches
`web_search` → `web_scrape`, so matching that instinct reduces tool-name hallucinations.

### 2.7 `WebResearchAgent` (lines 282–311)

```python
_WEB_RESEARCH_SYSTEM_PROMPT = """You are an expert web research assistant. ...
Always finish by calling submit_findings — do not answer in plain text."""
```
The web agent's instructions. Note the last line: it's told to **always** end with the terminal
tool, because this agent hands a string back to code, not to a human.

```python
class WebResearchAgent(BaseAgent):
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
```
Configures the shared engine for web research: Gemini primary, Groq backup, the **`(20, 60)`
wait schedule** you asked for, and `submit_findings` as the terminal tool. `max_iter=6` bounds
how many search/scrape rounds it can do.

```python
    def _process_output(self) -> str:
        if self._final_payload is not None:
            return self._final_payload
        last_ai = next((m for m in reversed(self.messages) if isinstance(m, AIMessage)), None)
        return str(last_ai.content) if last_ai else "No web findings."
```
How the web agent produces its return value: if `submit_findings` fired, return its payload
(the normal case). Otherwise (e.g. the model hit `max_iter` without finishing), fall back to the
last thing the model said, or a placeholder. This makes a stalled loop degrade gracefully
instead of crashing.

### 2.8 Semantic cache (lines 314–337)

```python
query_cache: list[tuple[list[float], str, str]] = []
max_cache_size = 20
similarity_threshold = 0.85

def cosine_similarity(query_embedding, cached_embedding):
    return dot(query_embedding, cached_embedding) / (norm(query_embedding) * norm(cached_embedding))
```
A tiny in-memory cache of `(embedding, query, answer)` triples. `cosine_similarity` measures how
close two embedding vectors are (1.0 = identical direction).

```python
def check_cache(query: str) -> str | None:
    if not query_cache:
        return None
    embedded_query = global_embeddings.embed_query(query)
    scores = [cosine_similarity(embedded_query, item[0]) for item in query_cache]
    if np.max(scores) > similarity_threshold:
        return query_cache[int(np.argmax(scores))][2]
    return None
```
Embed the incoming query, compare to every cached query, and if the closest one is above `0.85`
similarity, return that cached **answer** (`item[2]`). This lets a near-duplicate question skip
the whole agent. Returns `None` on a miss.

```python
def update_cache(query: str, answer: str) -> None:
    query_cache.append((global_embeddings.embed_query(query), query, answer))
    if len(query_cache) > max_cache_size:
        query_cache.pop(0)
```
Store a new `(embedding, query, answer)` triple and evict the oldest if over capacity (a simple
FIFO). **This is answer-level caching, not context management** — it doesn't bound the size of a
running conversation; it just short-circuits repeat questions in `process_query`.

### 2.9 Vector store (lines 340–362)

```python
def load_all_documents(documents_directory="documents"):
    all_documents = []
    for file in os.listdir(documents_directory):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(documents_directory, file))
            all_documents.extend(loader.load())
    return all_documents
```
Read every PDF in the folder into LangChain `Document` objects (one per page, roughly).

```python
def create_vector_database(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    return FAISS.from_documents(chunks, global_embeddings)
```
Split documents into ~1000-character chunks (200 chars of overlap so a sentence split across a
boundary isn't lost), embed each chunk, and index them in FAISS for similarity search.

```python
def initialize_vectorstore(documents_directory="documents"):
    all_documents = load_all_documents(documents_directory)
    if not all_documents:
        raise ValueError(f"No PDF documents found ...")
    return create_vector_database(all_documents)
```
The one call the app makes at startup: load + chunk + embed + index, returning a ready FAISS
store. Raises early if the folder has no PDFs.

### 2.10 `RAGAgent` — the orchestrator (lines 365–441)

```python
class _SearchDocsInput(BaseModel):
    query: str = Field(description="A focused search query for the local document knowledge base.")

class _ResearchWebInput(BaseModel):
    query: str = Field(description="The question to research on the web.")
```
Argument schemas for the orchestrator's two tools.

```python
_RAG_SYSTEM_PROMPT = """You are a helpful research assistant having an ongoing conversation ...
1. ALWAYS call search_documents first ...
2. Judge for yourself whether those passages actually answer the question ...
3. Only if the documents are insufficient or off-topic, call research_web ...
4. When you have enough information, write your final answer as a normal message with no tool call.
...
End every turn by briefly asking the user whether they would like more detail ...
then stop and wait for their reply ..."""
```
The orchestrator's brain. This prompt is what replaces the old hardcoded router: **the model
itself** decides sufficiency (step 2) and whether to escalate (step 3). Step 4 defines the
terminal condition (answer in plain text = loop ends). The final paragraph makes it
conversational and event-driven — it ends each turn by checking in and *waiting*, instead of
barreling ahead.

```python
class RAGAgent(BaseAgent):
    def __init__(self, vector_db: FAISS) -> None:
        self.vector_db = vector_db
        super().__init__(
            llm=orchestrator_llm,
            tools=self._build_tools(),
            system_prompt=_RAG_SYSTEM_PROMPT,
            final_tool_names=[],
            max_iter=8,
        )
```
Store the FAISS store on `self` (the `search_documents` tool needs it), then configure the
engine: Groq model, **`final_tool_names=[]`** (ends by answering in text), `max_iter=8`. Note
`self.vector_db` is set **before** `super().__init__`, because `_build_tools()` (called inside
`super().__init__`) references it.

```python
    def _process_output(self) -> str:
        for message in reversed(self.messages):
            if isinstance(message, AIMessage) and str(message.content).strip():
                return str(message.content)
        return "I could not find enough information to answer that question."
```
Return the most recent AI message that has **non-empty text**. Why not just the last message? If
the loop exhausted `max_iter` while the model was mid-tool-call, the last `AIMessage` might be a
tool-call turn with empty text — returning that would give the user `""`. Scanning backward for
real text (with a graceful fallback string) avoids that.

```python
    def _build_tools(self) -> list[BaseTool]:
        return [
            StructuredTool.from_function(
                func=self._search_documents,
                name="search_documents",
                description="Search the local PDF knowledge base ... Call this first.",
                args_schema=_SearchDocsInput,
            ),
            StructuredTool.from_function(
                func=self._research_web,
                name="research_web",
                description="Delegate to a web-research agent ... Use only when the local documents lack the answer.",
                args_schema=_ResearchWebInput,
            ),
        ]
```
Builds the orchestrator's two tools. Crucially, `func=self._search_documents` and
`func=self._research_web` are **bound methods** — they carry `self`, so the tools can reach
`self.vector_db` and construct sub-agents. This is *why* we build tools here at instance-creation
time rather than declaring them at class level (see §3).

```python
    def _search_documents(self, query: str) -> str:
        docs = self.vector_db.similarity_search(query, k=5)
        if not docs:
            return "No relevant passages found in the local documents."
        return "\n\n".join(doc.page_content for doc in docs)[:_MAX_DOCS_CHARS]
```
The `search_documents` tool: FAISS returns the 5 most similar chunks; we join their text and
truncate. **No model here** — this is a plain retrieval, the non-agentic part.

```python
    def _research_web(self, query: str) -> str:
        return WebResearchAgent().run(f"Research and answer this question: {query}")
```
The `research_web` tool: construct a **fresh** `WebResearchAgent` and run it. This is the
nesting — an agent invoked as a tool. A fresh instance per call means each web research starts
with clean history and a fresh `ResilientChat` (fresh sticky-fallback state).

### 2.11 Entry points (lines 444–491)

```python
def process_query(query: str, vector_db: FAISS | None = None) -> str:
    print(f"Processing query: {query}")
    if (cached_answer := check_cache(query)) is not None:
        print("Cache hit")
        return cached_answer
    if vector_db is None:
        vector_db = initialize_vectorstore()
    answer = RAGAgent(vector_db).run(query)
    update_cache(query, answer)
    return answer
```
The **stateless one-shot** entry point (used by the FastAPI `/query` endpoint). Check the cache;
on a miss, build the vector store if needed, run a **fresh** `RAGAgent` for this one query, cache
the answer, return it. Fresh agent per call = no memory between calls (that's what `/chat` and
`chat()` are for).

```python
def chat(documents_directory: str = "documents") -> None:
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
```
The **interactive, multi-turn** entry point. One `RAGAgent` persists for the whole session, so
`agent.run(user_input)` on each line keeps appending to the same `self.messages` — full context
across turns. The loop is driven by your `input()` events: answer one message, wait for the next.
This is the "feels like a real assistant" experience.

```python
def main():
    vector_db = initialize_vectorstore("documents")
    query = "What is Agentic RAG?"
    answer = process_query(query, vector_db)
    print(f"Answer: {answer}")

if __name__ == "__main__":
    chat()
```
`main()` is a non-interactive one-shot (handy for quick tests). Running the file directly launches
`chat()` — the interactive assistant.

---

## 3. Why `StructuredTool.from_function` instead of the `@tool` decorator?

Both produce a `StructuredTool` (a subclass of `BaseTool`); `@tool` is just a shortcut decorator.
We use `StructuredTool.from_function(...)` for two reasons:

1. **Our orchestrator tools are instance methods.** `@tool` is applied at *class-definition*
   time, when there's no `self` yet — you can't cleanly decorate `self._search_documents` so it
   closes over `self.vector_db`. `StructuredTool.from_function(func=self._search_documents, ...)`
   is called inside `_build_tools()` at *instance-construction* time, capturing the **bound**
   method. This is the crux: a tool that needs instance state (the vector store, the ability to
   spawn a sub-agent) must be built per instance.
2. **Explicit control.** `from_function` lets us pass `name`, `description`, and `args_schema`
   directly, so the names the model sees (`web_scrape`, `research_web`) and the wording of each
   description are exactly what we choose — independent of the Python function name.

`BaseTool` itself is never instantiated; it's only used as the **type hint** (`list[BaseTool]`)
because it's the common base class of every tool, including `StructuredTool`.

---

## 4. End-to-end execution traces

### 4a. A question the documents can answer

```
process_query("What are governance challenges of generative AI?")
  ├─ check_cache → miss
  ├─ RAGAgent.run(...)
  │    turn 1: llm.invoke → model calls search_documents("governance challenges ...")
  │            search_documents.invoke → 5 FAISS chunks appended
  │    turn 2: llm.invoke → model judges chunks sufficient → answers in plain text (no tool call)
  │            → done (no tool calls)
  │    _process_output → the answer text
  └─ update_cache; return answer
```

### 4b. A question needing the web (with Gemini rate-limited)

```
RAGAgent.run("What is the highest city in Colombia?")
  turn 1: search_documents → irrelevant chunks
  turn 2: model calls research_web("highest city in Colombia")
          research_web.invoke → WebResearchAgent().run(...)
              inner turn 1: llm.invoke → ResilientChat:
                  Gemini fails (429) → wait 20s → Gemini fails → wait 60s → Gemini fails
                  → switch to Groq (sticky) → Groq calls web_search
              inner turns 2..n: web_scrape / web_search on Groq
              inner turn n: model calls submit_findings("... Vetas ...")
                  → terminal tool → inner loop ends
              _process_output → the findings summary
          findings appended as a ToolMessage
  turn 3: model writes the final answer in plain text → done
  _process_output → the answer
```

---

## 5. Known limitations (so future-you isn't surprised)

- **Context management is a sliding window, not summarization.** `_windowed_messages` (§2.5.1)
  bounds what's sent per call, so cost/latency stay flat as a conversation grows — but a window
  *forgets*: once history exceeds `max_history_tokens`, the oldest turns are dropped and the
  assistant can no longer reference them. For long sessions where old context still matters, the
  next step up is summarization/compaction (replace old turns with an LLM summary) rather than
  dropping them. Also, `self.messages` still holds the full history in memory (only the *sent*
  payload is bounded), so process memory still grows across a very long session.
- **Answer accuracy is not guaranteed.** The architecture is sound, but research quality varies
  run to run — the web agent may scrape a weak source. No verification/reranking step exists.
- **Free-tier ceilings shape behavior.** Gemini's ~20 requests/min and Groq's ~8000 tokens/min
  drove the fallback design and the truncation caps. On paid tiers you could raise `max_iter`,
  loosen truncation, and shorten `retry_waits`.
- **`chat()`/`/chat` session state is in-memory, single-process.** Fine for one worker; a
  multi-worker deployment needs a shared store (e.g. Redis) keyed by conversation id.

---

## 6. The serving layer (`serving_api/main_v4.py`)

A FastAPI app exposing the pipeline over HTTP.

FastAPI is **headless** — there is no web page to open in a browser (that's what the Streamlit
app is for). You talk to it with `curl` (POST requests) or the auto-generated Swagger UI.

- **Startup (`lifespan`)** builds the FAISS store once and marks the app `ready`.
- **`GET /`** returns a small JSON index of the endpoints, so a browser hitting `/` gets
  something useful instead of a bare 404. (The chat/query endpoints are POST, so a browser GET
  can't reach them directly.)
- **`GET /docs`** — FastAPI's built-in interactive Swagger UI; the easiest way to click-test the
  POST endpoints without curl.
- **`GET /health`** returns 200 when ready.
- **`POST /chat`** — the conversational endpoint. Body `{message, conversation_id?}`. If no
  `conversation_id`, it mints one (`uuid4`) and creates a fresh `RAGAgent` for it; otherwise it
  reuses the stored agent so history carries across turns. A per-conversation `asyncio.Lock`
  serializes turns *within* one conversation (so two rapid requests don't corrupt one agent's
  history), while different conversations still run concurrently. The blocking `agent.run` is
  offloaded with `asyncio.to_thread` so it doesn't block the event loop. Returns
  `{conversation_id, answer}`.
- **`POST /query`** — the stateless one-shot (wraps `process_query`).
- **Error handling** — `except HTTPException: raise` comes *before* the generic `except
  Exception`, so a deliberate 400 (missing field) is returned as a real 400 instead of being
  swallowed and re-raised as a 500.

Run it with:
```bash
fastapi run serving_api/main_v4.py --host 0.0.0.0 --port 8000
```
Then:
```bash
# start a conversation
curl -s localhost:8000/chat -H 'content-type: application/json' \
     -d '{"message":"What are governance challenges of generative AI?"}'
# -> {"conversation_id":"<id>","answer":"..."}

# continue it (pass the id back)
curl -s localhost:8000/chat -H 'content-type: application/json' \
     -d '{"conversation_id":"<id>","message":"Expand on the first one."}'
```
